#!/usr/bin/env python3
"""Live RealSense RGB-D → local Ollama Qwen captions (≤8 words).

Designed for a Jetson + D457 (or any D400 RGB-D camera) and the local
qwen3.5:4b vision model. Capture, JPEG encode, and inference overlap so
the on-screen label always uses the newest frame.

JPEG stays on the CPU (libjpeg-turbo via OpenCV, ~0.5–1 ms at VLM size).
Moving encode to the GPU (nvJPEG) does not speed captions: Ollama owns the
GPU for vision tokens, and CPU encode overlaps with the previous request on
the capture thread. Faster captions come from a smaller VLM JPEG (fewer
vision tokens), not from GPU JPEG.

    python3 demo.py
    python3 demo.py --headless
    python3 demo.py --speech
    python3 demo.py --composite   # also show colorized depth to the VLM
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import pyrealsense2 as rs

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_HOST = "http://127.0.0.1:11434"
PROMPT = (
    "Live RGB-D camera. Distances in meters: nearest {near:.1f}, "
    "median {med:.1f}, left/center/right {left:.1f}/{center:.1f}/{right:.1f}. "
    "Name what you see in 8 words or less. No punctuation."
)
PROMPT_NO_DEPTH = "Name what you see in 8 words or less. No punctuation."
PROMPT_COMPOSITE = (
    "RGB left, colorized depth right. Distances m: nearest {near:.1f}, "
    "median {med:.1f}. Name what you see in 8 words or less. No punctuation."
)


@dataclass
class DepthStats:
    near_m: float = 0.0
    median_m: float = 0.0
    left_m: float = 0.0
    center_m: float = 0.0
    right_m: float = 0.0
    valid: bool = False


@dataclass
class FrameBundle:
    color_bgr: np.ndarray
    depth_vis: np.ndarray
    stats: DepthStats
    ts: float
    # Pre-baked RGB JPEG for Ollama (built on the capture thread). None when
    # --composite needs a runtime RGB|depth stitch before encode.
    vlm_jpeg: Optional[bytes] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST))
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=int, default=30)
    # 256 is the sweet spot measured on Thor + qwen3.5:4b: fewer vision tokens
    # than 320 with nearly the same caption quality; 160 barely helps further.
    p.add_argument("--vlm-width", type=int, default=256, help="JPEG width sent to the VLM")
    p.add_argument("--jpeg-quality", type=int, default=65)
    p.add_argument("--composite", action="store_true", help="Send RGB|depth image instead of RGB only")
    p.add_argument("--headless", action="store_true", help="Print captions only, no OpenCV window")
    p.add_argument("--speech", action="store_true", help="Speak each caption with espeak")
    p.add_argument("--espeak", default="espeak", help="espeak binary (default: espeak)")
    p.add_argument("--num-predict", type=int, default=20)
    p.add_argument("--num-ctx", type=int, default=512)
    return p.parse_args()


class Speaker:
    """Voice each caption with espeak without blocking the VLM loop."""

    def __init__(self, enabled: bool, binary: str = "espeak") -> None:
        self.enabled = enabled
        self.binary = binary
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def say(self, text: str) -> None:
        if not self.enabled or not text:
            return
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
            try:
                self._proc = subprocess.Popen(
                    [self.binary, "-s", "160", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                print(f"espeak failed: {exc}", file=sys.stderr)

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
            self._proc = None


def depth_stats(depth_m: np.ndarray) -> DepthStats:
    valid = depth_m[(depth_m > 0.15) & (depth_m < 20.0)]
    if valid.size < 50:
        return DepthStats()
    h, w = depth_m.shape
    thirds = []
    for x0, x1 in ((0, w // 3), (w // 3, 2 * w // 3), (2 * w // 3, w)):
        col = depth_m[:, x0:x1]
        col = col[(col > 0.15) & (col < 20.0)]
        thirds.append(float(np.percentile(col, 10)) if col.size else 0.0)
    return DepthStats(
        near_m=float(np.percentile(valid, 5)),
        median_m=float(np.median(valid)),
        left_m=thirds[0],
        center_m=thirds[1],
        right_m=thirds[2],
        valid=True,
    )


def clamp_words(text: str, n: int = 8) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    words = [w for w in cleaned.split() if w]
    return " ".join(words[:n])


def encode_jpeg(bgr: np.ndarray, width: int, quality: int) -> bytes:
    h, w = bgr.shape[:2]
    if w != width:
        height = max(1, int(round(h * (width / w))))
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def probe_zerocopy(color_frame: rs.video_frame) -> str:
    """Return a short ZERO-COPY / UPLOAD / NONE / n/a label for a color frame."""
    if not hasattr(rs, "gpu_frame") or not hasattr(color_frame, "get_gpu_data_or_upload"):
        return "n/a (no gpu_frame API — install CUDA zero-copy pyrealsense2)"
    gf = color_frame.as_gpu_frame()
    res = color_frame.get_gpu_data_or_upload()
    if res is None:
        return "NONE (non-CUDA build)"
    _addr, copied = res
    if gf and not copied:
        return "ZERO-COPY (gpu_frame device pointer aliases frame)"
    return "UPLOAD (SDK host→device copy)"


class RealSenseGrabber:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        vlm_width: int = 256,
        jpeg_quality: int = 65,
        preencode_jpeg: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.vlm_width = vlm_width
        self.jpeg_quality = jpeg_quality
        # Pre-encode on this thread so the VLM loop never waits on resize/JPEG;
        # leaves the GPU free for Ollama (CPU encode is ~1 ms).
        self.preencode_jpeg = preencode_jpeg
        self._pipe = rs.pipeline()
        self._align = rs.align(rs.stream.color)
        self._lock = threading.Lock()
        self._latest: Optional[FrameBundle] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._depth_scale = 0.001
        self.fps_cam = 0.0
        self.zerocopy_path = "unknown"

    def start(self) -> None:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = self._pipe.start(cfg)
        device = profile.get_device()
        print(f"camera: {device.get_info(rs.camera_info.name)}  "
              f"{device.get_info(rs.camera_info.serial_number)}  "
              f"{self.width}x{self.height}@{self.fps}", flush=True)
        print(f"pyrealsense2: {getattr(rs, '__file__', '?')}  "
              f"v{getattr(rs, '__version__', '?')}", flush=True)
        try:
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        except Exception:
            self._depth_scale = 0.001
        for sensor in device.sensors:
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1)
        # Prime a few frames so auto-exposure settles before the first JPEG.
        color = None
        for _ in range(8):
            frames = self._pipe.wait_for_frames(5000)
            color = frames.get_color_frame()
        if color:
            self.zerocopy_path = probe_zerocopy(color)
        print(f"cuda frame path: {self.zerocopy_path}", flush=True)
        self._thread = threading.Thread(target=self._loop, name="realsense", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        last = time.perf_counter()
        ema = 0.0
        while not self._stop.is_set():
            try:
                frames = self._align.process(self._pipe.wait_for_frames(2000))
            except Exception:
                if self._stop.is_set():
                    break
                continue
            depth_f = frames.get_depth_frame()
            color_f = frames.get_color_frame()
            if not depth_f or not color_f:
                continue
            color = np.asanyarray(color_f.get_data())
            depth_raw = np.asanyarray(depth_f.get_data())
            depth_m = depth_raw.astype(np.float32) * self._depth_scale
            # 0.03 is the usual D400 display scale (~max 8 m → 255).
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_TURBO)
            stats = depth_stats(depth_m)
            # Encode while this thread owns `color` (before the display copy) so
            # the infer loop only pays for Ollama HTTP round-trips.
            vlm_jpeg = (
                encode_jpeg(color, self.vlm_width, self.jpeg_quality)
                if self.preencode_jpeg
                else None
            )
            bundle = FrameBundle(color.copy(), depth_vis, stats, time.time(), vlm_jpeg)
            with self._lock:
                self._latest = bundle
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                inst = 1.0 / dt
                ema = inst if ema == 0 else (0.9 * ema + 0.1 * inst)
                self.fps_cam = ema

    def latest(self) -> Optional[FrameBundle]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self._pipe.stop()
        except Exception:
            pass


class OllamaCaptioner:
    def __init__(self, host: str, model: str, num_predict: int, num_ctx: int) -> None:
        parsed = urlparse(host if "://" in host else f"http://{host}")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 11434
        self.model = model
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self._url = f"http://{self.host}:{self.port}/api/chat"

    def warmup(self) -> None:
        blank = np.full((64, 64, 3), 32, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        print(f"loading {self.model} …", flush=True)
        t0 = time.perf_counter()
        text, _ = self.caption(buf.tobytes(), "Reply with the word ready.")
        print(f"model ready in {time.perf_counter() - t0:.1f}s  ({text or 'ok'})", flush=True)

    def caption(self, jpeg: bytes, prompt: str, on_token=None):
        last_err: Optional[Exception] = None
        for _attempt in range(2):
            try:
                return self._caption_once(jpeg, prompt, on_token)
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
                last_err = exc
                time.sleep(0.05)
        raise last_err  # type: ignore[misc]

    def _caption_once(self, jpeg: bytes, prompt: str, on_token=None):
        payload = {
            "model": self.model,
            "think": False,  # must be top-level for qwen3.5, not inside options
            "stream": True,
            "keep_alive": "24h",
            "options": {
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
                "temperature": 0.1,
                "top_k": 20,
                "top_p": 0.8,
            },
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(jpeg).decode("ascii")],
                }
            ],
        }
        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks: list[str] = []
        thinking: list[str] = []
        metrics = {}
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw in resp:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    msg = event.get("message") or {}
                    if msg.get("content"):
                        chunks.append(msg["content"])
                        if on_token:
                            on_token("".join(chunks))
                    if msg.get("thinking"):
                        thinking.append(msg["thinking"])
                        if on_token and not chunks:
                            on_token("".join(thinking))
                    if event.get("done"):
                        metrics = event
                        break
        except Exception as exc:
            text = "".join(chunks).strip() or "".join(thinking).strip()
            if text:
                return text, metrics
            raise exc
        text = "".join(chunks).strip() or "".join(thinking).strip()
        return text, metrics


class LiveState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.caption = "starting"
        self.latency_s = 0.0
        self.fps_vlm = 0.0
        self.error = ""
        self.running = True


def infer_loop(
    grabber: RealSenseGrabber,
    captioner: OllamaCaptioner,
    state: LiveState,
    args: argparse.Namespace,
    speaker: Speaker,
) -> None:
    ema = 0.0
    while state.running:
        bundle = grabber.latest()
        if bundle is None:
            time.sleep(0.01)
            continue
        if args.composite:
            rgb = cv2.resize(
                bundle.color_bgr,
                (args.vlm_width, int(args.vlm_width * bundle.color_bgr.shape[0] / bundle.color_bgr.shape[1])),
            )
            dep = cv2.resize(bundle.depth_vis, (rgb.shape[1], rgb.shape[0]))
            frame = np.hstack((rgb, dep))
            prompt = (
                PROMPT_COMPOSITE.format(near=bundle.stats.near_m, med=bundle.stats.median_m)
                if bundle.stats.valid
                else PROMPT_NO_DEPTH
            )
            jpeg = encode_jpeg(frame, frame.shape[1], args.jpeg_quality)
        else:
            prompt = (
                PROMPT.format(
                    near=bundle.stats.near_m,
                    med=bundle.stats.median_m,
                    left=bundle.stats.left_m,
                    center=bundle.stats.center_m,
                    right=bundle.stats.right_m,
                )
                if bundle.stats.valid
                else PROMPT_NO_DEPTH
            )
            jpeg = bundle.vlm_jpeg or encode_jpeg(bundle.color_bgr, args.vlm_width, args.jpeg_quality)

        def on_token(text: str) -> None:
            with state.lock:
                state.caption = clamp_words(text) or "…"

        try:
            t0 = time.perf_counter()
            raw, _metrics = captioner.caption(jpeg, prompt, on_token=on_token)
            dt = time.perf_counter() - t0
        except Exception as exc:
            # Keep the last good caption; a stream/keep-alive glitch used to
            # flash "ollama error" over the live label.
            with state.lock:
                state.error = str(exc)
            print(f"\nollama retryable error: {exc}", flush=True)
            time.sleep(0.2)
            continue
        label = clamp_words(raw)
        ema = (1.0 / dt) if ema == 0 else (0.7 * ema + 0.3 / dt)
        with state.lock:
            state.caption = label or "…"
            state.latency_s = dt
            state.fps_vlm = ema
            state.error = ""
        dist = f"{bundle.stats.median_m:.1f}m" if bundle.stats.valid else "n/a"
        line = f"{label:<40}  {dt*1000:5.0f} ms  {dist}"
        if args.headless:
            print(line, flush=True)
        else:
            print(f"\r{line}   ", end="", flush=True)
        speaker.say(label)


def draw_overlay(color: np.ndarray, depth_vis: np.ndarray, bundle: FrameBundle, state: LiveState, cam_fps: float) -> np.ndarray:
    vis = np.hstack((color, depth_vis))
    h, w = vis.shape[:2]
    top_h = 48
    bar_h = max(72, h // 6)
    canvas = np.zeros((top_h + h + bar_h, w, 3), dtype=np.uint8)
    canvas[top_h:top_h + h] = vis
    with state.lock:
        caption = state.caption
        latency = state.latency_s
        vlm_fps = state.fps_vlm
        err = state.error
    font = cv2.FONT_HERSHEY_SIMPLEX
    dist = (
        f"near {bundle.stats.near_m:.1f}m  mid {bundle.stats.median_m:.1f}m  "
        f"L/C/R {bundle.stats.left_m:.1f}/{bundle.stats.center_m:.1f}/{bundle.stats.right_m:.1f}m"
        if bundle.stats.valid
        else "depth n/a"
    )
    hud = f"{dist}   cam {cam_fps:.0f} fps   vlm {vlm_fps:.1f} Hz  {latency*1000:.0f} ms"
    if err:
        hud = err[:80]
    cv2.rectangle(canvas, (0, 0), (w, top_h), (20, 20, 20), -1)
    cv2.putText(canvas, hud, (16, 32), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "RGB", (16, top_h + 28), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "DEPTH", (color.shape[1] + 16, top_h + 28), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, top_h + h), (w, top_h + h + bar_h), (20, 20, 20), -1)
    scale = 1.1
    thickness = 2
    max_w = w - 32
    (tw, th), _ = cv2.getTextSize(caption, font, scale, thickness)
    if tw > max_w and tw > 0:
        scale = max(0.55, scale * max_w / tw)
        (tw, th), _ = cv2.getTextSize(caption, font, scale, thickness)
    x = max(16, (w - tw) // 2)
    y = top_h + h + (bar_h + th) // 2 - 4
    cv2.putText(canvas, caption, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = parse_args()
    state = LiveState()

    def handle_stop(_sig=None, _frame=None) -> None:
        state.running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    speaker = Speaker(args.speech, args.espeak)
    if args.speech:
        if shutil.which(args.espeak) is None:
            print(f"espeak not found ({args.espeak}). Install espeak or pass --espeak /path/to/espeak.", file=sys.stderr)
            return 1
        print(f"speech: {args.espeak}  (each caption)", flush=True)

    grabber = RealSenseGrabber(
        args.width,
        args.height,
        args.fps,
        vlm_width=args.vlm_width,
        jpeg_quality=args.jpeg_quality,
        preencode_jpeg=not args.composite,
    )
    captioner = OllamaCaptioner(args.host, args.model, args.num_predict, args.num_ctx)
    try:
        grabber.start()
    except Exception as exc:
        print(f"failed to start RealSense: {exc}", file=sys.stderr)
        return 1
    try:
        captioner.warmup()
    except Exception as exc:
        grabber.stop()
        print(f"failed to reach Ollama at {args.host}: {exc}", file=sys.stderr)
        print("Is `ollama serve` running, and is the model pulled?", file=sys.stderr)
        return 1

    worker = threading.Thread(
        target=infer_loop, args=(grabber, captioner, state, args, speaker), name="vlm", daemon=True
    )
    worker.start()
    print("captions (q to quit):\n", flush=True)

    if args.headless:
        try:
            while state.running:
                time.sleep(0.1)
        finally:
            print()
            speaker.stop()
            grabber.stop()
        return 0

    win = "RGB-D VLM"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while state.running:
            bundle = grabber.latest()
            if bundle is None:
                if cv2.waitKey(10) in (ord("q"), 27):
                    break
                continue
            canvas = draw_overlay(bundle.color_bgr, bundle.depth_vis, bundle, state, grabber.fps_cam)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except cv2.error as exc:
        print(f"\nOpenCV window failed ({exc}); continuing headless. Ctrl+C to stop.", flush=True)
        while state.running:
            time.sleep(0.1)
    finally:
        state.running = False
        speaker.stop()
        grabber.stop()
        cv2.destroyAllWindows()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
