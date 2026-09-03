#!/usr/bin/env python3
"""Live RealSense RGB-D → CUDA BLIP captions (faster than the Ollama demo).

Leaves demo.py (Ollama / qwen3.5) untouched. This path runs a small vision
captioner on the Jetson GPU in-process — no JPEG, no HTTP.

Measured on AGX Thor (warm):
  Ollama qwen3.5:4b  ~280–350 ms / caption
  BLIP-base (CUDA)   ~80 ms / caption

    source .venv/bin/activate
    python demo_cuda.py
    python demo_cuda.py --headless
    python demo_cuda.py --speech
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Reuse camera / HUD / speech from the Ollama demo without changing it.
from demo import (  # noqa: E402
    LiveState,
    RealSenseGrabber,
    Speaker,
    clamp_words,
    draw_overlay,
)

DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_HF_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("CUDA_VLM_MODEL", DEFAULT_MODEL),
                   help="Hugging Face BLIP caption model id")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--vlm-width", type=int, default=384,
                   help="RGB width fed to BLIP (height scales with aspect)")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--speech", action="store_true")
    p.add_argument("--espeak", default="espeak")
    p.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME),
                   help="Hugging Face cache dir (models download here once)")
    return p.parse_args()


class BlipCudaCaptioner:
    """In-process CUDA BLIP captioner — RGB ndarray in, short text out."""

    def __init__(self, model_id: str, max_new_tokens: int = 12, vlm_width: int = 384) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch — install Jetson CUDA torch wheels")
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = torch.device("cuda")
        self.max_new_tokens = max_new_tokens
        self.vlm_width = vlm_width
        self.model_id = model_id
        print(f"loading {model_id} on {torch.cuda.get_device_name(0)} …", flush=True)
        t0 = time.perf_counter()
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = (
            BlipForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16)
            .to(self.device)
            .eval()
        )
        print(f"model ready in {time.perf_counter() - t0:.1f}s", flush=True)

    def warmup(self) -> None:
        blank = np.full((self.vlm_width, self.vlm_width, 3), 32, dtype=np.uint8)
        t0 = time.perf_counter()
        text = self.caption(blank)
        print(f"warmup caption in {time.perf_counter() - t0:.2f}s  ({text or 'ok'})", flush=True)

    def caption(self, color_bgr: np.ndarray) -> str:
        h, w = color_bgr.shape[:2]
        if w != self.vlm_width:
            nh = max(1, int(round(h * (self.vlm_width / w))))
            color_bgr = cv2.resize(color_bgr, (self.vlm_width, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        # BLIP sometimes prefixes "a photo of …" — strip for the 8-word HUD.
        for prefix in ("a photo of ", "a picture of ", "an image of "):
            if text.lower().startswith(prefix):
                text = text[len(prefix) :]
                break
        return text.strip()


def infer_loop(
    grabber: RealSenseGrabber,
    captioner: BlipCudaCaptioner,
    state: LiveState,
    args: argparse.Namespace,
    speaker: Speaker,
) -> None:
    ema = 0.0
    while state.running:
        bundle = grabber.latest()
        if bundle is None:
            time.sleep(0.005)
            continue
        try:
            t0 = time.perf_counter()
            raw = captioner.caption(bundle.color_bgr)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        except Exception as exc:
            with state.lock:
                state.error = str(exc)
            print(f"\ncuda vlm error: {exc}", flush=True)
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


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.makedirs(args.hf_home, exist_ok=True)

    state = LiveState()

    def handle_stop(_sig=None, _frame=None) -> None:
        state.running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    speaker = Speaker(args.speech, args.espeak)
    if args.speech:
        if shutil.which(args.espeak) is None:
            print(f"espeak not found ({args.espeak}).", file=sys.stderr)
            return 1
        print(f"speech: {args.espeak}", flush=True)

    # No JPEG pre-encode — BLIP takes RGB tensors on GPU.
    grabber = RealSenseGrabber(
        args.width, args.height, args.fps, preencode_jpeg=False
    )
    try:
        captioner = BlipCudaCaptioner(args.model, args.max_new_tokens, args.vlm_width)
    except Exception as exc:
        print(f"failed to load CUDA VLM: {exc}", file=sys.stderr)
        print("Install deps: pip install -r requirements-cuda.txt", file=sys.stderr)
        return 1

    try:
        grabber.start()
    except Exception as exc:
        print(f"failed to start RealSense: {exc}", file=sys.stderr)
        return 1

    try:
        captioner.warmup()
    except Exception as exc:
        grabber.stop()
        print(f"CUDA VLM warmup failed: {exc}", file=sys.stderr)
        return 1

    worker = threading.Thread(
        target=infer_loop, args=(grabber, captioner, state, args, speaker), name="vlm-cuda", daemon=True
    )
    worker.start()
    print("CUDA captions (q to quit):\n", flush=True)

    if args.headless:
        try:
            while state.running:
                time.sleep(0.1)
        finally:
            print()
            speaker.stop()
            grabber.stop()
        return 0

    win = "RGB-D CUDA VLM"
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
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except cv2.error as exc:
        print(f"\nOpenCV window failed ({exc}); continuing headless.", flush=True)
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
