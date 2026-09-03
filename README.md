# RGB-D VLM demo

Live captions from a RealSense stereo depth camera and a local Ollama Qwen vision model. The loop reads aligned RGB + depth, asks the model to name the scene in **8 words or less**, and updates a window (and optional speech) as fast as the model can go.

On a Jetson AGX Thor with a D457 and `qwen3.5:4b`, captions typically land around **1 second** after the model is warm.

## What you need

- Intel RealSense D400-series camera (developed on a **D457** over GMSL; USB D400 units should work too)
- [librealsense](https://github.com/realsenseai/librealsense) 2.58.x
- Python 3.12 with OpenCV and NumPy
- [Ollama](https://ollama.com) serving a vision-capable Qwen model
- Optional: `espeak` for spoken captions

Confirm the camera is visible:

```bash
rs-enumerate-devices -s
```

Confirm Ollama has a vision model (this repo defaults to `qwen3.5:4b`):

```bash
ollama list
curl -s http://127.0.0.1:11434/api/tags
```

If the model is missing:

```bash
ollama pull qwen3.5:4b
```

## Install

System packages already expected on this Jetson image: OpenCV, NumPy, Ollama.

**RealSense must be the local CUDA + zero-copy build** (not the PyPI wheel). Build
[librealsense](https://github.com/realsenseai/librealsense) with:

```bash
-DBUILD_WITH_CUDA=ON -DBUILD_WITH_CUDA_ZEROCOPY=ON -DBUILD_PYTHON_BINDINGS=ON
```

and `sudo make install` so `/usr/local` has `librealsense2` + `pyrealsense2`.

Python deps into a local virtual environment (avoids PEP 668 with system Python):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
./scripts/use_zerocopy_pyrealsense.sh   # link /usr/local pyrealsense2 into .venv
```

If `.venv` already exists:

```bash
source .venv/bin/activate
./scripts/use_zerocopy_pyrealsense.sh
python test_zerocopy.py                 # expect ZERO-COPY / PASS
```

Confirm the camera is free (no `ros2` `realsense2_camera` holding `/dev/video*`).

Optional speech:

```bash
sudo apt install espeak
```

## Run

```bash
source .venv/bin/activate
python demo.py
```

On start, the demo prints `cuda frame path: ZERO-COPY (...)` when the CUDA zero-copy
build is active. The VLM path still JPEG-encodes on the CPU today; `test_zerocopy.py`
is the direct GPU-pointer check.

Press `q` or Esc to quit. Ctrl+C also works.

Useful variants:

```bash
python demo.py --speech          # speak each caption with espeak
python demo.py --headless        # captions only, no OpenCV window
python demo.py --composite       # send RGB|depth to the model, not just RGB
```

The window shows RGB | colorized depth, range stats above the images, and the live caption below.

A `Gtk-Message: Failed to load module "canberra-gtk-module"` line is harmless (OpenCV’s GTK window looking for click sounds). Ignore it, or install `libcanberra-gtk-module` to silence it.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `qwen3.5:4b` (or `$OLLAMA_MODEL`) | Ollama vision model |
| `--host` | `http://127.0.0.1:11434` (or `$OLLAMA_HOST`) | Ollama base URL |
| `--width` / `--height` / `--fps` | `640` / `360` / `30` | Camera streams |
| `--vlm-width` | `256` | JPEG width sent to the VLM (vision-token bound) |
| `--jpeg-quality` | `65` | JPEG quality |
| `--composite` | off | Also show colorized depth to the VLM |
| `--headless` | off | No GUI |
| `--speech` | off | Speak each caption with espeak |
| `--espeak` | `espeak` | Path to the espeak binary |
| `--num-predict` | `20` | Max tokens from the model |
| `--num-ctx` | `512` | Context window |

## How it stays fast

- A capture thread always keeps the latest aligned RGB-D frame **and** a
  pre-baked VLM JPEG; inference never waits on resize/encode or a stale buffer.
- JPEG stays on the **CPU** (OpenCV / libjpeg-turbo, ~1 ms). GPU JPEG (nvJPEG)
  was measured not to help: Ollama already owns the GPU for vision tokens, and
  CPU encode overlaps with the previous request on the capture thread.
- Default `--vlm-width 256` cuts vision tokens vs 320 with little quality loss
  on `qwen3.5:4b` (Thor). Depth is passed as meters in the prompt (`near`,
  `mid`, `L/C/R`) so the model gets RGB-D without extra vision tokens.
  `--composite` is the slower “see the depth image” path.
- Thinking is disabled (`think: false`) so Qwen 3.5 does not spend the token budget on a reasoning trace.
- The model is kept loaded (`keep_alive` 24h). First caption after a cold start is slower.

## HUD distances

All values are meters, from valid depth pixels (0.15–20 m):

- **near** — 5th percentile (closest object, ignoring specks)
- **mid** — median of the frame
- **L/C/R** — 10th percentile in the left, center, and right thirds

## Layout

```
demo.py                              live capture + Ollama loop
test_zerocopy.py                     CUDA zero-copy smoke test
requirements.txt                     numpy / opencv (pyrealsense2 via script)
scripts/use_zerocopy_pyrealsense.sh  link /usr/local CUDA bindings into .venv
scripts/rs_zerocopy_env.sh           LD_LIBRARY_PATH helper
```
