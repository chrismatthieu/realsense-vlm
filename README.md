# RGB-D VLM demo

Live captions from a RealSense stereo depth camera on Jetson. Two backends:

| Demo | Backend | Warm latency (Thor) | Best for |
| --- | --- | --- | --- |
| `demo.py` | Ollama + `qwen3.5:4b` (JPEG over HTTP) | ~280–350 ms | Richer language / depth in the prompt |
| `demo_cuda.py` | In-process CUDA **BLIP** | ~70–110 ms | Lowest latency |

Both show RGB | colorized depth, range stats, and an **≤8 word** caption (optional speech).

## What you need

- Intel RealSense D400-series camera (developed on a **D457** over GMSL; USB D400 units work too)
- [librealsense](https://github.com/realsenseai/librealsense) 2.58.x built with **CUDA + zero-copy** (see Install)
- Python 3.12 with OpenCV and NumPy
- **Ollama demo:** [Ollama](https://ollama.com) with a vision model (default `qwen3.5:4b`)
- **CUDA demo:** Jetson CUDA PyTorch + transformers (`requirements-cuda.txt`)
- Optional: `espeak` for spoken captions

Confirm the camera is visible and free (nothing holding `/dev/video*`):

```bash
rs-enumerate-devices -s
fuser /dev/video0 2>&1   # should print nothing
```

### Ollama model (for `demo.py` only)

```bash
ollama list
curl -s http://127.0.0.1:11434/api/tags
# if missing:
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
python test_zerocopy.py                 # expect ZERO-COPY / PASS
```

For the faster CUDA caption demo, also install Jetson CUDA torch (SBSA / CUDA 13.x index):

```bash
pip install -r requirements-cuda.txt
```

Optional speech:

```bash
sudo apt install espeak
```

## Run

```bash
source .venv/bin/activate
```

### Ollama / Qwen (`demo.py`)

```bash
python demo.py
python demo.py --speech
python demo.py --headless
python demo.py --composite       # send RGB|depth image to the model
```

Prints `cuda frame path: ZERO-COPY (...)` when the zero-copy SDK build is active.
Uses a CPU JPEG + Ollama HTTP; depth distances are embedded in the text prompt.

### CUDA BLIP (`demo_cuda.py`) — faster

In-process GPU captioning — **no JPEG, no Ollama**. First run downloads
`Salesforce/blip-image-captioning-base` into `.hf_cache/`.

```bash
python demo_cuda.py
python demo_cuda.py --headless
python demo_cuda.py --speech
```

Captions are shorter / less “chatty” than Qwen; pick this when latency matters more
than LLM reasoning.

Press `q` or Esc to quit. Ctrl+C also works.

A `Gtk-Message: Failed to load module "canberra-gtk-module"` line is harmless
(OpenCV’s GTK window). Ignore it, or install `libcanberra-gtk-module` to silence it.

## Options

### Shared

| Flag | Default | Meaning |
| --- | --- | --- |
| `--width` / `--height` / `--fps` | `640` / `360` / `30` | Camera streams |
| `--headless` | off | No GUI |
| `--speech` | off | Speak each caption with espeak |
| `--espeak` | `espeak` | Path to the espeak binary |

### `demo.py` (Ollama)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `qwen3.5:4b` (or `$OLLAMA_MODEL`) | Ollama vision model |
| `--host` | `http://127.0.0.1:11434` (or `$OLLAMA_HOST`) | Ollama base URL |
| `--vlm-width` | `256` | JPEG width sent to the VLM |
| `--jpeg-quality` | `65` | JPEG quality |
| `--composite` | off | Also show colorized depth to the VLM |
| `--num-predict` | `20` | Max tokens from the model |
| `--num-ctx` | `512` | Context window |

### `demo_cuda.py` (CUDA BLIP)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Salesforce/blip-image-captioning-base` | Hugging Face BLIP id |
| `--vlm-width` | `384` | RGB width fed to BLIP |
| `--max-new-tokens` | `12` | Max generated tokens |
| `--hf-home` | `.hf_cache/` | Model download cache |

## How it stays fast

**Ollama path (`demo.py`)**

- Capture thread keeps the latest aligned RGB-D frame and a pre-baked VLM JPEG.
- JPEG stays on the CPU (~1 ms). GPU JPEG does not help: Ollama owns the GPU for
  vision tokens.
- Default `--vlm-width 256` limits vision tokens. Depth is meters in the prompt
  (`near` / `mid` / `L/C/R`); `--composite` is slower.
- `think: false`, `keep_alive` 24h. First caption after a cold start is slower.

**CUDA path (`demo_cuda.py`)**

- Same RealSense capture / HUD; inference is BLIP `float16` on the Jetson GPU.
- No JPEG/HTTP round-trip — RGB goes straight into the model.
- On Thor, warm captions land around **70–110 ms** (vs ~300 ms for Ollama Qwen).

**Zero-copy**

- Requires `-DBUILD_WITH_CUDA_ZEROCOPY=ON` and `./scripts/use_zerocopy_pyrealsense.sh`.
- Verify with `python test_zerocopy.py`.

## HUD distances

All values are meters, from valid depth pixels (0.15–20 m):

- **near** — 5th percentile (closest object, ignoring specks)
- **mid** — median of the frame
- **L/C/R** — 10th percentile in the left, center, and right thirds

## Layout

```
demo.py                              Ollama / Qwen live captions
demo_cuda.py                         CUDA BLIP live captions (faster)
test_zerocopy.py                     CUDA zero-copy smoke test
requirements.txt                     numpy / opencv (pyrealsense2 via script)
requirements-cuda.txt                Jetson CUDA torch + transformers for demo_cuda
scripts/use_zerocopy_pyrealsense.sh  link /usr/local CUDA bindings into .venv
scripts/rs_zerocopy_env.sh           LD_LIBRARY_PATH helper
```
