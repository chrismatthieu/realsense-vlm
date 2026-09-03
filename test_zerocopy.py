#!/usr/bin/env python3
"""Verify RealSense CUDA zero-copy works from this project's venv.

Expects the CUDA+ZEROCOPY librealsense2/pyrealsense2 install under /usr/local
(wired in via scripts/use_zerocopy_pyrealsense.sh).

    source .venv/bin/activate
    python test_zerocopy.py
"""

from __future__ import annotations

import sys

import pyrealsense2 as rs


def main() -> int:
    print(f"pyrealsense2: {rs.__file__}")
    print(f"version:      {getattr(rs, '__version__', '?')}  "
          f"({getattr(rs, '__full_version__', '?')})")
    if not hasattr(rs, "gpu_frame"):
        print("FAIL: gpu_frame API missing — still on a non-CUDA wheel?", file=sys.stderr)
        return 1

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        pipe.start(cfg)
    except Exception as exc:
        print(f"FAIL: could not start camera: {exc}", file=sys.stderr)
        print("Is the D435 free? (stop ros2 realsense2_camera / other grabbers)", file=sys.stderr)
        return 1

    try:
        for _ in range(30):
            pipe.wait_for_frames()

        any_zc = False
        for i in range(5):
            color = pipe.wait_for_frames().get_color_frame()
            if not color:
                continue
            gf = color.as_gpu_frame()
            res = color.get_gpu_data_or_upload()
            if res is None:
                path = "NONE"
                addr = None
                copied = None
            else:
                addr, copied = res
                path = "ZERO-COPY" if not copied else "UPLOAD"
            if gf and path == "ZERO-COPY":
                any_zc = True
            print(
                f"frame {color.get_frame_number():>4}  "
                f"{color.get_width()}x{color.get_height()}  "
                f"gpu_frame={'yes' if gf else 'no ':3}  "
                f"device={hex(addr) if addr else None}  "
                f"path={path}"
            )
    finally:
        pipe.stop()

    if any_zc:
        print("\nPASS: ZERO-COPY active (gpu_frame device pointer aliases the frame).")
        return 0

    print("\nFAIL: frames did not use the zero-copy path.", file=sys.stderr)
    print("Rebuild librealsense with -DBUILD_WITH_CUDA=ON -DBUILD_WITH_CUDA_ZEROCOPY=ON,",
          file=sys.stderr)
    print("re-run scripts/use_zerocopy_pyrealsense.sh, and ensure LD_LIBRARY_PATH includes /usr/local/lib.",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
