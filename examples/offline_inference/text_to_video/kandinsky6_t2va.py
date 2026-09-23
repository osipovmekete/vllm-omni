# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Kandinsky 6 text/image-to-video-and-audio (TI2VA) offline example.

Generates a video with a synchronized 44.1 kHz audio track from a text prompt
(and optionally a reference image) using ``Kandinsky6TI2VAPipeline`` and writes
an MP4 (H.264 + AAC).

The model argument is the converted checkpoint bundle produced by
``tools/kandinsky6/build_bundle.py`` (see recipes/Kandinsky/Kandinsky6-TI2VA.md).

Usage (Pro default geometry: 480x864, 125 frames @ 24 fps, 50 steps, CFG 5):
    python kandinsky6_t2va.py \
        --model /path/to/kandinsky6_bundle \
        --prompt "A golden retriever runs along a sunny beach, waves crashing" \
        --enable-cpu-offload \
        --output kandinsky6_t2va.mp4

Image-to-video-and-audio:
    python kandinsky6_t2va.py --model ... --image first_frame.png --prompt "..."

Video only (skip the audio branch):
    python kandinsky6_t2va.py --model ... --no-audio
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.kandinsky6 import get_kandinsky6_video_generation_defaults


def parse_args() -> argparse.Namespace:
    defaults = get_kandinsky6_video_generation_defaults()
    parser = argparse.ArgumentParser(description="Kandinsky 6 text/image-to-video-and-audio generation.")
    parser.add_argument("--model", required=True, help="Path to the converted Kandinsky 6 checkpoint bundle.")
    parser.add_argument(
        "--prompt",
        default="A golden retriever runs along a sunny beach, waves crashing, cinematic footage",
        help="Text prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt. Default: the pipeline's built-in K6 negative prompt.",
    )
    parser.add_argument("--image", default=None, help="Optional reference image (path) for I2VA.")
    parser.add_argument(
        "--height", type=int, default=defaults.height, help=f"Video height (default {defaults.height})."
    )
    parser.add_argument("--width", type=int, default=defaults.width, help=f"Video width (default {defaults.width}).")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=defaults.num_frames,
        help=f"Number of frames; 4k+1 (default {defaults.num_frames} = 32 latent frames).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=defaults.num_inference_steps,
        help=f"Sampling steps (default {defaults.num_inference_steps}).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=defaults.guidance_scale,
        help=f"CFG scale (default {defaults.guidance_scale}).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-audio", action="store_true", help="Generate video only (skip the audio branch).")
    parser.add_argument("--output", default=defaults.output, help="Output MP4 path.")
    parser.add_argument(
        "--enable-cpu-offload", action="store_true", help="Component-level CPU offload (recommended on 80 GB)."
    )
    parser.add_argument(
        "--enable-layerwise-offload", action="store_true", help="Stream DiT blocks from CPU (smaller GPU footprint)."
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for the DiT.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    return parser.parse_args()


def _frames_to_uint8(images) -> np.ndarray:
    """Unwrap OmniRequestOutput.images into a (T, H, W, C) uint8 array."""
    data = images
    while (
        isinstance(data, list)
        and data
        and isinstance(data[0], (list, np.ndarray))
        and not (isinstance(data[0], np.ndarray) and data[0].ndim == 3)
    ):
        data = data[0]
    if isinstance(data, list):
        data = np.stack(data)
    frames = np.asarray(data)
    if frames.ndim == 5:
        frames = frames[0]
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)
    return frames


def main() -> None:
    args = parse_args()

    omni = Omni(
        model=args.model,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )

    prompt: dict = {"prompt": args.prompt}
    if args.negative_prompt is not None:
        prompt["negative_prompt"] = args.negative_prompt
    if args.image is not None:
        from PIL import Image

        prompt["multi_modal_data"] = {"image": Image.open(args.image).convert("RGB")}

    sampling = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        extra_args={"sample_audio": not args.no_audio},
    )

    print(
        f"Generating {args.width}x{args.height}x{args.num_frames} in {args.num_inference_steps} steps (audio={not args.no_audio})"
    )
    start = time.perf_counter()
    result = omni.generate(prompt, sampling)
    elapsed = time.perf_counter() - start
    print(f"Total generation time: {elapsed:.1f}s")

    output = result[0] if isinstance(result, list) else result
    if not output.images:
        raise ValueError("No video frames found in OmniRequestOutput.")
    frames = _frames_to_uint8(output.images)

    mm = output.multimodal_output or {}
    audio = mm.get("audio")
    fps = float(mm.get("fps", 24.0))
    sample_rate = int(mm.get("audio_sample_rate", 44100))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    if audio is not None:
        audio = np.squeeze(np.asarray(audio)).astype(np.float32)
    output_path.write_bytes(mux_video_audio_bytes(frames, audio, fps=fps, audio_sample_rate=sample_rate))

    print(f"Saved {output_path}: {frames.shape[0]} frames @ {fps:g} fps ({frames.shape[0] / fps:.2f}s)")
    if audio is not None:
        print(f"Audio: {audio.shape[-1] / sample_rate:.2f}s at {sample_rate} Hz")
    else:
        print("Audio: none (--no-audio)")


if __name__ == "__main__":
    main()
