# Kandinsky 6 TI2VA — GPU (unqualified, v1)

> Text/image-to-video-and-audio serving with Kandinsky 6

## Summary

- Vendor: not yet published
- Model: Kandinsky 6 (TI2VA — text/image-to-video-and-audio)
- Task: Joint text/image-to-video-and-audio generation
- Mode: Online serving with the OpenAI-compatible API
- Hardware: Not yet qualified on specific hardware (see "Qualification scope" below)
- Maintainer: Community

## When to use this recipe

Use this recipe to serve Kandinsky 6's TI2VA pipeline — a single DiT that
jointly denoises video and audio latents, conditioned on a text prompt and,
optionally, a reference image (image-to-video). This is a v1, correctness-
first native integration: it ports Kandinsky 6's transformer, VAEs, and
denoise loop onto vLLM-Omni's own tensor-parallel primitives and scheduler
convention, but does not yet wire in MagCache/NaviCache acceleration, a
registered NABLA sparse-attention backend, or CFG-parallel/distributed
execution — see "Known limitations".

## Supported model contract

| Task | Entrypoint | Input | Output |
|---|---|---|---|
| Text-to-video-and-audio | `vllm serve <model> --omni` | text prompt | synchronized MP4 (H.264 video + AAC audio) |
| Image-to-video-and-audio | same | text prompt + reference image | synchronized MP4, first frame conditioned on the reference image |

- Audio generation is on by default (`sample_audio=True`); pass
  `extra_args.sample_audio=false` in the request to generate video only.
- Batch size is 1 request per generation call (upstream limitation of the
  ported denoise loop; concurrent requests are still served by vLLM-Omni's
  own request scheduler, just not batched together within one DiT forward
  pass yet).
- Default resolution/frame count/step count follow the pipeline's own
  defaults (512x768, 121 frames, 50 steps); override via the standard
  `height`/`width`/`num_frames`/`num_inference_steps` sampling params.

Source model specifications (checkpoint id, exact resolution/duration
limits, validated model card values) are not yet available — no Kandinsky 6
checkpoint has been published to the Hub as of this recipe. Fill in this
table from the canonical model card once one exists; do not infer limits
from the code alone.

## References

- Upstream model card: not yet published
- Native implementation source: `k6_video` (internal portable-core +
  port-generation repo — `src/kandinsky/ports/templates/vllm/` and
  `src/kandinsky/ports/overrides/vllm/`)
- Joint video+audio native-model reference: [`recipes/MiniMaxAI/MiniMax-H3.md`](../MiniMaxAI/MiniMax-H3.md)

## Hardware

- Accelerator model and per-device memory: not yet qualified
- Number of devices: 1 (multi-GPU tensor parallelism is implemented in the
  transformer's linear layers but not yet validated end-to-end)
- Device interconnect: N/A (single-device path only, until validated)
- Host memory: not yet measured
- Qualification scope: **none yet** — this recipe describes the serving
  command and model contract only; no throughput/memory/quality numbers
  have been measured on real hardware. Replace this section with real
  measurements before treating this as a qualified deployment profile.

## Software environment

- OS: any vLLM-Omni-supported Linux distribution
- Python: matches the installed vLLM-Omni environment
- Driver / runtime: CUDA (matches the installed vLLM-Omni environment)
- vLLM version: matches the installed vLLM-Omni environment
- vLLM-Omni version or commit: this repository, `vllm_omni/diffusion/models/kandinsky6/`

## Command

```bash
# Add the exact model id once a Kandinsky 6 checkpoint is published to the Hub.
vllm serve <kandinsky-6-checkpoint> --omni
```

## Verification

```bash
curl http://localhost:8000/v1/... \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<kandinsky-6-checkpoint>",
    "prompt": "a cat walking on a windowsill at sunset",
    "num_frames": 121,
    "height": 512,
    "width": 768
  }'
# Expect a synchronized MP4 response with H.264 video and an AAC audio track.
```

## Notes

- Memory usage: not yet measured.
- Key flags: `extra_args.sample_audio` (bool, default true),
  `extra_args.visual_cond_scheme` (defaults to `pretrain` for text-only,
  `tail_cond_first_frame` when an image is supplied).
- Known limitations:
  - No MagCache/NaviCache (cache-acceleration) support — listed in
    `_NO_CACHE_ACCELERATION` in `vllm_omni/diffusion/registry.py` so a
    `cache_backend` override degrades gracefully instead of erroring.
  - NABLA sparse attention and the framework's pluggable attention-backend
    registry (the FASTVIDEO_VSA-style integration point) are not wired up;
    attention runs through the native dense/flash dispatch reused from the
    portable core.
  - CFG-parallel and other distributed-execution strategies are not yet
    validated for this model.
  - `_load_components` loads directly from a converted-checkpoint bundle
    directory (`transformer/`, `vae/`, `audio_vae/`, `text_encoder/`,
    `text_encoder_2/`, `tokenizer/`, `tokenizer_2/` subfolders — the same
    layout the Diffusers port's `convert_checkpoint.py` produces) rather
    than vLLM-Omni's streamed-prefetch loader.

## Supported features

| Feature | Status | Guide |
|---|---|---|
| Text-to-video-and-audio | Supported | — |
| Image-to-video-and-audio | Supported | — |
| Tensor parallelism | Implemented, not validated | — |
| CFG parallelism | Not supported | — |
| MagCache / NaviCache | Not supported | — |
| Quantization | Not yet tested | [`docs/user_guide/diffusion/quantization.md`](../../docs/user_guide/diffusion/quantization.md) |
| LoRA | Not supported | — |
