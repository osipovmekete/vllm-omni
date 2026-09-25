# Kandinsky 6 TI2VA — H100 (v1, single GPU, CPU offload)

> Text/image-to-video-and-audio serving with Kandinsky 6 Pro

## Summary

- Vendor: not yet published on the Hub (weights supplied as a local raw
  checkpoint tree; see "Building the checkpoint bundle")
- Model: Kandinsky 6 Pro (TI2VA — text/image-to-video-and-audio)
- Task: Joint text/image-to-video-and-audio generation
- Mode: Offline (`Omni(...)`) and online serving with the OpenAI-compatible
  `/v1/videos` API
- Hardware: 1x NVIDIA H100 80GB HBM3 with component-level CPU offload
- Maintainer: Community

## When to use this recipe

Use this recipe to serve Kandinsky 6's TI2VA pipeline — a single DiT that
jointly denoises video and audio latents, conditioned on a text prompt and,
optionally, a reference image (image-to-video). Tensor parallel, CFG
parallel, Ulysses, ring, pipeline parallel, HSDP, layerwise offload,
MagCache, TeaCache, Cache-DiT, FP8, and tiled VAE patch parallel are wired.
NaviCache and a registered NABLA backend are not. TeaCache coefficients are
uncalibrated. See "Known limitations".

## Supported model contract

| Task | Entrypoint | Input | Output |
|---|---|---|---|
| Text-to-video-and-audio | `vllm serve <bundle> --omni` / `Omni(model=<bundle>)` | text prompt | synchronized MP4 (H.264 video + AAC 44.1 kHz mono audio) |
| Image-to-video-and-audio | same | text prompt + reference image | synchronized MP4, first frame conditioned on the reference image |

- Audio generation is on by default (`sample_audio=True`); pass
  `extra_args.sample_audio=false` to generate video only.
- Batch size is 1 request per generation call (upstream limitation of the
  ported denoise loop; concurrent requests are still served by vLLM-Omni's
  own request scheduler, just not batched together within one DiT forward
  pass yet).
- Serving defaults follow the Pro production geometry registered in
  `vllm_omni/model_extras/kandinsky6.py`: **864x480, 125 frames @ 24 fps
  (5.2 s), 50 steps, CFG 5.0**. Any `4k+1` frame count and 16-divisible
  resolution can be requested explicitly via `height`/`width`/`num_frames`/
  `num_inference_steps` — the model is *not* fixed-duration.
- Audio is always 44.1 kHz; its length is derived from the video duration.
- Negative prompt defaults to the pipeline's built-in K6 negative prompt
  when omitted.

## References

- Native implementation source: `k6_video` (internal portable-core +
  port-generation repo — `src/kandinsky/ports/templates/vllm/` and
  `src/kandinsky/ports/overrides/vllm/`); the Pro DiT config comes from
  `src/kandinsky/configs/k6_pro_125_480_864_mCache_mOffload.yaml`.
- Joint video+audio native-model reference: [`recipes/MiniMaxAI/MiniMax-H3.md`](../MiniMaxAI/MiniMax-H3.md)
- Bundle builder: [`tools/kandinsky6/build_bundle.py`](../../tools/kandinsky6/build_bundle.py)
- Offline example: [`examples/offline_inference/text_to_video/kandinsky6_t2va.py`](../../examples/offline_inference/text_to_video/kandinsky6_t2va.py)

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 80GB HBM3
  (driver 570.133.20)
- Number of devices: 1 (multi-GPU tensor parallelism is implemented in the
  transformer's linear layers but not yet validated end-to-end)
- Device interconnect: N/A (single-device path)
- Host memory: 1.4 TB installed; the bf16 DiT (~56 GB) plus the text
  encoders are staged on the host under `--enable-cpu-offload`, so budget
  at least ~90 GB of free host RAM.
- Qualification scope: single-GPU T2VA at the Pro default geometry and at a
  reduced smoke geometry, offline and via `vllm serve`. Numbers below were
  measured once on the shared box described above; treat them as
  indicative, not as a benchmark.

## Software environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.13
- Driver / runtime: CUDA 12.9 (torch 2.13.0+cu129)
- vLLM version: 0.29.0
- vLLM-Omni version or commit: this repository, `vllm_omni/diffusion/models/kandinsky6/`
- diffusers 0.40.0, transformers 5.14.1
- Attention: no `flash_attn_interface` / `flash_attn` / `sageattention`
  installed → the DiT's `attention_engine: "auto"` falls back to PyTorch
  SDPA. All timings below are SDPA timings; FA3 is expected to be
  substantially faster (the `k6_video` reference uses it) but has not been
  measured through this port.

## Building the checkpoint bundle

`Kandinsky6TI2VAPipeline._load_components` expects a Diffusers-style bundle
directory (`model_index.json`, `transformer/`, `vae/`, `text_encoder/`,
`tokenizer/`, `text_encoder_2/`, `tokenizer_2/`, `audio_vae/`,
`scheduler/`). Build it once from the raw Kandinsky 6 weight tree:

```bash
# Raw tree layout expected under --weights-root:
#   <dit>.safetensors            single-file DiT (mixed F32/BF16, ~70 GB)
#   vae/                         diffusers-format HunyuanVideo VAE
#   text_encoder/                Qwen2.5-VL-7B HF repo (model + processor/tokenizer)
#   text_encoder2/               CLIP text encoder HF repo
#   tod_vae/ext_weights/v1-44.pth
#   bigvgan_vocoder/             config.json + bigvgan_generator.pt
python tools/kandinsky6/build_bundle.py \
  --weights-root /path/to/weights_k6 \
  --dit-file <dit>.safetensors \
  --out /path/to/weights_k6/vllm_omni_bundle
```

What it does:

- `transformer/`: streams the raw safetensors, applies the two key renames
  (`visual_blocks.N.` → `visual_transformer_blocks.N.`,
  `audio_outLayer.` → `audio_out_layer.`), casts F32 → bf16, and writes
  ~57 GB of sharded `diffusion_pytorch_model-*.safetensors` + index next to
  a `config.json` holding the Pro `dit:` block (`attention_engine: "auto"`,
  `text_token_padding: false`). Pass `--verify` to check the re-keyed names
  against the module tree (zero missing/unexpected keys expected).
- `vae/`, `text_encoder/`, `tokenizer/`, `text_encoder_2/`, `tokenizer_2/`:
  symlinks to the raw component dirs (`--copy` to materialize).
- `audio_vae/`: `Kandinsky6AudioVAE` (MMAudio TOD VAE + BigVGAN-v2,
  `scaling_factor=0.5302`) saved via `save_pretrained`, so the bundle loads
  without the original `.pth`/`.pt` paths.
- `scheduler/scheduler_config.json`: `{"shift": 5.0}`.

The DiT loads with meta-device init + `load_state_dict(assign=True)`; the
full bundle loads in ~27 s from a warm page cache.

## Command

```bash
vllm serve /path/to/weights_k6/vllm_omni_bundle --omni \
  --host 127.0.0.1 --port 8091 \
  --num-gpus 1 --enable-cpu-offload
```

`--enable-cpu-offload` enables model-level offload with mutual exclusion
between the transformer and the two text encoders (they never co-reside on
the GPU). Without it the bf16 DiT (~56 GB) plus Qwen2.5-VL-7B (~16 GB) plus
the VAEs and activations do not fit an 80 GB device at the Pro geometry.

Offline (writes an MP4 with the audio track muxed in):

```bash
python examples/offline_inference/text_to_video/kandinsky6_t2va.py \
  --model /path/to/weights_k6/vllm_omni_bundle \
  --prompt "A golden retriever runs along a sunny beach, waves crashing, cinematic footage" \
  --seed 42 --enable-cpu-offload --output kandinsky6_t2va.mp4
# Image-to-video-and-audio: add --image first_frame.png
# Video only:               add --no-audio
```

The generic `examples/offline_inference/text_to_video/text_to_video.py`
also works with the bundle path and picks up the same Pro defaults.

## Verification

`/v1/videos` takes multipart form fields (not a JSON body). Submit, poll,
then download:

```bash
VID=$(curl -s -X POST http://127.0.0.1:8091/v1/videos \
  -F prompt="A golden retriever runs along a sunny beach, waves crashing" \
  -F size=864x480 -F num_frames=125 -F num_inference_steps=50 -F seed=42 \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

# Poll until "status": "completed"
curl -s http://127.0.0.1:8091/v1/videos/$VID

curl -s -o kandinsky6.mp4 http://127.0.0.1:8091/v1/videos/$VID/content
# Expect an MP4 with an H.264 video stream (125 frames, 24 fps, 864x480)
# and an AAC audio stream (44100 Hz, mono) of matching duration.
```

Smoke-size request for quick validation (~30 s end-to-end on one H100):
`-F size=512x320 -F num_frames=25 -F num_inference_steps=10`.

## Measurements

Single H100 80GB, `--enable-cpu-offload`, SDPA attention, seed 42, same
prompt as above. `text_to_video.py` / `vllm serve` figures:

| Geometry | Steps | DiT step time | Total generation | Peak GPU (reserved) | GPU after load |
|---|---|---|---|---|---|
| 864x480, 125 frames (Pro default) | 50 | 26.9 s/it | 1377 s (22.9 min) | 74.97 GiB | 17.7 GiB |
| 512x320, 25 frames (smoke) | 10 | 1.4 s/it | 29.5 s | 74.97 GiB | 17.7 GiB |

Notes on the numbers:

- Peak reserved memory is dominated by the resident bf16 DiT (~56 GB) plus
  SDPA attention workspace for ~50k visual tokens at the Pro geometry.
- Step time is the CFG pair (conditional + unconditional DiT forward) at
  SDPA; no cache acceleration is applied.
- Output was sanity-checked against the `k6_video` production CLI
  (`kandy generate`, same prompt and seed, MagCache and prompt expansion
  off, FA3 + module offload, ~18 min): both produce a coherent golden
  retriever running along a beach with breaking waves across all 125
  frames, with an audible synchronized track of identical length (5.22 s).
  Bit-exact parity is not expected (different noise plumbing), so the
  scenes differ in composition.
- Audio is peak-normalized to full scale by default, as in the production
  pipeline; pass `extra_args.audio_normalization="clip"` to keep the raw
  decoded amplitude instead.

## Notes

- Key flags: `extra_args.sample_audio` (bool, default true),
  `extra_args.audio_normalization` (`"normalize"` default | `"clip"`),
  `extra_args.visual_cond_scheme` (defaults to `pretrain` for text-only,
  `tail_cond_first_frame` when an image is supplied).
- The post-process payload is a flat dict (`video`, `audio` as float32 in
  `[-1, 1]`, `audio_sample_rate=44100`, `fps`) so the serving muxer picks
  up the correct sample rate.
- Acceleration that is wired:
  - Tensor parallel: the DiT loader narrows full checkpoint tensors onto
    each rank's `ColumnParallelLinear` / `RowParallelLinear` shard.
  - CFG parallel (`--cfg-parallel-size 2`): rank 0 runs the conditional
    DiT, rank 1 the unconditional one, then both all-gather video and
    audio velocities.
  - HSDP: `_hsdp_shard_conditions` covers the indexed transformer blocks.
    Do not combine with tensor parallel.
  - Sequence parallel: visual tokens (and their RoPE) are sharded;
    Ulysses all-to-all runs inside visual self-attention, Ring uses the
    PyTorch ring kernel. Text stays replicated. Audio queries gather the
    full visual sequence.
  - Pipeline parallel: `visual_transformer_blocks` are split with
    `PPMissingLayer`. Embeddings stay on the first stage, output heads on
    the last. Activations and the final velocity are sent between stages.
  - VAE patch parallel: Hunyuan decode splits the latent width when
    `vae_patch_parallel_size > 1` and tiling is on.
  - MagCache / TeaCache / Cache-DiT are registered. MagCache reuses the
    Pro `mag_ratios`. TeaCache coefficients are **uncalibrated** (copied
    from Qwen-Image) so skips are not quality-neutral. Cache-DiT targets
    `visual_transformer_blocks` only and falls back to the same step cache
    if the fused `(video, audio)` block return is rejected.
  - FP8: online quantization runs in the DiT loader via
    `process_weights_after_loading` after the sharded assign.
  - `attention_engine: auto` uses FlashAttention-3 when
    `flash_attn_interface` imports.
- Known limitations:
  - NABLA sparse attention is not registered in the framework attention
    backend. NaviCache is not wired.
  - Prompt expansion, NF4 Qwen, and the super-resolution cascade are not
    part of this port.
  - Distributed layerwise offload still uses one process unless another
    parallel axis sets `world_size > 1` (for example tensor parallel).
  - `_load_components` loads the bundle directly rather than the
    streamed-prefetch loader.

## Supported features

| Feature | Status | Guide |
|---|---|---|
| Text-to-video-and-audio | Supported (validated on H100) | — |
| Image-to-video-and-audio | Supported (not yet validated on hardware) | — |
| CPU offload (`--enable-cpu-offload`) | Supported (required on 80 GB) | — |
| Layerwise offload (`--enable-layerwise-offload`) | Supported (about 36 GiB peak on the smoke geometry) | — |
| Tensor parallelism | Wired (sharded DiT load) | — |
| CFG parallelism | Wired (`--cfg-parallel-size 2`) | — |
| Ulysses / Ring sequence parallel | Wired (visual tokens only) | — |
| Pipeline parallelism | Wired (visual block split) | — |
| HSDP | Wired (not with tensor parallel) | — |
| VAE patch parallel | Wired (width-split decode) | — |
| MagCache | Wired (Pro ratios, step reuse) | — |
| TeaCache | Wired, coefficients uncalibrated | — |
| Cache-DiT | Wired on `visual_transformer_blocks` | — |
| FP8 | Wired (online quant; smoke peak about 46 GiB vs 75 GiB bf16) | [`docs/user_guide/diffusion/quantization.md`](../../docs/user_guide/diffusion/quantization.md) |
| NaviCache | Not supported | — |
| LoRA | Not supported | — |
