# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_kandinsky6_pipeline_import_and_registry():
    from vllm_omni.diffusion.models.kandinsky6 import (
        AutoencoderKLHunyuanVideo,
        Kandinsky6AudioVAE,
        Kandinsky6TI2VAPipeline,
        Kandinsky6Transformer3DModel,
        KandinskyFlowMatchScheduler,
        get_kandinsky6_post_process_func,
        get_kandinsky6_pre_process_func,
    )
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
        _NO_CACHE_ACCELERATION,
    )

    assert Kandinsky6TI2VAPipeline is not None
    assert Kandinsky6Transformer3DModel is not None
    assert AutoencoderKLHunyuanVideo is not None
    assert Kandinsky6AudioVAE is not None
    assert KandinskyFlowMatchScheduler is not None
    assert get_kandinsky6_post_process_func is not None
    assert get_kandinsky6_pre_process_func is not None

    assert _DIFFUSION_MODELS["Kandinsky6TI2VAPipeline"] == (
        "kandinsky6",
        "pipeline_kandinsky6",
        "Kandinsky6TI2VAPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["Kandinsky6TI2VAPipeline"] == "get_kandinsky6_post_process_func"
    assert _DIFFUSION_PRE_PROCESS_FUNCS["Kandinsky6TI2VAPipeline"] == "get_kandinsky6_pre_process_func"
    assert "Kandinsky6TI2VAPipeline" in _NO_CACHE_ACCELERATION


def test_kandinsky6_component_discovery_declarations():
    from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6TI2VAPipeline

    assert Kandinsky6TI2VAPipeline._dit_modules == ["transformer"]
    assert Kandinsky6TI2VAPipeline._encoder_modules == ["text_encoder", "text_encoder_2"]
    assert Kandinsky6TI2VAPipeline._vae_modules == ["vae", "audio_vae"]
    assert Kandinsky6TI2VAPipeline.supports_step_execution is False
    assert Kandinsky6TI2VAPipeline.support_audio_output is True
    assert Kandinsky6TI2VAPipeline.support_image_input is True


def test_kandinsky6_pipeline_satisfies_capability_protocols():
    """isinstance checks against the @runtime_checkable protocols the
    framework actually uses to detect these capabilities (io_support.py's
    supports_audio_output, etc.) — stronger than just checking the class
    attribute exists."""
    from vllm_omni.diffusion.models.interface import (
        SupportAudioOutput,
        SupportImageInput,
        SupportsComponentDiscovery,
    )
    from vllm_omni.diffusion.models.kandinsky6 import Kandinsky6TI2VAPipeline

    assert isinstance(Kandinsky6TI2VAPipeline, SupportAudioOutput)
    assert isinstance(Kandinsky6TI2VAPipeline, SupportImageInput)
    assert isinstance(Kandinsky6TI2VAPipeline, SupportsComponentDiscovery)


class _FakeAudioVAE:
    downsample_factor = 1024


def test_kandinsky6_post_process_func_packages_video_and_audio():
    """Pure-function test: no model/weights needed. Confirms the post-process
    payload shape matches what io_support.py -> output_formatter.py ->
    media_utils.py's PyAV muxing expects (the same {"video", "audio",
    "audio_sample_rate"} shape MiniMax H3's own post-process function
    produces)."""
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_post_process_func

    post_process = get_kandinsky6_post_process_func(od_config=None)
    video = np.zeros((1, 4, 8, 8, 3), dtype=np.uint8)
    audio = np.zeros((100,), dtype=np.int16)

    result = post_process({"video": video, "audio": audio, "audio_sample_rate": 44100}, output_type="np")

    assert result["payload"]["video"] is video
    assert result["payload"]["audio"] is audio
    assert result["payload"]["audio_sample_rate"] == 44100
    assert result["metadata"] == {}


def test_kandinsky6_post_process_func_omits_audio_when_none():
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_post_process_func

    post_process = get_kandinsky6_post_process_func(od_config=None)
    video = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)

    result = post_process({"video": video, "audio": None, "audio_sample_rate": None}, output_type="np")

    assert result["payload"]["video"] is video
    assert "audio" not in result["payload"]
    assert "audio_sample_rate" not in result["payload"]


def test_kandinsky6_post_process_func_unwraps_batched_audio_list():
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_post_process_func

    post_process = get_kandinsky6_post_process_func(od_config=None)
    video = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
    audio_item = np.zeros((50,), dtype=np.int16)

    result = post_process({"video": video, "audio": [audio_item], "audio_sample_rate": 44100}, output_type="np")

    assert result["payload"]["audio"] is audio_item


def test_kandinsky6_pre_process_func_is_identity_for_now():
    """v1 scope: the pre-process hook is registered (matching the framework's
    convention for image-conditioned models like SanaImageToVideoPipeline)
    but does not yet validate/transform the request — documented follow-up,
    not a silent gap."""
    from vllm_omni.diffusion.models.kandinsky6 import get_kandinsky6_pre_process_func

    pre_process = get_kandinsky6_pre_process_func(od_config=None)
    sentinel = object()

    assert pre_process(sentinel) is sentinel
