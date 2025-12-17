import importlib
import sys
import types
from pathlib import Path

import pytest


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture(autouse=True)
def stub_third_party(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.nn = types.ModuleType("torch.nn")
    fake_F = types.ModuleType("torch.nn.functional")
    fake_F.interpolate = lambda input, *args, **kwargs: input
    fake_F.cosine_similarity = lambda a, b: types.SimpleNamespace(item=lambda: 0.0)
    fake_torch.nn.functional = fake_F
    fake_torch.mean = lambda tensor: types.SimpleNamespace(item=lambda: 0.0)
    fake_torch.abs = lambda tensor: tensor
    fake_torch.randn_like = lambda x: x
    fake_torch.Tensor = object
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_torch.nn
    sys.modules["torch.nn.functional"] = fake_F

    fake_gradio = types.ModuleType("gradio")
    fake_gradio.update = lambda **kwargs: {"__update__": kwargs}
    fake_gradio.Accordion = lambda *args, **kwargs: DummyContext()
    fake_gradio.Row = lambda *args, **kwargs: DummyContext()
    fake_gradio.Checkbox = lambda *args, **kwargs: types.SimpleNamespace()
    fake_gradio.Radio = lambda *args, **kwargs: types.SimpleNamespace()
    fake_gradio.Slider = lambda *args, **kwargs: types.SimpleNamespace()
    fake_gradio.HTML = lambda *args, **kwargs: None
    sys.modules["gradio"] = fake_gradio

    fake_modules = types.ModuleType("modules")
    fake_modules.__path__ = []
    fake_scripts = types.ModuleType("modules.scripts")

    class DummyScript:
        pass

    fake_scripts.Script = DummyScript
    fake_scripts.AlwaysVisible = object()
    fake_modules.scripts = fake_scripts
    sys.modules["modules"] = fake_modules
    sys.modules["modules.scripts"] = fake_scripts
    fake_scripts_pkg = types.ModuleType("scripts")
    fake_scripts_pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "scripts")]
    sys.modules["scripts"] = fake_scripts_pkg

    yield

    for name in ["scripts.sada_forge", "scripts"]:
        sys.modules.pop(name, None)


@pytest.fixture
def sada_module(stub_third_party):
    return importlib.import_module("scripts.sada_forge")


def test_disables_when_total_steps_low(monkeypatch, sada_module):
    apply_called = False

    def fake_apply(*args, **kwargs):
        nonlocal apply_called
        apply_called = True
        return "patched"

    monkeypatch.setattr(sada_module, "apply_sada_acceleration", fake_apply)
    sada_module.cleanup_sada_patches()

    script = sada_module.SADAForForge()
    sd_model = types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet="base"))
    p = types.SimpleNamespace(steps=1, sd_model=sd_model, extra_generation_params={})

    script.process_before_every_sampling(p, True, "SDXL (Balanced)", 0.2, 15, 45, 0.02)

    assert apply_called is False
    assert sada_module._sada_state["is_active"] is False
    assert p.extra_generation_params == {}


@pytest.mark.parametrize(
    ("total_steps", "expected_range"),
    [
        (2, (0, 1)),
        (3, (1, 2)),
        (4, (1, 3)),
        (5, (2, 4)),
    ],
)
def test_acc_range_scaling_small_steps(monkeypatch, sada_module, total_steps, expected_range):
    captured = {}

    def fake_apply(unet_patcher, skip_ratio, acc_range, early_exit_threshold, total_steps):
        captured["acc_range"] = acc_range
        captured["total_steps"] = total_steps
        return "patched"

    monkeypatch.setattr(sada_module, "apply_sada_acceleration", fake_apply)
    sada_module.cleanup_sada_patches()

    script = sada_module.SADAForForge()
    sd_model = types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet="base"))
    p = types.SimpleNamespace(steps=total_steps, sd_model=sd_model, extra_generation_params={})

    script.process_before_every_sampling(p, True, "SDXL (Balanced)", 0.2, 15, 45, 0.02)

    assert captured["acc_range"] == expected_range
    assert captured["total_steps"] == total_steps
    assert p.extra_generation_params["SADA_range_actual"] == f"{expected_range[0]}-{expected_range[1]}"
    assert p.sd_model.forge_objects.unet == "patched"
