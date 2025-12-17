import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyComponent:
    def change(self, *_, **__):
        return None


torch_nn_functional_stub = types.SimpleNamespace(
    interpolate=lambda *args, **kwargs: args[0] if args else None,
    cosine_similarity=lambda *_args, **_kwargs: types.SimpleNamespace(item=lambda: 0.0),
)
torch_stub = types.SimpleNamespace(
    randn_like=lambda x: x,
    mean=lambda x: x,
    abs=abs,
    arange=lambda *args, **kwargs: None,
    nn=types.SimpleNamespace(functional=torch_nn_functional_stub),
)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_stub.nn)
sys.modules.setdefault("torch.nn.functional", torch_nn_functional_stub)

gradio_stub = types.SimpleNamespace(
    Accordion=lambda *_, **__: _DummyContext(),
    Checkbox=lambda *_, **__: _DummyComponent(),
    Radio=lambda *_, **__: _DummyComponent(),
    Slider=lambda *_, **__: _DummyComponent(),
    Row=lambda *_, **__: _DummyContext(),
    HTML=lambda *_, **__: None,
    update=lambda **kwargs: kwargs,
)
sys.modules.setdefault("gradio", gradio_stub)

scripts_stub = types.SimpleNamespace(Script=object, AlwaysVisible=object())
sys.modules.setdefault("modules", types.SimpleNamespace(scripts=scripts_stub))
sys.modules.setdefault("modules.scripts", scripts_stub)

from scripts import sada_forge


class DummyModel:
    def __init__(self):
        self.calls = 0
        self.apply_model = self._base_apply_model

    def _base_apply_model(self, x, timestep, **kwargs):
        self.calls += 1
        return x


class DummyUNetPatcher:
    def __init__(self):
        self.model = DummyModel()
        self.output_block_patch = None
        self.patch_set_calls = 0
        self.patch_clear_calls = 0

    def clone(self):
        return self

    def set_model_output_block_patch(self, patch_fn):
        self.output_block_patch = patch_fn
        if patch_fn is None:
            self.patch_clear_calls += 1
        else:
            self.patch_set_calls += 1


class FailingClearUNetPatcher(DummyUNetPatcher):
    def set_model_output_block_patch(self, patch_fn):
        if patch_fn is None:
            raise RuntimeError("simulated clear failure")
        super().set_model_output_block_patch(patch_fn)


def reset_state():
    sada_forge.cleanup_sada_patches()


def test_cleanup_removes_forward_patch_and_restores_model():
    reset_state()
    unet = DummyUNetPatcher()
    original_apply_model = unet.model.apply_model

    patched_unet = sada_forge.apply_sada_acceleration(
        unet_patcher=unet,
        skip_ratio=0.2,
        acc_range=(0, 1),
        early_exit_threshold=0.01,
        total_steps=10,
    )

    assert patched_unet.output_block_patch is not None
    assert patched_unet.model.apply_model is not original_apply_model
    assert patched_unet.patch_set_calls == 1

    sada_forge.cleanup_sada_patches()

    assert patched_unet.output_block_patch is None
    assert patched_unet.patch_clear_calls == 1
    assert patched_unet.model.apply_model is original_apply_model
    assert sada_forge._sada_state["patched_unet"] is None


def test_repeated_toggle_does_not_accumulate_unet_patches():
    reset_state()
    unet = DummyUNetPatcher()
    original_apply_model = unet.model.apply_model

    for _ in range(2):
        sada_forge.apply_sada_acceleration(
            unet_patcher=unet,
            skip_ratio=0.1,
            acc_range=(0, 1),
            early_exit_threshold=0.01,
            total_steps=8,
        )
        sada_forge.cleanup_sada_patches()

        assert unet.output_block_patch is None
        assert unet.model.apply_model is original_apply_model
        assert sada_forge._sada_state["patched_unet"] is None

    assert unet.patch_set_calls == 2
    assert unet.patch_clear_calls == 2


def test_cleanup_warns_once_and_resets_state_on_patch_clear_failure(capsys):
    reset_state()
    unet = FailingClearUNetPatcher()

    sada_forge.apply_sada_acceleration(
        unet_patcher=unet,
        skip_ratio=0.1,
        acc_range=(0, 1),
        early_exit_threshold=0.01,
        total_steps=8,
    )

    sada_forge.cleanup_sada_patches()
    output = capsys.readouterr().out

    assert "Cleanup warning - failed to clear UNet forward patch" in output
    assert sada_forge._sada_state["is_active"] is False
    assert sada_forge._sada_state["patched_unet"] is None
    assert sada_forge._sada_state["cleanup_warning_logged"] is False


def test_cleanup_handles_missing_forward_patch_method():
    reset_state()

    class NoPatchSetter(DummyUNetPatcher):
        def set_model_output_block_patch(self, patch_fn):
            raise AttributeError("should not be called")

    unet = NoPatchSetter()
    returned = sada_forge.apply_sada_acceleration(
        unet_patcher=unet,
        skip_ratio=0.1,
        acc_range=(0, 1),
        early_exit_threshold=0.01,
        total_steps=8,
    )

    sada_forge.cleanup_sada_patches()

    assert returned is unet
    assert sada_forge._sada_state["patched_unet"] is None
    assert sada_forge._sada_state["original_apply_model"] is None
