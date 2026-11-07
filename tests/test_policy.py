"""Tests for LeRobotPolicy.

The injected-``predict_fn`` tests need no torch and no lerobot. The
``_default_predict`` seam tests install ``sys.modules`` fakes for torch and the
lerobot modules the seam imports, and assert the *wiring* (which lerobot helper is
called with which arguments) — real tensor behaviour is lerobot's contract,
drift-guarded by the ``lerobot-seam`` CI job that imports the real symbols.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.task import TaskEnvelope
from inspect_robots.types import Observation

from inspect_robots_so101 import packing
from inspect_robots_so101.config import LeRobotPolicyConfig
from inspect_robots_so101.policy import LeRobotPolicy
from inspect_robots_so101.smoothing import smooth


def _policy_cfg(**over: Any) -> LeRobotPolicyConfig:
    values: dict[str, Any] = {"cam_height": 4, "cam_width": 4}
    values.update(over)
    return LeRobotPolicyConfig(**values)


def _obs(
    instruction: str | None = "do it",
    *,
    image_shape: tuple[int, ...] = (4, 4, 3),
) -> Observation:
    img = np.zeros(image_shape, dtype=np.uint8)
    return Observation(
        images={"front": img},
        state={"joint_pos": np.zeros(6)},
        instruction=instruction,
    )


def _fake_predict(actions: np.ndarray):
    captured: dict[str, Any] = {}

    def _predict(obs: Any) -> np.ndarray:
        captured["obs"] = obs
        return actions

    return _predict, captured


def test_info_and_config_zero_arg() -> None:
    pol = LeRobotPolicy()
    assert pol.info.name == "lerobot"
    assert pol.info.action_space.dim == 6
    assert pol.info.action_space.semantics is not None
    assert pol.info.action_space.semantics.control_mode == "joint_pos"
    assert pol.info.control_hz is None  # load-bearing: keeps compat warning-free
    assert pol.info.observation_space.state_keys == frozenset({"joint_pos"})
    assert pol.config.action_horizon == 50
    assert pol.config.rtc is False
    assert pol.config.smooth is False
    assert pol.info.checkpoint == "lerobot/smolvla_base"


def test_normalized_info_declares_normalized_state() -> None:
    pol = LeRobotPolicy(_policy_cfg(use_degrees=False))
    state = pol.info.observation_space.state
    assert state is not None and state.fields[0].unit == "normalized"


def test_act_builds_raw_lerobot_obs_and_chunk() -> None:
    actions = np.arange(2 * 6, dtype=float).reshape(2, 6)
    predict, captured = _fake_predict(actions)
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="pick up the cube"))
    chunk = pol.act(_obs())

    assert len(chunk) == 2
    assert np.array_equal(chunk.actions[0].data, actions[0])
    assert chunk.control_hz is None  # in-process: embodiment paces
    assert chunk.inference_latency_s is not None
    # The payload is the RAW robot format (what a lerobot Robot.get_observation
    # returns, plus "task"): "<motor>.pos" floats and frames keyed by camera name.
    obs = captured["obs"]
    assert obs["task"] == "pick up the cube"
    assert [obs[key] for key in packing.motor_keys()] == [0.0] * 6
    assert isinstance(obs["shoulder_pan.pos"], float)
    assert obs["front"].shape == (4, 4, 3)
    assert "observation.state" not in obs  # packing is lerobot's job, not ours
    assert pol.num_inferences == 1


def test_act_uses_empty_instruction_when_none() -> None:
    predict, captured = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction=None))
    pol.act(_obs(instruction=None))
    assert captured["obs"]["task"] == ""


def test_act_truncates_chunk_to_chunk_size() -> None:
    predict, _ = _fake_predict(np.arange(5 * 6, dtype=float).reshape(5, 6))
    pol = LeRobotPolicy(_policy_cfg(chunk_size=2), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    chunk = pol.act(_obs())
    assert len(chunk) == 2  # model returned 5; consume only the first chunk_size
    assert np.array_equal(chunk.actions[1].data, np.arange(6, 12, dtype=float))


def test_act_empty_actions_raises() -> None:
    predict, _ = _fake_predict(np.zeros((0, 6)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="empty action chunk"):
        pol.act(_obs())


def test_act_wrong_action_width_raises() -> None:
    predict, _ = _fake_predict(np.zeros((2, 7)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=r"expected \(N, 6\)"):
        pol.act(_obs())


def test_act_missing_camera_raises() -> None:
    predict, _ = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(images={}, state={"joint_pos": np.zeros(6)})
    with pytest.raises(ValueError, match="missing camera"):
        pol.act(obs)


def test_act_missing_state_raises() -> None:
    predict, _ = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(images={"front": np.zeros((4, 4, 3), np.uint8)}, state={})
    with pytest.raises(ValueError, match="missing state key"):
        pol.act(obs)


def test_config_object_overrides_flat() -> None:
    pol = LeRobotPolicy(_policy_cfg(chunk_size=3))
    assert pol.config.action_horizon == 3
    assert packing.TOTAL_DIM == 6  # sanity


@pytest.mark.parametrize("shape", [(2, 4, 3), (4, 4, 1)])
def test_act_rejects_wrong_camera_shape(shape: tuple[int, ...]) -> None:
    predict, _ = _fake_predict(np.zeros((1, 6)))
    pol = LeRobotPolicy(_policy_cfg(), predict_fn=predict)
    pol.reset(Scene(id="s", instruction="x"))

    expected = (4, 4, 3)
    with pytest.raises(ValueError) as exc:
        pol.act(_obs(image_shape=shape))
    assert str(exc.value) == f"camera 'front' returned shape {shape}, expected {expected}"


# ---------------------------------------------------------------------------
# _default_predict seam: sys.modules fakes for torch + the lerobot modules.
# ---------------------------------------------------------------------------


class FakeTensor:
    """Minimal torch.Tensor stand-in: ndim/shape/slicing/unsqueeze/squeeze/numpy."""

    def __init__(self, arr: Any) -> None:
        self.arr = np.asarray(arr, dtype=np.float64)

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    def unsqueeze(self, dim: int) -> FakeTensor:
        return FakeTensor(np.expand_dims(self.arr, dim))

    def squeeze(self, dim: int) -> FakeTensor:
        return FakeTensor(np.squeeze(self.arr, dim))

    def __getitem__(self, idx: Any) -> FakeTensor:
        return FakeTensor(self.arr[idx])

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.arr


def _install_lerobot_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chunk: np.ndarray,
    image_features: dict[str, Any] | None = None,
    rename_map: dict[str, str] | None = None,
    hub_adapter: bool = False,
    supports_rtc: bool = True,
) -> dict[str, Any]:
    """Install fakes for torch, peft, huggingface_hub and the lerobot modules the seam
    imports."""
    calls: dict[str, Any] = {}
    if image_features is None:
        image_features = {"observation.images.front": SimpleNamespace(shape=(3, 4, 4))}

    class _FakePolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(image_features=image_features)
            calls["policy_config"] = self.config

        def to(self, device: str) -> None:
            calls["to_device"] = device

        def eval(self) -> None:
            calls["eval"] = True

        def supports_rtc(self) -> bool:
            return supports_rtc

        def predict_action_chunk(self, batch: Any, **kw: Any) -> FakeTensor:
            calls["predict_batch"] = batch
            calls.setdefault("predict_kwargs", []).append(kw)
            return FakeTensor(chunk)

    class _FakePolicyClass:
        @staticmethod
        def from_pretrained(path: str, **kw: Any) -> _FakePolicy:
            calls["from_pretrained"] = path
            calls["from_pretrained_kwargs"] = kw
            return _FakePolicy()

    def get_policy_class(name: str) -> type[_FakePolicyClass]:
        calls.setdefault("policy_types", []).append(name)
        return _FakePolicyClass

    def make_pre_post_processors(policy_cfg: Any, pretrained_path: Any = None, **kw: Any) -> Any:
        calls["processors_cfg"] = policy_cfg
        calls["processors_path"] = pretrained_path
        calls["processors_kwargs"] = kw

        def preprocessor(obs: Any) -> Any:
            calls["preprocessed"] = obs
            return {"batch": obs}

        def postprocessor(tensor: FakeTensor) -> FakeTensor:
            calls.setdefault("postprocessed", []).append(tensor)
            return tensor

        preprocessor.steps = [  # type: ignore[attr-defined]
            SimpleNamespace(),
            SimpleNamespace(rename_map=rename_map or {}),
        ]
        return preprocessor, postprocessor

    class _PeftConfig:
        base_model_name_or_path = "base/model"
        revision = "abc123"

        @staticmethod
        def from_pretrained(path: str) -> _PeftConfig:
            calls["peft_config_path"] = path
            return _PeftConfig()

    class _PeftModel:
        @staticmethod
        def from_pretrained(base: Any, path: str, config: Any) -> Any:
            calls["peft_model"] = (base, path, config)
            return base

    class _PreTrainedConfig:
        @staticmethod
        def from_pretrained(path: str) -> SimpleNamespace:
            calls["pretrained_config_path"] = path
            return SimpleNamespace(path=path)

    def file_exists(repo: str, filename: str) -> bool:
        calls["hub_file_exists"] = (repo, filename)
        return hub_adapter

    peft = ModuleType("peft")
    peft.PeftConfig = _PeftConfig  # type: ignore[attr-defined]
    peft.PeftModel = _PeftModel  # type: ignore[attr-defined]
    hub = ModuleType("huggingface_hub")
    hub.file_exists = file_exists  # type: ignore[attr-defined]
    configs = ModuleType("lerobot.configs")
    configs_policies = ModuleType("lerobot.configs.policies")
    configs_policies.PreTrainedConfig = _PreTrainedConfig  # type: ignore[attr-defined]
    rtc = ModuleType("lerobot.policies.rtc")
    rtc_config = ModuleType("lerobot.policies.rtc.configuration_rtc")
    rtc_config.RTCConfig = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]

    torch_mod = ModuleType("torch")
    torch_mod.no_grad = nullcontext  # type: ignore[attr-defined]

    def _stack(tensors: list[FakeTensor], dim: int = 0) -> FakeTensor:
        return FakeTensor(np.stack([t.arr for t in tensors], axis=dim))

    torch_mod.stack = _stack  # type: ignore[attr-defined]
    torch_mod.from_numpy = lambda arr: arr  # type: ignore[attr-defined]

    lerobot = ModuleType("lerobot")
    policies = ModuleType("lerobot.policies")
    factory = ModuleType("lerobot.policies.factory")
    factory.get_policy_class = get_policy_class  # type: ignore[attr-defined]
    factory.make_pre_post_processors = make_pre_post_processors  # type: ignore[attr-defined]
    lerobot.policies = policies  # type: ignore[attr-defined]
    policies.factory = factory  # type: ignore[attr-defined]
    lerobot.configs = configs  # type: ignore[attr-defined]
    configs.policies = configs_policies  # type: ignore[attr-defined]

    modules = {
        "torch": torch_mod,
        "peft": peft,
        "huggingface_hub": hub,
        "lerobot.configs": configs,
        "lerobot.configs.policies": configs_policies,
        "lerobot": lerobot,
        "lerobot.policies": policies,
        "lerobot.policies.factory": factory,
        "lerobot.policies.rtc": rtc,
        "lerobot.policies.rtc.configuration_rtc": rtc_config,
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _seam_cfg(**over: Any) -> LeRobotPolicyConfig:
    return _policy_cfg(device="cpu", **over)


def test_default_predict_wires_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = np.arange(1 * 2 * 6, dtype=float).reshape(1, 2, 6)
    calls = _install_lerobot_fakes(monkeypatch, chunk=chunk)
    pol = LeRobotPolicy(_seam_cfg())  # no predict_fn: exercises the real seam
    pol.reset(Scene(id="s", instruction="grab it"))
    out = pol.act(_obs(instruction="grab it"))

    # Model load: factory class + from_pretrained + device + eval.
    assert calls["policy_types"] == ["smolvla"]
    assert calls["from_pretrained"] == "lerobot/smolvla_base"
    assert calls["hub_file_exists"] == ("lerobot/smolvla_base", "adapter_config.json")
    assert "peft_model" not in calls
    assert calls["to_device"] == "cpu"
    assert calls["eval"] is True
    # Processor pipelines built from the loaded policy's config with device overrides.
    assert calls["processors_cfg"] is calls["policy_config"]
    assert calls["processors_path"] == "lerobot/smolvla_base"
    assert calls["processors_kwargs"] == {
        "preprocessor_overrides": {"device_processor": {"device": "cpu"}},
        "postprocessor_overrides": {"device_processor": {"device": "cpu"}},
    }
    # The raw payload packed into the batch the preprocessor takes: state in motor
    # order, each camera as float CHW in 0..1 under observation.images.<cam>, task.
    batch = calls["preprocessed"]
    assert batch["task"] == "grab it"
    assert batch["observation.state"].shape == (1, 6)
    assert batch["observation.state"].dtype == np.float32
    image = batch["observation.images.front"]
    assert image.shape == (1, 3, 4, 4) and image.dtype == np.float32
    assert calls["predict_batch"] == {"batch": batch}
    assert len(calls["postprocessed"]) == 2
    assert len(out) == 2
    assert np.array_equal(out.actions[0].data, chunk[0, 0])
    assert np.array_equal(out.actions[1].data, chunk[0, 1])


def test_default_predict_unsqueezes_2d_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = np.arange(2 * 6, dtype=float).reshape(2, 6)  # no batch dim from the model
    _install_lerobot_fakes(monkeypatch, chunk=chunk)
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    out = pol.act(_obs())
    assert len(out) == 2
    assert np.array_equal(out.actions[1].data, chunk[1])


def test_default_predict_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, 6)))
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    pol.act(_obs())
    assert calls["policy_types"] == ["smolvla"]  # lazy closure built exactly once
    assert pol.num_inferences == 2


def test_bind_task_loads_before_the_first_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, 6)))
    pol = LeRobotPolicy(_seam_cfg())
    pol.bind_task(TaskEnvelope(name="t", max_steps=10))
    assert calls["policy_types"] == ["smolvla"]
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    assert calls["policy_types"] == ["smolvla"]  # not loaded again


def test_default_predict_rejects_camera_feature_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lerobot_fakes(
        monkeypatch,
        chunk=np.ones((1, 2, 6)),
        image_features={"observation.images.top": SimpleNamespace(shape=(3, 4, 4))},
    )
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=r"do not cover configured camera\(s\) \['front'\]"):
        pol.act(_obs())


def test_default_predict_loads_adapter_onto_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, 6)))
    pol = LeRobotPolicy(_seam_cfg(pretrained_path=str(tmp_path)))
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    # The base weights the adapter names, under the fine-tuned checkpoint's config.
    assert calls["from_pretrained"] == "base/model"
    assert calls["from_pretrained_kwargs"] == {
        "config": SimpleNamespace(path=str(tmp_path)),
        "revision": "abc123",
    }
    _, path, _ = calls["peft_model"]
    assert path == str(tmp_path)
    assert calls["processors_path"] == str(tmp_path)


def test_default_predict_local_dir_without_adapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, 6)))
    pol = LeRobotPolicy(_seam_cfg(pretrained_path=str(tmp_path)))
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    assert calls["from_pretrained"] == str(tmp_path)
    assert "hub_file_exists" not in calls and "peft_model" not in calls


def test_default_predict_detects_hub_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, 6)), hub_adapter=True)
    pol = LeRobotPolicy(_seam_cfg(pretrained_path="me/adapter"))
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    assert calls["peft_config_path"] == "me/adapter"
    assert calls["from_pretrained"] == "base/model"


def test_default_predict_accepts_renamed_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    # The checkpoint calls the camera "base_0_rgb"; its preprocessor renames "front".
    _install_lerobot_fakes(
        monkeypatch,
        chunk=np.ones((1, 2, 6)),
        image_features={"observation.images.base_0_rgb": SimpleNamespace(shape=(3, 4, 4))},
        rename_map={"observation.images.front": "observation.images.base_0_rgb"},
    )
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    assert len(pol.act(_obs())) == 2


def test_default_predict_stitches_chunks_with_rtc(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = np.arange(1 * 5 * 6, dtype=float).reshape(1, 5, 6)
    calls = _install_lerobot_fakes(monkeypatch, chunk=chunk)
    pol = LeRobotPolicy(_seam_cfg(rtc=True, chunk_size=2))
    pol.reset(Scene(id="s", instruction="x"))
    assert len(pol.act(_obs())) == 2
    pol.act(_obs())
    # The checkpoint's config is built with RTC over the executed horizon.
    assert calls["from_pretrained_kwargs"]["config"].rtc_config.execution_horizon == 2
    first, second = calls["predict_kwargs"]
    assert first == {"prev_chunk_left_over": None, "inference_delay": 0, "execution_horizon": 2}
    # The next chunk is guided onto the three actions the first one left unexecuted.
    assert np.array_equal(second["prev_chunk_left_over"].arr, chunk[:, 2:])
    # A new trial starts unguided.
    pol.reset(Scene(id="t", instruction="x"))
    pol.act(_obs())
    assert calls["predict_kwargs"][2]["prev_chunk_left_over"] is None


def test_default_predict_without_rtc_passes_no_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, 6)))
    pol = LeRobotPolicy(_seam_cfg())
    pol.reset(Scene(id="s", instruction="x"))
    pol.act(_obs())
    assert calls["predict_kwargs"] == [{}]


def test_default_predict_rejects_rtc_without_support(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_lerobot_fakes(monkeypatch, chunk=np.ones((1, 2, 6)), supports_rtc=False)
    pol = LeRobotPolicy(_seam_cfg(rtc=True))
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="needs a policy that supports RTC"):
        pol.act(_obs())


def test_smooth_reads_the_whole_chunk_off_the_cubic_before_truncating() -> None:
    chunk = np.random.default_rng(0).normal(size=(50, 6))
    pol = LeRobotPolicy(_policy_cfg(smooth=True, chunk_size=10), predict_fn=lambda obs: chunk)
    pol.reset(Scene(id="s", instruction="x"))
    out = np.array([a.data for a in pol.act(_obs()).actions])
    assert pol.config.smooth is True
    assert np.allclose(out, smooth(chunk)[:10])
