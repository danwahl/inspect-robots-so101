"""``LeRobotPolicy`` — a Inspect Robots policy backed by a LeRobot checkpoint.

LeRobot policies (ACT, SmolVLA, π0, diffusion, …) are ordinary ``nn.Module``\\ s
loaded from the Hugging Face Hub and run **in process** on the GPU. Unlike the
YAM/MolmoAct2 stack (where the model owns its own HTTP server), LeRobot models are
a library you import — so the heavy, GPU-bound dependencies (``torch`` +
``lerobot`` + the checkpoint) live behind a single injectable seam: a
``predict_fn`` that maps a **raw robot-style observation dict** (``"<motor>.pos"``
floats, camera frames keyed by camera name, and ``"task"``) to an action chunk.

The default ``predict_fn`` (``_default_predict``) lazily builds the policy and its
pre/post-processors from a pretrained path and packs each observation into the batch
those processors expect: ``observation.state`` in motor order,
``observation.images.<cam>`` as float CHW in 0..1, and ``task``. Images go at the
camera's resolution; the checkpoint's own processors and model resize them. Tests
exercise this wiring with ``sys.modules`` fakes (no torch, no lerobot, no network),
and the ``lerobot-seam`` CI job re-imports the real symbols on Python 3.12 to catch
upstream drift. ``import inspect_robots_so101`` never imports torch.

The action chunk this policy returns is already in the robot's **native motor
units** (degrees for the joints, 0..100 for the gripper): LeRobot's postprocessor
unnormalizes for us, so the embodiment commands the values verbatim (after its
safety clamp). There is therefore no gripper renormalization here.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_so101 import packing
from inspect_robots_so101.config import LeRobotPolicyConfig, action_box, observation_space

# A RAW robot-style observation dict -> an (N, action_dim) action-chunk array.
# Keys: "<motor>.pos" floats (packing.motor_keys()), camera frames keyed by the
# camera name, and "task" (the instruction) — exactly what a lerobot Robot's
# get_observation() returns, plus the task string. The default predict_fn packs it
# into "observation.state" / "observation.images.<cam>" itself.
LeRobotObs = Mapping[str, Any]
PredictFn = Callable[[LeRobotObs], npt.NDArray[np.floating[Any]]]

# LeRobot key conventions (see lerobot/utils/constants.py). Hardcoded so this
# module needs no lerobot import.
OBS_STATE = "observation.state"
OBS_IMAGE_PREFIX = "observation.images."
TASK_KEY = "task"
# What PEFT saves beside a LoRA adapter's weights.
ADAPTER_CONFIG = "adapter_config.json"


def _batch(obs: LeRobotObs, cameras: tuple[str, ...]) -> dict[str, Any]:
    """Pack the raw payload into the unprocessed batch a LeRobot preprocessor takes.

    State is ``(1, 6)`` float32 in motor order; each camera becomes
    ``observation.images.<cam>``, ``(1, 3, H, W)`` float32 in 0..1.
    """
    state = np.asarray([obs[k] for k in packing.motor_keys()], dtype=np.float32)[None]
    batch: dict[str, Any] = {OBS_STATE: state, TASK_KEY: obs[TASK_KEY]}
    for cam in cameras:
        image = np.asarray(obs[cam], dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        batch[f"{OBS_IMAGE_PREFIX}{cam}"] = np.ascontiguousarray(image)
    return batch


def _has_adapter(path: str) -> bool:
    """Whether the checkpoint at ``path`` (a local directory or a Hub repo) is a PEFT adapter."""
    if Path(path).is_dir():
        return (Path(path) / ADAPTER_CONFIG).is_file()
    from huggingface_hub import file_exists

    return bool(file_exists(path, ADAPTER_CONFIG))


def _load_policy(cfg: LeRobotPolicyConfig) -> Any:
    """The checkpoint's policy, with its LoRA adapter applied if it is one.

    An adapter is loaded the way ``lerobot.policies.factory.make_policy`` loads one:
    the base weights its ``adapter_config.json`` names, built under the fine-tuned
    checkpoint's own config, then the adapter on top.
    """
    from lerobot.policies.factory import get_policy_class

    policy_cls = get_policy_class(cfg.policy_type)
    if not _has_adapter(cfg.pretrained_path):
        return policy_cls.from_pretrained(cfg.pretrained_path)
    from lerobot.configs.policies import PreTrainedConfig
    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(cfg.pretrained_path)
    base = policy_cls.from_pretrained(
        peft_config.base_model_name_or_path,
        config=PreTrainedConfig.from_pretrained(cfg.pretrained_path),
        revision=peft_config.revision,
    )
    return PeftModel.from_pretrained(base, cfg.pretrained_path, config=peft_config)


def _rename_map(preprocessor: Any) -> dict[str, str]:
    """The observation renaming a checkpoint's saved preprocessor applies, if any."""
    for step in getattr(preprocessor, "steps", ()):
        rename = getattr(step, "rename_map", None)
        if rename:
            return dict(rename)
    return {}


def _default_predict(cfg: LeRobotPolicyConfig) -> PredictFn:
    """Build an in-process LeRobot inference closure from a pretrained checkpoint.

    Load the policy class from the factory, build the matching pre/post-processor
    pipelines, then per call pack the batch → preprocess → ``predict_action_chunk`` →
    postprocess back to native motor units.
    """
    import torch
    from lerobot.policies.factory import make_pre_post_processors

    policy = _load_policy(cfg)
    policy.to(cfg.device)
    policy.eval()  # torch nn.Module inference mode (not Python's eval builtin)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
        postprocessor_overrides={"device_processor": {"device": cfg.device}},
    )

    # A checkpoint trained under a rename_map keeps the policy-side names in its
    # config and renames the cameras in its saved preprocessor.
    rename = _rename_map(preprocessor)
    policy_image_features = dict(policy.config.image_features)
    missing = [
        cam
        for cam in cfg.cameras
        if rename.get(f"{OBS_IMAGE_PREFIX}{cam}", f"{OBS_IMAGE_PREFIX}{cam}")
        not in policy_image_features
    ]
    if missing:
        raise ValueError(
            f"checkpoint {cfg.pretrained_path!r} was trained with image features "
            f"{sorted(policy_image_features)} which do not cover configured camera(s) "
            f"{missing} (expected {[OBS_IMAGE_PREFIX + cam for cam in missing]}); "
            "set LeRobotPolicyConfig.cameras to the camera names the checkpoint expects"
        )

    def _predict(obs: LeRobotObs) -> npt.NDArray[np.floating[Any]]:
        observation = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in _batch(obs, cfg.cameras).items()
        }
        batch = preprocessor(observation)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        out = torch.stack(
            [postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1
        ).squeeze(0)
        return out.detach().cpu().numpy()  # type: ignore[no-any-return]

    return _predict


class LeRobotPolicy:
    """Inspect Robots policy wrapping a LeRobot checkpoint for the SO-ARM action space."""

    def __init__(
        self,
        config: LeRobotPolicyConfig | None = None,
        *,
        predict_fn: PredictFn | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else LeRobotPolicyConfig.from_kwargs(**flat)
        self._predict_fn = predict_fn
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name="lerobot",
            action_space=action_box(),  # semantics only; the embodiment owns limits
            observation_space=observation_space(
                self._cfg.cam_height,
                self._cfg.cam_width,
                self._cfg.cameras,
                use_degrees=self._cfg.use_degrees,
            ),
            # Intentionally None: advertising a rate would trip a (harmless) compat
            # control_rate warning. The embodiment paces the rollout.
            control_hz=None,
        )
        self.config = PolicyConfig(action_horizon=self._cfg.chunk_size)

    def _predict(self) -> PredictFn:
        """Lazily build the real inference closure on first use."""
        if self._predict_fn is None:
            self._predict_fn = _default_predict(self._cfg)
        return self._predict_fn

    def reset(self, scene: Scene) -> None:
        """Stash the scene's instruction (fed to the VLA verbatim)."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Build the raw robot-style observation, run inference, return the chunk.

        The chunk is truncated to the first ``chunk_size`` actions (see
        ``LeRobotPolicyConfig.chunk_size``).
        """
        cfg = self._cfg
        try:
            images = {cam: observation.images[cam] for cam in cfg.cameras}
        except KeyError as exc:
            raise ValueError(f"observation missing camera {exc} for lerobot policy") from exc
        expected = (cfg.cam_height, cfg.cam_width, 3)
        for name, img in images.items():
            if img.shape != expected:
                raise ValueError(f"camera {name!r} returned shape {img.shape}, expected {expected}")
        if cfg.state_key not in observation.state:
            raise ValueError(f"observation missing state key {cfg.state_key!r}")
        state = packing.validate_dim(observation.state[cfg.state_key])

        # The raw robot format: "<motor>.pos" floats, camera frames keyed by camera
        # name, and "task".
        lerobot_obs: dict[str, Any] = {
            **packing.to_action_dict(state),
            TASK_KEY: self._instruction or "",
            **images,
        }

        t0 = time.perf_counter()
        raw = self._predict()(lerobot_obs)
        elapsed = time.perf_counter() - t0

        actions = np.asarray(raw, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != packing.TOTAL_DIM:
            raise ValueError(
                f"policy returned actions of shape {actions.shape}; "
                f"expected (N, {packing.TOTAL_DIM})"
            )
        if actions.shape[0] == 0:
            raise ValueError("policy returned an empty action chunk")
        actions = actions[: cfg.chunk_size]

        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in actions],
            control_hz=None,
            inference_latency_s=elapsed,
        )
