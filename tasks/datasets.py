import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
except ModuleNotFoundError:
    LeRobotDataset = None
    LeRobotDatasetMetadata = None


def _make_chunk_starts(start: int, end: int, action_horizon: int):

    if end - start < action_horizon:
        return []

    last_valid_i = end - action_horizon
    starts = list(range(start, last_valid_i, action_horizon))
    if not starts or starts[-1] != last_valid_i:
        starts.append(last_valid_i)
    return starts


class LiberoLeRobotQDataset(Dataset):
    def __init__(
        self,
        repo_id: str = "HuggingFaceVLA/libero",
        action_horizon: int = 10,
        action_dim: int = 7,
        image_size: int = 64,
        task_id: int | None = None,
    ):
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.image_size = image_size
        if LeRobotDataset is None or LeRobotDatasetMetadata is None:
            raise ModuleNotFoundError(
                "lerobot.datasets is required for LiberoLeRobotQDataset. "
                "Use LiberoCriticDataTaskDataset for saved critic_data files."
            )
        meta = LeRobotDatasetMetadata(repo_id)
        H = action_horizon

        task = None
        if task_id is not None:
            if task_id not in meta.tasks["task_index"].values:
                raise ValueError(
                    f"Unknown task_id {task_id!r}. Available task_ids: " f"{sorted(meta.tasks['task_index'].tolist())}"
                )
            task = meta.tasks.index[meta.tasks["task_index"] == task_id][0]

        delta_timestamps = {
            "action": [i / meta.fps for i in range(2 * H)],
            "observation.images.image": [i / meta.fps for i in range(H + 1)],
            "observation.images.image2": [i / meta.fps for i in range(H + 1)],
        }
        self.dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

        self.chunk_starts = []
        self.chunk_ends = []
        for episode in meta.episodes:
            if task is not None and task not in episode["tasks"]:
                continue
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            for chunk_start in _make_chunk_starts(start, end, action_horizon):
                self.chunk_starts.append(chunk_start)
                self.chunk_ends.append(end)

    def __len__(self):
        return len(self.chunk_starts)

    def _images(self, x):

        x = torch.as_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.max() > 2.0:
            x = x / 255.0
        if tuple(x.shape[-2:]) != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if tuple(x.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected image shape (T,3,{self.image_size},{self.image_size}), got {tuple(x.shape)}")
        return x

    def __getitem__(self, i):
        chunk_start = self.chunk_starts[i]
        episode_end = self.chunk_ends[i]
        H = self.action_horizon

        sample = self.dataset[chunk_start]

        images = self._images(sample["observation.images.image"])
        images2 = self._images(sample["observation.images.image2"])

        action_flat = torch.as_tensor(sample["action"]).float()
        if action_flat.ndim == 1:
            action_flat = action_flat.view(2 * H, self.action_dim)

        done = torch.tensor(float(chunk_start + H >= episode_end))

        image = images[0]
        image2 = images2[0]
        action = action_flat[:H]

        if done.item():
            next_image, next_image2 = image, image2
            next_action = action
        else:
            next_image, next_image2 = images[H], images2[H]
            next_action = action_flat[H : 2 * H]

        if "reward" in sample:
            reward = torch.as_tensor(sample["reward"]).float()
            reward = reward.max().squeeze() if reward.numel() > 1 else reward.squeeze()
        else:
            reward = done.clone()

        return {
            "image": image,
            "image2": image2,
            "action": action,
            "reward": reward,
            "next_image": next_image,
            "next_image2": next_image2,
            "next_action": next_action,
            "done": done,
            "valid": 1.0 - done,
            "executed_steps": torch.tensor(float(H)),
        }


class LiberoCriticDataTaskDataset(Dataset):

    def __init__(
        self,
        critic_data_root: str = "critic_data/smolvla_clean_action",
        suite: str = "libero_goal",
        task_id: int = 0,
        action_horizon: int = 10,
        action_dim: int = 7,
        image_size: int = 64,
        action_key: str = "a0_env",
        image2_key: str = "observation.images.image2",
    ):
        self.critic_data_root = Path(critic_data_root)
        self.suite = suite
        self.task_id = task_id
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.image_size = image_size
        self.action_key = action_key

        self.image2_key = image2_key

        self.path = self.critic_data_root / suite / f"task{task_id:02d}.pt"
        if not self.path.exists():
            raise FileNotFoundError(f"Critic data task file not found: {self.path}")

        records = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(records, list):
            raise ValueError(f"Expected a list of records in {self.path}, got {type(records).__name__}")
        if len(records) == 0:
            raise ValueError(f"Critic data task file is empty: {self.path}")

        self.records = sorted(
            records, key=lambda r: (int(r.get("episode_id", -1)), int(r.get("step", -1)), int(r.get("chunk_id", -1)))
        )
        self.next_indices = []
        for i, record in enumerate(self.records):
            if i + 1 < len(self.records) and self.records[i + 1].get("episode_id") == record.get("episode_id"):
                self.next_indices.append(i + 1)
            else:
                self.next_indices.append(None)

    def __len__(self):
        return len(self.records)

    def _image(self, x):
        x = torch.as_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.max() > 2.0:
            x = x / 255.0
        if tuple(x.shape[-2:]) != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if tuple(x.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected image shape (T,3,{self.image_size},{self.image_size}), got {tuple(x.shape)}")
        if x.shape[0] != 1:
            raise ValueError(f"Expected one image frame per critic_data record, got {x.shape[0]}")
        return x[0]

    def _action(self, record):
        if self.action_key not in record:
            raise KeyError(f"Action key {self.action_key!r} not found in record")

        action = torch.as_tensor(record[self.action_key]).float()
        if action.ndim == 3:
            if action.shape[0] != 1:
                raise ValueError(f"Expected action batch dimension 1, got {tuple(action.shape)}")
            action = action[0]
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action shape (T,{self.action_dim}), got {tuple(action.shape)}")
        if action.shape[0] < self.action_horizon:
            raise ValueError(f"Need at least {self.action_horizon} action steps, got {action.shape[0]}")
        return action[: self.action_horizon]

    def __getitem__(self, i):
        record = self.records[i]
        batch = record["batch"]
        image = self._image(batch["observation.images.image"])
        image2 = self._image(batch[self.image2_key])
        action = self._action(record)

        next_i = self.next_indices[i]
        done = bool(batch.get("next.done", False) or batch.get("next.truncated", False) or next_i is None)
        if done:
            next_image = image
            next_image2 = image2
            next_action = action
        else:
            next_record = self.records[next_i]
            next_batch = next_record["batch"]
            next_image = self._image(next_batch["observation.images.image"])
            next_image2 = self._image(next_batch[self.image2_key])
            next_action = self._action(next_record)

        success_step = record.get("success_step")
        step = int(record.get("step", 0))
        executed_steps = int(record.get("executed_steps", self.action_horizon))
        reward = float(success_step is not None and step <= int(success_step) < step + executed_steps)

        return {
            "image": image,
            "image2": image2,
            "action": action,
            "reward": torch.tensor(reward),
            "next_image": next_image,
            "next_image2": next_image2,
            "next_action": next_action,
            "done": torch.tensor(float(done)),
            "valid": torch.tensor(float(not done)),
            "executed_steps": torch.tensor(float(executed_steps)),
            "episode_id": torch.tensor(int(record.get("episode_id", -1))),
            "chunk_id": torch.tensor(int(record.get("chunk_id", -1))),
            "step": torch.tensor(int(record.get("step", -1))),
            "task_id": torch.tensor(int(record.get("task_id", self.task_id))),
        }


class LiberoCriticDataAGFDataset(LiberoCriticDataTaskDataset):

    def __getitem__(self, i):
        record = self.records[i]
        out = dict(record["batch"])
        out["_success"] = torch.tensor(float(record.get("success", 0.0)))
        out["_step"] = torch.tensor(int(record.get("step", -1)))
        out["_chunk_id"] = torch.tensor(int(record.get("chunk_id", -1)))
        out["_task_id"] = torch.tensor(int(record.get("task_id", self.task_id)))
        return out


class RoboCasaCriticDataTaskDataset(Dataset):

    CAMERA_KEYS = (
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_right",
    )

    def __init__(
        self,
        critic_data_root: str = "critic_data/pi05_robocasa_clean_action",
        group_name: str = "atomic",
        task_name: str = None,
        action_horizon: int = 50,
        action_dim: int = 12,
        image_size: int = 64,
        action_key: str = "a0_raw",
    ):
        if task_name is None:
            raise ValueError(
                "task_name is required for RoboCasaCriticDataTaskDataset "
                "(the QPILOTS-RoboCasa critic is trained per task)."
            )

        self.critic_data_root = Path(critic_data_root)
        self.group_name = group_name
        self.task_name = task_name
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.image_size = image_size
        self.action_key = action_key

        group_dir = self.critic_data_root / group_name
        summary_path = group_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"summary.json not found: {summary_path}")
        with open(summary_path) as f:
            summary = json.load(f)

        entry = next((v for v in summary.values() if v["task"] == task_name), None)
        if entry is None:
            available = sorted(v["task"] for v in summary.values())
            raise ValueError(f"Unknown task_name {task_name!r}. Available: {available}")

        self.path = Path(entry["path"])
        if not self.path.exists():
            raise FileNotFoundError(f"Critic data task file not found: {self.path}")

        records = torch.load(self.path, map_location="cpu", weights_only=False)
        if not isinstance(records, list):
            raise ValueError(f"Expected a list of records in {self.path}, got {type(records).__name__}")
        if len(records) == 0:
            raise ValueError(f"Critic data task file is empty: {self.path}")

        self.records = sorted(
            records, key=lambda r: (int(r.get("episode_id", -1)), int(r.get("step", -1)), int(r.get("chunk_id", -1)))
        )
        self.next_indices = []
        for i, record in enumerate(self.records):
            if i + 1 < len(self.records) and self.records[i + 1].get("episode_id") == record.get("episode_id"):
                self.next_indices.append(i + 1)
            else:
                self.next_indices.append(None)

    def __len__(self):
        return len(self.records)

    def _image(self, x):
        x = torch.as_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        if x.max() > 2.0:
            x = x / 255.0
        if tuple(x.shape[-2:]) != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if tuple(x.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected image shape (T,3,{self.image_size},{self.image_size}), got {tuple(x.shape)}")
        if x.shape[0] != 1:
            raise ValueError(f"Expected one image frame per critic_data record, got {x.shape[0]}")
        return x[0]

    def _images(self, batch):
        return tuple(self._image(batch[k]) for k in self.CAMERA_KEYS)

    def _state(self, batch):
        state = torch.as_tensor(batch["observation.state"]).float()
        if state.ndim == 2:
            if state.shape[0] != 1:
                raise ValueError(f"Expected state batch dimension 1, got {tuple(state.shape)}")
            state = state[0]
        return state

    def _action(self, record):
        if self.action_key not in record:
            raise KeyError(f"Action key {self.action_key!r} not found in record")

        action = torch.as_tensor(record[self.action_key]).float()
        if action.ndim == 3:
            if action.shape[0] != 1:
                raise ValueError(f"Expected action batch dimension 1, got {tuple(action.shape)}")
            action = action[0]
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action shape (T,{self.action_dim}), got {tuple(action.shape)}")
        if action.shape[0] < self.action_horizon:
            raise ValueError(f"Need at least {self.action_horizon} action steps, got {action.shape[0]}")
        return action[: self.action_horizon]

    def __getitem__(self, i):
        record = self.records[i]
        batch = record["batch"]
        image, image2, image3 = self._images(batch)
        state = self._state(batch)
        action = self._action(record)

        next_i = self.next_indices[i]
        done = bool(batch.get("next.done", False) or batch.get("next.truncated", False) or next_i is None)
        if done:
            next_image, next_image2, next_image3 = image, image2, image3
            next_state = state
            next_action = action
        else:
            next_record = self.records[next_i]
            next_batch = next_record["batch"]
            next_image, next_image2, next_image3 = self._images(next_batch)
            next_state = self._state(next_batch)
            next_action = self._action(next_record)

        success_step = record.get("success_step")
        step = int(record.get("step", 0))
        executed_steps = int(record.get("executed_steps", self.action_horizon))
        reward = float(success_step is not None and step <= int(success_step) < step + executed_steps)

        return {
            "image": image,
            "image2": image2,
            "image3": image3,
            "agent_pos": state,
            "action": action,
            "reward": torch.tensor(reward),
            "next_image": next_image,
            "next_image2": next_image2,
            "next_image3": next_image3,
            "next_agent_pos": next_state,
            "next_action": next_action,
            "done": torch.tensor(float(done)),
            "valid": torch.tensor(float(not done)),
            "executed_steps": torch.tensor(float(executed_steps)),
            "episode_id": torch.tensor(int(record.get("episode_id", -1))),
            "chunk_id": torch.tensor(int(record.get("chunk_id", -1))),
            "step": torch.tensor(int(record.get("step", -1))),
        }


class RoboCasaCriticDataAGFDataset(RoboCasaCriticDataTaskDataset):

    def __getitem__(self, i):
        record = self.records[i]
        out = dict(record["batch"])
        out["_success"] = torch.tensor(float(record.get("success", 0.0)))
        out["_step"] = torch.tensor(int(record.get("step", -1)))
        out["_chunk_id"] = torch.tensor(int(record.get("chunk_id", -1)))
        return out


def make_robocasa_critic_dataloader(
    critic_data_root: str = "critic_data/pi05_robocasa_clean_action",
    group_name: str = "atomic",
    task_name: str = None,
    action_horizon: int = 50,
    action_dim: int = 12,
    image_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    action_key: str = "a0_raw",
):
    dataset = RoboCasaCriticDataTaskDataset(
        critic_data_root=critic_data_root,
        group_name=group_name,
        task_name=task_name,
        action_horizon=action_horizon,
        action_dim=action_dim,
        image_size=image_size,
        action_key=action_key,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return dataset, loader


def make_libero_q_dataloader(
    repo_id: str = "HuggingFaceVLA/libero",
    action_horizon: int = 10,
    action_dim: int = 7,
    image_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    task_id: int | None = None,
):
    dataset = LiberoLeRobotQDataset(
        repo_id=repo_id, action_horizon=action_horizon, action_dim=action_dim, image_size=image_size, task_id=task_id
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return dataset, loader


def make_critic_data_task_dataloader(
    critic_data_root: str = "critic_data/smolvla_clean_action",
    suite: str = "libero_goal",
    task_id: int = 0,
    action_horizon: int = 10,
    action_dim: int = 7,
    image_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    action_key: str = "a0_env",
    image2_key: str = "observation.images.image2",
):
    dataset = LiberoCriticDataTaskDataset(
        critic_data_root=critic_data_root,
        suite=suite,
        task_id=task_id,
        action_horizon=action_horizon,
        action_dim=action_dim,
        image_size=image_size,
        action_key=action_key,
        image2_key=image2_key,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return dataset, loader
