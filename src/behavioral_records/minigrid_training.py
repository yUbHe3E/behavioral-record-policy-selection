from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np

from behavioral_records.canonical import stable_hash


HIERARCHY_PHASES = (
    "complete_target",
    "unlock_door",
    "park_carried_object",
    "open_passage_door",
    "acquire_key",
    "open_key_box",
    "move_door_blocker",
    "clear_access_blocker",
)


def make_onehot_cnn_env(
    environment_id: str,
    *,
    target_height: int = 16,
    target_width: int = 16,
) -> Any:
    """Build an official MiniGrid task with a task-agnostic spatial encoding.

    The wrapper changes only the observation representation. Environment
    dynamics, termination, and reward remain those registered by MiniGrid.
    Smaller curriculum layouts are zero-padded to a shared size so one policy
    can be transferred prospectively across official task variants.
    """

    import gymnasium as gym
    import minigrid  # noqa: F401
    from gymnasium import ObservationWrapper, spaces
    from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
    from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    class FullOneHotObservation(ObservationWrapper):
        """One-hot encode full observations, including four agent directions.

        MiniGrid's partial-observation one-hot wrapper allocates only the three
        door states. In a full observation the agent tile reuses that field for
        direction 0..3, so a fourth state slot is required.
        """

        def __init__(self, env: Any) -> None:
            super().__init__(env)
            source_shape = env.observation_space["image"].shape
            self._object_bits = len(OBJECT_TO_IDX)
            self._color_bits = len(COLOR_TO_IDX)
            self._state_bits = max(len(STATE_TO_IDX), 4)
            channels = self._object_bits + self._color_bits + self._state_bits
            image_space = spaces.Box(
                low=0,
                high=1,
                shape=(source_shape[0], source_shape[1], channels),
                dtype=np.uint8,
            )
            self.observation_space = spaces.Dict(
                {**env.observation_space.spaces, "image": image_space}
            )

        def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
            image = observation["image"]
            output = np.zeros(self.observation_space["image"].shape, dtype=np.uint8)
            for row in range(image.shape[0]):
                for column in range(image.shape[1]):
                    object_index, color_index, state_index = (
                        int(value) for value in image[row, column]
                    )
                    output[row, column, object_index] = 1
                    output[row, column, self._object_bits + color_index] = 1
                    output[
                        row,
                        column,
                        self._object_bits + self._color_bits + state_index,
                    ] = 1
            return {**observation, "image": output}

    class PadSpatialObservation(ObservationWrapper):
        def __init__(self, env: Any) -> None:
            super().__init__(env)
            source_shape = tuple(int(value) for value in env.observation_space.shape)
            if len(source_shape) != 3:
                raise ValueError(f"Expected HWC observation, received {source_shape}")
            source_height, source_width, channels = source_shape
            if source_height > target_height or source_width > target_width:
                raise ValueError(
                    "Source grid does not fit frozen padding target: "
                    f"source={source_shape}, target=({target_height}, {target_width})"
                )
            self._source_height = source_height
            self._source_width = source_width
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(target_height, target_width, channels),
                dtype=np.uint8,
            )

        def observation(self, observation: np.ndarray) -> np.ndarray:
            output = np.zeros(self.observation_space.shape, dtype=np.uint8)
            output[: self._source_height, : self._source_width, :] = observation
            return output

    env = gym.make(environment_id)
    env = FullyObsWrapper(env)
    env = FullOneHotObservation(env)
    env = ImgObsWrapper(env)
    return PadSpatialObservation(env)


def small_grid_policy_kwargs() -> dict[str, Any]:
    """Return SB3 policy arguments for HWC one-hot MiniGrid observations."""

    import torch
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class SmallGridExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: Any, features_dim: int = 256) -> None:
            super().__init__(observation_space, features_dim)
            height, width, channels = observation_space.shape
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Flatten(),
            )
            with torch.no_grad():
                sample = torch.zeros(1, channels, height, width)
                flattened = int(self.cnn(sample).shape[1])
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(flattened, features_dim),
                torch.nn.ReLU(),
            )

        def forward(self, observations: Any) -> Any:
            # The custom wrapper deliberately exposes HWC. MiniGrid one-hot
            # values are already in [0, 1], so no image normalization is used.
            channel_first = observations.float().permute(0, 3, 1, 2)
            return self.projection(self.cnn(channel_first))

    return {
        "features_extractor_class": SmallGridExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
        "net_arch": {"pi": [128], "vf": [128]},
    }


def spatial_grid_policy_kwargs() -> dict[str, Any]:
    """Return a coordinate-aware extractor that preserves exact grid position."""

    import torch
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class SpatialGridExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: Any, features_dim: int = 256) -> None:
            super().__init__(observation_space, features_dim)
            height, width, channels = observation_space.shape
            self.height = int(height)
            self.width = int(width)
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(channels + 2, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
            )
            flattened = 32 * self.height * self.width
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(flattened, features_dim),
                torch.nn.ReLU(),
            )

        def forward(self, observations: Any) -> Any:
            channel_first = observations.float().permute(0, 3, 1, 2)
            batch = channel_first.shape[0]
            y = torch.linspace(-1.0, 1.0, self.height, device=observations.device)
            x = torch.linspace(-1.0, 1.0, self.width, device=observations.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            coordinates = torch.stack((yy, xx), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
            encoded = torch.cat((channel_first, coordinates), dim=1)
            return self.projection(self.cnn(encoded))

    return {
        "features_extractor_class": SpatialGridExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
        "net_arch": {"pi": [128], "vf": [128]},
    }


def flat_onehot_policy_kwargs() -> dict[str, Any]:
    """Return a position-preserving MLP for the padded one-hot grid."""

    return {
        "normalize_images": False,
        "net_arch": {"pi": [512, 256], "vf": [256, 128]},
    }


def _grid_objects(env: Any, *, object_type: str) -> list[tuple[int, int]]:
    base = env.unwrapped
    positions: list[tuple[int, int]] = []
    for x in range(base.grid.width):
        for y in range(base.grid.height):
            cell = base.grid.get(x, y)
            if cell is not None and cell.type == object_type:
                positions.append((x, y))
    return positions


def plan_to_face(env: Any, target: tuple[int, int]) -> list[int]:
    """Shortest action plan that faces a target without entering its cell."""

    from minigrid.core.constants import DIR_TO_VEC

    base = env.unwrapped
    actions = base.actions
    start = (int(base.agent_pos[0]), int(base.agent_pos[1]), int(base.agent_dir))
    frontier: deque[tuple[int, int, int]] = deque([start])
    parents: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {
        start: None
    }

    def reconstruct(state: tuple[int, int, int]) -> list[int]:
        route: list[int] = []
        while parents[state] is not None:
            prior, action = parents[state]  # type: ignore[misc]
            route.append(action)
            state = prior
        route.reverse()
        return route

    while frontier:
        x, y, direction = frontier.popleft()
        dx, dy = (int(value) for value in DIR_TO_VEC[direction])
        if (x + dx, y + dy) == target:
            return reconstruct((x, y, direction))

        successors = [
            ((x, y, (direction - 1) % 4), int(actions.left)),
            ((x, y, (direction + 1) % 4), int(actions.right)),
        ]
        forward = (x + dx, y + dy, direction)
        if 0 <= forward[0] < base.grid.width and 0 <= forward[1] < base.grid.height:
            cell = base.grid.get(forward[0], forward[1])
            if cell is None or cell.can_overlap():
                successors.append((forward, int(actions.forward)))
        for successor, action in successors:
            if successor not in parents:
                parents[successor] = ((x, y, direction), action)
                frontier.append(successor)
    raise RuntimeError(f"No reachable position faces target {target}")


def plan_to_face_any(
    env: Any,
    targets: Iterable[tuple[int, int]],
) -> tuple[list[int], tuple[int, int]]:
    candidates: list[tuple[int, tuple[int, int], list[int]]] = []
    for target in targets:
        try:
            plan = plan_to_face(env, target)
        except RuntimeError:
            continue
        candidates.append((len(plan), target, plan))
    if not candidates:
        raise RuntimeError("No reachable target-facing state")
    _, target, plan = min(candidates, key=lambda row: (row[0], row[1]))
    return plan, target


def _reachable_plan_rows(
    env: Any,
    positions: Iterable[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], list[int]]]:
    rows: list[tuple[int, tuple[int, int], list[int]]] = []
    for position in positions:
        try:
            plan = plan_to_face(env, position)
        except RuntimeError:
            continue
        rows.append((len(plan), position, plan))
    return sorted(rows, key=lambda row: (row[0], row[1]))


def _safe_drop_advice(env: Any) -> tuple[int, tuple[int, int]]:
    base = env.unwrapped
    door_positions = set(_grid_objects(env, object_type="door"))
    forbidden = set(door_positions)
    for door_x, door_y in door_positions:
        forbidden.update(
            {
                (door_x - 1, door_y),
                (door_x + 1, door_y),
                (door_x, door_y - 1),
                (door_x, door_y + 1),
            }
        )
    empty_cells = [
        (x, y)
        for x in range(1, base.grid.width - 1)
        for y in range(1, base.grid.height - 1)
        if base.grid.get(x, y) is None
        and (x, y) not in forbidden
        and (x, y) != tuple(int(value) for value in base.agent_pos)
    ]
    rows = _reachable_plan_rows(env, empty_cells)
    if not rows:
        raise RuntimeError("No safe reachable drop position")

    def clearance(position: tuple[int, int]) -> int:
        return min(
            abs(position[0] - door[0]) + abs(position[1] - door[1])
            for door in door_positions
        )

    _, target, plan = min(rows, key=lambda row: (-clearance(row[1]), row[0], row[1]))
    return (plan[0] if plan else int(base.actions.drop)), target


def _access_blocker_advice(
    env: Any,
    *,
    target_positions: Iterable[tuple[int, int]],
    locked_doors: Iterable[tuple[int, int]],
    blocking_balls: Iterable[tuple[int, int]],
) -> tuple[int, tuple[int, int]] | None:
    """Find a reachable movable object whose removal exposes real progress.

    ObstructedMaze can place a key in the only standing cell from which a
    door-blocking ball or key box can be manipulated.  A purely one-object
    heuristic then sees neither downstream object as reachable.  This helper
    performs a deterministic one-step counterfactual: temporarily remove each
    reachable non-target key/ball and retain it only when that makes a target,
    locked door, key box, or door blocker newly reachable.  The simulator is
    restored before returning; the chosen action still executes normally.
    """

    base = env.unwrapped
    target_positions = list(target_positions)
    locked_doors = list(locked_doors)
    blocking_balls = list(blocking_balls)
    key_boxes = [
        position
        for position in _grid_objects(env, object_type="box")
        if (box := base.grid.get(*position)).contains is not None
        and box.contains.type == "key"
    ]
    progress_targets = sorted(
        set(target_positions + locked_doors + blocking_balls + key_boxes)
    )
    baseline_reachable = {
        position for _, position, _ in _reachable_plan_rows(env, progress_targets)
    }
    movable_positions = [
        position
        for object_type in ("key", "ball")
        for position in _grid_objects(env, object_type=object_type)
        if base.grid.get(*position) is not base.obj
    ]
    candidates: list[
        tuple[int, int, tuple[int, int], list[int]]
    ] = []
    for plan_length, position, plan in _reachable_plan_rows(env, movable_positions):
        obj = base.grid.get(*position)
        if obj is None:
            continue
        base.grid.set(*position, None)
        try:
            newly_reachable = {
                candidate
                for _, candidate, _ in _reachable_plan_rows(env, progress_targets)
                if candidate != position
            } - baseline_reachable
        finally:
            base.grid.set(*position, obj)
        if newly_reachable:
            candidates.append((-len(newly_reachable), plan_length, position, plan))
    if not candidates:
        return None
    _, _, position, plan = min(candidates)
    return (plan[0] if plan else int(base.actions.pickup)), position


def obstructedmaze_expert_advice(env: Any) -> dict[str, Any]:
    """Return one training-only expert action for the current simulator state.

    Unlike a fixed demonstration, this advisor can label states visited after a
    learned policy deviates. It never receives a policy-selection requirement.
    """

    base = env.unwrapped
    actions = base.actions
    target_positions = [
        (x, y)
        for x in range(base.grid.width)
        for y in range(base.grid.height)
        if base.grid.get(x, y) is base.obj
    ]
    target_rows = _reachable_plan_rows(env, target_positions)
    if target_rows and base.carrying is None:
        plan = target_rows[0][2]
        return {
            "action": plan[0] if plan else int(actions.pickup),
            "phase": "complete_target",
            "subgoal": target_rows[0][1],
        }

    door_positions = _grid_objects(env, object_type="door")
    locked_doors = [
        position
        for position in door_positions
        if (door := base.grid.get(*position)) is not None
        and door.type == "door"
        and door.is_locked
    ]
    if base.carrying is not None:
        matching_rows: list[tuple[int, tuple[int, int], list[int]]] = []
        if base.carrying.type == "key":
            matching_rows = _reachable_plan_rows(
                env,
                (
                    position
                    for position in locked_doors
                    if base.grid.get(*position).color == base.carrying.color
                ),
            )
        if matching_rows:
            plan = matching_rows[0][2]
            return {
                "action": plan[0] if plan else int(actions.toggle),
                "phase": "unlock_door",
                "subgoal": matching_rows[0][1],
            }
        action, position = _safe_drop_advice(env)
        return {"action": action, "phase": "park_carried_object", "subgoal": position}

    unlocked_closed = [
        position
        for position in door_positions
        if (door := base.grid.get(*position)) is not None
        and door.type == "door"
        and not door.is_locked
        and not door.is_open
    ]
    unlocked_rows = _reachable_plan_rows(env, unlocked_closed)
    if unlocked_rows:
        plan = unlocked_rows[0][2]
        return {
            "action": plan[0] if plan else int(actions.toggle),
            "phase": "open_passage_door",
            "subgoal": unlocked_rows[0][1],
        }

    reachable_locked_rows = _reachable_plan_rows(env, locked_doors)
    reachable_locked_colors = {
        base.grid.get(*position).color for _, position, _ in reachable_locked_rows
    }
    priority_keys = _reachable_plan_rows(
        env,
        (
            position
            for position in _grid_objects(env, object_type="key")
            if base.grid.get(*position).color in reachable_locked_colors
            or any(
                abs(position[0] - door_position[0])
                + abs(position[1] - door_position[1])
                == 1
                and base.grid.get(*door_position).color
                == base.grid.get(*position).color
                for door_position in locked_doors
            )
        ),
    )
    if priority_keys:
        plan = priority_keys[0][2]
        return {
            "action": plan[0] if plan else int(actions.pickup),
            "phase": "acquire_key",
            "subgoal": priority_keys[0][1],
        }

    priority_boxes = _reachable_plan_rows(
        env,
        (
            position
            for position in _grid_objects(env, object_type="box")
            if (box := base.grid.get(*position)).contains is not None
            and box.contains.type == "key"
            and (
                box.contains.color in reachable_locked_colors
                or any(
                    abs(position[0] - door_position[0])
                    + abs(position[1] - door_position[1])
                    == 1
                    and base.grid.get(*door_position).color == box.contains.color
                    for door_position in locked_doors
                )
            )
        ),
    )
    if priority_boxes:
        plan = priority_boxes[0][2]
        return {
            "action": plan[0] if plan else int(actions.toggle),
            "phase": "open_key_box",
            "subgoal": priority_boxes[0][1],
        }

    non_target_balls = [
        position
        for position in _grid_objects(env, object_type="ball")
        if base.grid.get(*position) is not base.obj
    ]
    blocking_balls = [
        position
        for position in non_target_balls
        if any(
            abs(position[0] - door[0]) + abs(position[1] - door[1]) == 1
            for door in locked_doors
        )
    ]
    blocker_rows = _reachable_plan_rows(env, blocking_balls)
    if blocker_rows:
        plan = blocker_rows[0][2]
        return {
            "action": plan[0] if plan else int(actions.pickup),
            "phase": "move_door_blocker",
            "subgoal": blocker_rows[0][1],
        }

    access_blocker = _access_blocker_advice(
        env,
        target_positions=target_positions,
        locked_doors=locked_doors,
        blocking_balls=blocking_balls,
    )
    if access_blocker is not None:
        action, position = access_blocker
        return {
            "action": action,
            "phase": "clear_access_blocker",
            "subgoal": position,
        }

    fallback_boxes = _reachable_plan_rows(env, _grid_objects(env, object_type="box"))
    if fallback_boxes:
        plan = fallback_boxes[0][2]
        return {
            "action": plan[0] if plan else int(actions.toggle),
            "phase": "open_key_box",
            "subgoal": fallback_boxes[0][1],
        }
    raise RuntimeError("No valid expert advice for current ObstructedMaze state")


def make_hierarchical_context_env(environment_id: str) -> Any:
    """Add an objective-blind high-level phase and subgoal marker to MiniGrid.

    The seven phase slots and one subgoal-location channel are generated by a
    deterministic training/task adapter. They never contain a policy-selection
    requirement or preferred-policy label. This is a hybrid hierarchy, not a
    claim of fully end-to-end policy learning.
    """

    import gymnasium as gym
    from gymnasium import spaces

    class HierarchicalContextWrapper(gym.Wrapper):
        def __init__(self, env: Any) -> None:
            super().__init__(env)
            height, width, channels = env.observation_space.shape
            self._base_channels = int(channels)
            self.current_advice: dict[str, Any] | None = None
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(height, width, channels + len(HIERARCHY_PHASES) + 1),
                dtype=np.uint8,
            )

        def _augment(
            self,
            observation: np.ndarray,
            advice: dict[str, Any] | None,
        ) -> np.ndarray:
            output = np.zeros(self.observation_space.shape, dtype=np.uint8)
            output[:, :, : self._base_channels] = observation
            if advice is None:
                return output
            phase_index = HIERARCHY_PHASES.index(str(advice["phase"]))
            output[0, 0, self._base_channels + phase_index] = 1
            subgoal_x, subgoal_y = (int(value) for value in advice["subgoal"])
            if 0 <= subgoal_x < output.shape[0] and 0 <= subgoal_y < output.shape[1]:
                output[
                    subgoal_x,
                    subgoal_y,
                    self._base_channels + len(HIERARCHY_PHASES),
                ] = 1
            return output

        def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            observation, info = self.env.reset(**kwargs)
            self.current_advice = obstructedmaze_expert_advice(self.env)
            return self._augment(observation, self.current_advice), info

        def step(self, action: int):
            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                self.current_advice = None
            else:
                try:
                    self.current_advice = obstructedmaze_expert_advice(self.env)
                except RuntimeError as exc:
                    self.current_advice = None
                    info = {**info, "hierarchy_context_error": f"{type(exc).__name__}: {exc}"}
            return (
                self._augment(observation, self.current_advice),
                reward,
                terminated,
                truncated,
                info,
            )

    return HierarchicalContextWrapper(make_onehot_cnn_env(environment_id))


def collect_obstructedmaze_1dl_expert_episode(
    *,
    seed: int,
    environment_id: str = "MiniGrid-ObstructedMaze-1Dl-v0",
) -> dict[str, Any]:
    """Collect a deterministic training-only expert episode for 1Dl.

    The planner uses simulator state solely to create imitation data. Its
    actions and observations never enter the judge-visible evidence or hidden
    oracle. The implementation intentionally supports only the registered 1Dl
    task with a visible key and unobstructed locked door.
    """

    env = make_onehot_cnn_env(environment_id)
    observations: list[np.ndarray] = []
    action_rows: list[int] = []
    total_return = 0.0
    terminated = truncated = False

    def act_many(actions_to_take: Iterable[int]) -> None:
        nonlocal observation, total_return, terminated, truncated
        for action in actions_to_take:
            if terminated or truncated:
                raise RuntimeError("Expert episode ended before completing its plan")
            observations.append(np.asarray(observation, dtype=np.uint8).copy())
            action_rows.append(int(action))
            observation, reward, terminated, truncated, _ = env.step(int(action))
            total_return += float(reward)

    try:
        observation, _ = env.reset(seed=seed)
        base = env.unwrapped
        keys = _grid_objects(env, object_type="key")
        doors = _grid_objects(env, object_type="door")
        balls = _grid_objects(env, object_type="ball")
        if len(keys) != 1 or len(doors) != 1:
            raise RuntimeError(
                f"Expected one visible key and door, found keys={len(keys)}, doors={len(doors)}"
            )
        target_balls = [
            position
            for position in balls
            if base.grid.get(*position) is base.obj
        ]
        if len(target_balls) != 1:
            raise RuntimeError(f"Expected one target ball, found {len(target_balls)}")

        act_many(plan_to_face(env, keys[0]))
        act_many([int(base.actions.pickup)])
        if base.carrying is None or base.carrying.type != "key":
            raise RuntimeError("Expert failed to pick up the key")

        act_many(plan_to_face(env, doors[0]))
        act_many([int(base.actions.toggle)])
        door = base.grid.get(*doors[0])
        if door is None or door.type != "door" or not door.is_open:
            raise RuntimeError("Expert failed to open the door")

        empty_cells = [
            (x, y)
            for x in range(1, base.grid.width - 1)
            for y in range(1, base.grid.height - 1)
            if base.grid.get(x, y) is None
            and (x, y) != tuple(int(value) for value in base.agent_pos)
        ]
        drop_plan, _ = plan_to_face_any(env, empty_cells)
        act_many(drop_plan)
        act_many([int(base.actions.drop)])
        if base.carrying is not None:
            raise RuntimeError("Expert failed to drop the used key")

        act_many(plan_to_face(env, target_balls[0]))
        act_many([int(base.actions.pickup)])
        success = bool(terminated and total_return > 0.0)
        if not success:
            raise RuntimeError("Expert failed to complete the official task")
        actions_list = [int(action) for action in action_rows]
        return {
            "seed": seed,
            "environment_id": environment_id,
            "success": success,
            "return": total_return,
            "steps": len(actions_list),
            "observations": observations,
            "actions": actions_list,
            "action_sequence_hash": stable_hash(actions_list, 20),
        }
    finally:
        env.close()


def collect_obstructedmaze_expert_episode(
    *,
    seed: int,
    environment_id: str,
) -> dict[str, Any]:
    """Collect a state-planning expert episode for official ObstructedMaze tasks.

    The solver opens reachable passage doors, moves door-adjacent non-target
    balls, opens key boxes, and unlocks reachable doors until the target ball
    becomes reachable. It is deliberately exhaustive instead of using a hidden
    semantic preference; this makes it suitable only as policy-training data.
    """

    env = make_onehot_cnn_env(environment_id)
    observations: list[np.ndarray] = []
    action_rows: list[int] = []
    total_return = 0.0
    terminated = truncated = False

    def act_many(actions_to_take: Iterable[int]) -> None:
        nonlocal observation, total_return, terminated, truncated
        for action in actions_to_take:
            if terminated or truncated:
                raise RuntimeError("Expert episode ended before completing its plan")
            observations.append(np.asarray(observation, dtype=np.uint8).copy())
            action_rows.append(int(action))
            observation, reward, terminated, truncated, _ = env.step(int(action))
            total_return += float(reward)

    def reachable_rows(positions: Iterable[tuple[int, int]]) -> list[tuple[int, tuple[int, int], list[int]]]:
        rows: list[tuple[int, tuple[int, int], list[int]]] = []
        for position in positions:
            try:
                plan = plan_to_face(env, position)
            except RuntimeError:
                continue
            rows.append((len(plan), position, plan))
        return sorted(rows, key=lambda row: (row[0], row[1]))

    def drop_carried() -> None:
        base = env.unwrapped
        if base.carrying is None:
            return
        door_positions = set(_grid_objects(env, object_type="door"))
        forbidden = set(door_positions)
        for door_x, door_y in door_positions:
            forbidden.update(
                {
                    (door_x - 1, door_y),
                    (door_x + 1, door_y),
                    (door_x, door_y - 1),
                    (door_x, door_y + 1),
                }
            )
        empty_cells = [
            (x, y)
            for x in range(1, base.grid.width - 1)
            for y in range(1, base.grid.height - 1)
            if base.grid.get(x, y) is None
            and (x, y) not in forbidden
            and (x, y) != tuple(int(value) for value in base.agent_pos)
        ]
        rows = reachable_rows(empty_cells)
        if not rows:
            raise RuntimeError("Expert could not find a safe position to drop a carried object")
        def door_clearance(position: tuple[int, int]) -> int:
            return min(
                abs(position[0] - door[0]) + abs(position[1] - door[1])
                for door in door_positions
            )

        _, _, selected_plan = min(
            rows,
            key=lambda row: (-door_clearance(row[1]), row[0], row[1]),
        )
        act_many(selected_plan)
        act_many([int(base.actions.drop)])
        if base.carrying is not None:
            raise RuntimeError("Expert failed to drop carried object")

    try:
        observation, _ = env.reset(seed=seed)
        base = env.unwrapped
        target_positions = [
            (x, y)
            for x in range(base.grid.width)
            for y in range(base.grid.height)
            if base.grid.get(x, y) is base.obj
        ]
        if len(target_positions) != 1:
            raise RuntimeError(f"Expected one target object, found {len(target_positions)}")
        target = target_positions[0]

        operation_count = 0
        operation_trace: list[str] = []
        while not (terminated or truncated):
            operation_count += 1
            if operation_count > 128:
                raise RuntimeError(
                    "Expert exceeded the bounded operation count; recent operations="
                    + repr(operation_trace[-20:])
                )

            target_rows = reachable_rows([target])
            if target_rows and base.carrying is None:
                operation_trace.append("pickup_target")
                act_many(target_rows[0][2])
                act_many([int(base.actions.pickup)])
                continue

            door_positions = _grid_objects(env, object_type="door")
            locked_doors = [
                position
                for position in door_positions
                if (door := base.grid.get(*position)) is not None
                and door.type == "door"
                and door.is_locked
            ]

            if base.carrying is not None:
                matching_rows: list[tuple[int, tuple[int, int], list[int]]] = []
                if base.carrying.type == "key":
                    matching_rows = reachable_rows(
                        position
                        for position in locked_doors
                        if base.grid.get(*position).color == base.carrying.color
                    )
                if matching_rows:
                    operation_trace.append(f"unlock:{matching_rows[0][1]}")
                    act_many(matching_rows[0][2])
                    act_many([int(base.actions.toggle)])
                    continue
                operation_trace.append(f"drop:{base.carrying.type}:{base.carrying.color}")
                drop_carried()
                continue

            unlocked_closed = [
                position
                for position in door_positions
                if (door := base.grid.get(*position)) is not None
                and door.type == "door"
                and not door.is_locked
                and not door.is_open
            ]
            unlocked_rows = reachable_rows(unlocked_closed)
            if unlocked_rows:
                operation_trace.append(f"open_unlocked:{unlocked_rows[0][1]}")
                act_many(unlocked_rows[0][2])
                act_many([int(base.actions.toggle)])
                continue

            reachable_locked_rows = reachable_rows(locked_doors)
            reachable_locked_colors = {
                base.grid.get(*position).color
                for _, position, _ in reachable_locked_rows
            }
            priority_key_rows = reachable_rows(
                position
                for position in _grid_objects(env, object_type="key")
                if base.grid.get(*position).color in reachable_locked_colors
                or any(
                    abs(position[0] - door_position[0])
                    + abs(position[1] - door_position[1])
                    == 1
                    and base.grid.get(*door_position).color
                    == base.grid.get(*position).color
                    for door_position in locked_doors
                )
            )
            if priority_key_rows:
                operation_trace.append(f"pickup_key:{priority_key_rows[0][1]}")
                act_many(priority_key_rows[0][2])
                act_many([int(base.actions.pickup)])
                continue

            priority_box_rows = reachable_rows(
                position
                for position in _grid_objects(env, object_type="box")
                if (box := base.grid.get(*position)).contains is not None
                and box.contains.type == "key"
                and (
                    box.contains.color in reachable_locked_colors
                    or any(
                        abs(position[0] - door_position[0])
                        + abs(position[1] - door_position[1])
                        == 1
                        and base.grid.get(*door_position).color == box.contains.color
                        for door_position in locked_doors
                    )
                )
            )
            if priority_box_rows:
                operation_trace.append(f"open_box:{priority_box_rows[0][1]}")
                act_many(priority_box_rows[0][2])
                act_many([int(base.actions.toggle)])
                continue

            non_target_balls = [
                position
                for position in _grid_objects(env, object_type="ball")
                if base.grid.get(*position) is not base.obj
            ]
            blocking_balls = [
                position
                for position in non_target_balls
                if any(
                    abs(position[0] - door[0]) + abs(position[1] - door[1]) == 1
                    for door in locked_doors
                )
            ]
            blocker_rows = reachable_rows(blocking_balls)
            if blocker_rows:
                operation_trace.append(f"move_blocker:{blocker_rows[0][1]}")
                act_many(blocker_rows[0][2])
                act_many([int(base.actions.pickup)])
                drop_carried()
                continue

            box_rows = reachable_rows(_grid_objects(env, object_type="box"))
            if box_rows:
                operation_trace.append(f"open_box:{box_rows[0][1]}")
                act_many(box_rows[0][2])
                act_many([int(base.actions.toggle)])
                continue

            locked_colors = {
                base.grid.get(*position).color
                for _, position, _ in reachable_locked_rows
            }
            key_rows = reachable_rows(
                position
                for position in _grid_objects(env, object_type="key")
                if base.grid.get(*position).color in locked_colors
                or any(
                    abs(position[0] - door_position[0])
                    + abs(position[1] - door_position[1])
                    == 1
                    and base.grid.get(*door_position).color
                    == base.grid.get(*position).color
                    for door_position in locked_doors
                )
            )
            if key_rows:
                operation_trace.append(f"pickup_key:{key_rows[0][1]}")
                act_many(key_rows[0][2])
                act_many([int(base.actions.pickup)])
                continue

            raise RuntimeError(
                "Expert has no reachable target, door, blocker, box, or matching key; "
                f"recent_operations={operation_trace[-20:]}; "
                f"locked_doors={locked_doors}; blockers={blocking_balls}; "
                f"keys={_grid_objects(env, object_type='key')}; "
                f"boxes={_grid_objects(env, object_type='box')}; "
                f"agent={tuple(int(value) for value in base.agent_pos)}; "
                "door_neighbors="
                + repr(
                    {
                        position: [
                            (
                                neighbor,
                                None
                                if base.grid.get(*neighbor) is None
                                else base.grid.get(*neighbor).type,
                            )
                            for neighbor in (
                                (position[0] - 1, position[1]),
                                (position[0] + 1, position[1]),
                                (position[0], position[1] - 1),
                                (position[0], position[1] + 1),
                            )
                        ]
                        for position in locked_doors
                    }
                )
            )

        success = bool(terminated and total_return > 0.0)
        if not success:
            raise RuntimeError("Expert failed to complete the official task")
        actions_list = [int(action) for action in action_rows]
        return {
            "seed": seed,
            "environment_id": environment_id,
            "success": success,
            "return": total_return,
            "steps": len(actions_list),
            "observations": observations,
            "actions": actions_list,
            "action_sequence_hash": stable_hash(actions_list, 20),
        }
    finally:
        env.close()
