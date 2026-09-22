import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.values = np.zeros(2 * capacity, dtype=np.float64)
        self.position = 0
        self.size = 0

    @property
    def total(self) -> float:
        return float(self.values[1])

    def add(self, priority: float) -> int:
        tree_index = self.position + self.capacity
        self.update(tree_index, priority)
        data_index = self.position
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return data_index

    def update(self, tree_index: int, priority: float) -> None:
        change = priority - self.values[tree_index]
        self.values[tree_index] = priority
        parent = tree_index >> 1
        while parent >= 1:
            self.values[parent] += change
            parent >>= 1

    def retrieve(self, value: float) -> tuple[int, float, int]:
        tree_index = 1
        while tree_index < self.capacity:
            left = tree_index << 1
            right = left | 1
            if value <= self.values[left]:
                tree_index = left
            else:
                value -= self.values[left]
                tree_index = right
        return (
            tree_index,
            float(self.values[tree_index]),
            tree_index - self.capacity,
        )


class PrioritizedReplay:
    def __init__(
        self,
        capacity: int,
        state_dimension: int,
        action_count: int,
        alpha: float,
        priority_epsilon: float,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.tree = SumTree(capacity)
        self.maximum_priority = 1.0
        self.states = np.zeros((capacity, state_dimension), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dimension), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.ones((capacity, action_count), dtype=bool)

    def __len__(self) -> int:
        return self.tree.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        index = self.tree.add(self.maximum_priority**self.alpha)
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = float(done)
        self.next_masks[index] = next_mask

    def sample(
        self,
        batch_size: int,
        beta: float,
        generator: np.random.Generator,
    ) -> tuple[np.ndarray, ...]:
        if len(self) < batch_size or self.tree.total <= 0.0:
            raise RuntimeError("replay buffer is not ready")
        segment = self.tree.total / batch_size
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        data_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        for sample_index in range(batch_size):
            value = generator.uniform(
                segment * sample_index,
                segment * (sample_index + 1),
            )
            tree_index, priority, data_index = self.tree.retrieve(value)
            tree_indices[sample_index] = tree_index
            data_indices[sample_index] = data_index
            priorities[sample_index] = priority
        probabilities = priorities / self.tree.total
        weights = (len(self) * probabilities) ** (-beta)
        weights /= weights.max()
        return (
            self.states[data_indices],
            self.actions[data_indices],
            self.rewards[data_indices],
            self.next_states[data_indices],
            self.dones[data_indices],
            weights.astype(np.float32),
            tree_indices,
            self.next_masks[data_indices],
        )

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        temporal_difference_errors: np.ndarray,
    ) -> None:
        for tree_index, error in zip(tree_indices, temporal_difference_errors):
            priority = float(abs(error)) + self.priority_epsilon
            self.maximum_priority = max(self.maximum_priority, priority)
            self.tree.update(int(tree_index), priority**self.alpha)

