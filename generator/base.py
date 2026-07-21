from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DatasetItem:
    prompt: np.ndarray
    answer: np.ndarray


class Generator(ABC):
    seed: int = 42

    @abstractmethod
    def _sample_one(self) -> DatasetItem: ...

    def sample(self, n: int = 1) -> List[DatasetItem]:
        return [self._sample_one() for _ in range(n)]

    @staticmethod
    def _hash(item: DatasetItem) -> bytes:
        return item.prompt.tobytes()

    def generate_val(self, n: int) -> List[DatasetItem]:
        val, seen = [], set()
        while len(val) < n:
            item = self._sample_one()
            h = self._hash(item)
            if h in seen:
                continue
            seen.add(h)
            val.append(item)
        self.val_hashes = seen
        return val

    def sample_train(self, n: int = 1) -> List[DatasetItem]:
        if not hasattr(self, "val_hashes"):
            raise RuntimeError("Call `generate_val(...)' before sampling train")
        out = []
        while len(out) < n:
            item = self._sample_one()
            if self._hash(item) not in self.val_hashes:
                out.append(item)
        return out


# gen = KVRetrievalGenerator(64, 64, 64)
# iters = 0
# gen_val_start = time.perf_counter()
# val = gen.generate_val(1000)
# gen_val_end = time.perf_counter()
# print(f"Val generated in {gen_val_end - gen_val_start:.2f} s")
# start_time = time.perf_counter()
# while time.perf_counter() - start_time < 3.0:
#     gen.sample_train(128)
#     iters += 1
# print(f"Throughput: {iters / 3.0:.2f} train batches |b|=128 per second")
# print(blake2b(np.array([1, 2, 3, 4, 5]).tobytes()).hexdigest())
# print(int.from_bytes(blake2b(np.array([1, 2, 3, 4, 5]).tobytes()).digest(), "big"))
