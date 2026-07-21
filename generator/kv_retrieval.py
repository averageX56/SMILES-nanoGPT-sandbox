from typing import Tuple

import numpy as np

from .base import DatasetItem, Generator


class KVRetrievalGenerator(Generator):
    PAD_ID = 0
    QUERY_MARKER_ID = 1
    N_SPECIAL = 2

    def __init__(
        self,
        k_card: int,
        v_card: int,
        n_pairs: int | Tuple[int, int],
        duplicate_keys: bool = False,
        seed: int | None = 42,
    ):
        """
        Generator for KV retrieval task.

        :param k_card: Amount of possible keys.
        :type k_card: int
        :param v_card: Amount of possible values.
        :type v_card: int
        :param n_pairs: Amount of pairs in each prompt. `n_pairs' if the parameter is an
        integer, and a value in [n_pairs[0], n_pairs[1]) if it is a tuple.
        :type n_pairs: int | Tuple[int, int]
        :param duplicate_keys: Whether to allow duplicate keys in the prompt. The value
        for the last occurence is considered an answer in this case.
        :type duplicate_keys: bool
        :param seed: Randomization seed, None for non-reproducible environment.
        :type seed: int | None
        """
        self.k_card = k_card
        self.v_card = v_card
        self.n_pairs = n_pairs

        self.k_token_ids = np.arange(k_card) + self.N_SPECIAL
        self.v_token_ids = np.arange(v_card) + self.N_SPECIAL + k_card

        self.rng = np.random.default_rng(seed)

        self.duplicate_keys = duplicate_keys

    def _sample_one(self) -> DatasetItem:
        if isinstance(self.n_pairs, int):
            n_pairs = self.n_pairs
        else:
            n_pairs = int(self.rng.integers(self.n_pairs[0], self.n_pairs[1]))
        keys = self.rng.choice(
            self.k_token_ids, size=n_pairs, replace=self.duplicate_keys
        )
        values = self.rng.choice(self.v_token_ids, size=n_pairs, replace=True)
        kv_end = 2 * self.n_pairs

        seq = np.empty(
            2 * n_pairs + 2, dtype=np.int64
        )  # n pairs + query separator + query
        seq[0:kv_end:2] = keys
        seq[1:kv_end:2] = values
        seq[-2] = self.QUERY_MARKER_ID

        qi = int(self.rng.integers(n_pairs))
        seq[-1] = keys[qi]
        answer = np.array([values[qi]], dtype=np.int64)

        return DatasetItem(prompt=seq, answer=answer)
