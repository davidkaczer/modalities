from pathlib import Path
from typing import Union

import jq
import pytest
import tiktoken
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from llmgym.tests.dataset_loader.large_file_lines_reader import LargeFileLinesReader

# TODO: discuss with team mmap based on plain text
#  (less memory consumed, more time required for loading during training)
#  vs.
#  mmap based on encoded, tokenized texts as bytes (using more memory but faster loading speed.)
#  In theory an experiment could help to see,
#  if the additional performance for loading during training is actually used.
#  Might be that the bottleneck is actually not the DataLoader but instead the forward/backward passes.


dummy_path = Path("/home/shared/openwebtext/head20000_openwebtext2_en.jsonl")
enc = tiktoken.get_encoding("gpt2")
MAX_SEQ_LEN = 2048


class LocalDataset(Dataset):
    def __init__(self, raw_data_path: Union[str, Path], index_path: Union[str, Path], jq_filter: str = ".text"):
        self.raw_data_path = Path(raw_data_path)
        self.index_path = Path(index_path)
        self.reader = LargeFileLinesReader(self.raw_data_path, self.index_path, lazy_init=True)
        self.jq_filter = jq.compile(jq_filter)

    def __len__(self) -> int:
        return len(self.content)

    def __getitem__(self, idx: int) -> str:
        return self._tokenize(self.jq_filter.input_text(self.reader[idx]).first())

    def _tokenize(self, line: str) -> torch.Tensor:
        ids = enc.encode_ordinary(line)
        ids.append(enc.eot_token)
        ids_tensor = torch.tensor(ids)[:MAX_SEQ_LEN]
        orig_length = ids_tensor.shape[0]
        padded_ids = torch.nn.functional.pad(ids_tensor, (0, MAX_SEQ_LEN - orig_length))
        return padded_ids


def measure_loading_and_iterating(path: Path):
    from time import time

    start_time = time()
    ds = LocalDataset(path)
    ckpt_time_1 = time()
    loading_time = ckpt_time_1 - start_time
    dl = DataLoader(ds, batch_size=5)
    for _ in dl:
        pass
    ckpt_time_2 = time()
    iteration_time = ckpt_time_2 - start_time
    amount_of_tokens = 0
    for batch in dl:
        amount_of_tokens += torch.count_nonzero(batch)
    print(f"Final Batch: {batch}")
    print(f"Final Batch #tokens: {torch.count_nonzero(batch)}")
    return loading_time, iteration_time, amount_of_tokens


@pytest.mark.slow
def test_measure_bytes_per_sec():
    loading_time, iteration_time, _ = measure_loading_and_iterating(dummy_path)
    print(f"Loading Time: {loading_time} sec")
    print(f"Iteration Time: {iteration_time} sec")
    assert False


@pytest.mark.slow
def test_measure_tokens_per_sec():
    loading_time, iteration_time, amount_of_tokens = measure_loading_and_iterating(dummy_path)
    average_tokens_per_second = amount_of_tokens / (loading_time + iteration_time)
    print(f"ø-Tokens per Second: {average_tokens_per_second}/s")
    assert False
