from __future__ import annotations

import os
import pickle
from pathlib import Path

import jq
import numpy as np
from torch.utils.data.dataset import Dataset as TorchdataSet
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..dataloader.large_file_lines_reader import LargeFileLinesReader


class Dataset(TorchdataSet):
    def __init__(self, raw_data_path: Path, block_size: int):
        self.raw_data_path = raw_data_path
        self.block_size = block_size


class MemMapDataset(Dataset):
    def __init__(self, raw_data_path: Path, block_size: int, tokenizer: PreTrainedTokenizer, jq_pattern: str = ".text"):
        """
        :param raw_data_path: Path to a jsonl file, which holds text data
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        :param tokenizer: TODO
        :param jq_pattern: jq-pattern applied on every jsonl-entry. Results are afterwards tokenized and packed
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size)

        self.reader = LargeFileLinesReader(self.raw_data_path)
        self.jq_filter = jq.compile(jq_pattern)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> str:
        obj = self.tokenizer(
            self.jq_filter.input_text(self.reader[idx]).first(),
            max_length=self.block_size,
            padding="max_length",
            truncation=True,
        )
        return obj


class PackedMemMapDatasetBase(Dataset):
    INT_SIZE_IN_BYTES = 4
    HEADER_SIZE_IN_BYTES = 8

    def __init__(self, raw_data_path: Path, block_size: int):
        """
        :param raw_data_path: Path to a packed binary file (*.pbin).
                              Use `llm_gym create_packed_data` to create one based on a jsonl-file.
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size)
        if not self.raw_data_path.is_file():
            raise FileNotFoundError(
                f"Packed Data was not found at {self.raw_data_path}."
                f"Create on in advance by using `llm_gym create_packed_data`."
            )

        # get number of total bytes in file
        with self.raw_data_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            self.total_bytes = f.tell()
            f.seek(0)

        # get number of bytes in data section
        self.data_len = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=0,
            shape=(self.HEADER_SIZE_IN_BYTES,),
        ).view(f"S{self.HEADER_SIZE_IN_BYTES}")
        self.data_len = int.from_bytes(self.data_len, byteorder="big")

        # get index
        self.index_base = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=self.HEADER_SIZE_IN_BYTES + self.data_len,
            shape=(self.total_bytes - self.data_len - self.HEADER_SIZE_IN_BYTES,),
        ).view(f"S{self.total_bytes-self.data_len-self.HEADER_SIZE_IN_BYTES}")
        self.index_base = pickle.loads(self.index_base)


class PackedMemMapDatasetContinuous(PackedMemMapDatasetBase):
    def __init__(self, raw_data_path: Path, block_size: int):
        """
        :param raw_data_path: Path to a packed binary file (*.pbin).
                              Use `llm_gym create_packed_data` to create one based on a jsonl-file.
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size)

        # get number of total tokens in file
        total_tokens = self.data_len // self.INT_SIZE_IN_BYTES
        self.num_samples = total_tokens // self.block_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        tokens_as_byte_strings = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=self.HEADER_SIZE_IN_BYTES + idx * self.INT_SIZE_IN_BYTES * self.block_size,
            shape=(self.INT_SIZE_IN_BYTES * self.block_size,),
        ).view(f"S{self.INT_SIZE_IN_BYTES}")
        tokens = [int.from_bytes(token, byteorder="big") for token in tokens_as_byte_strings]
        attention_mask = [1] * len(tokens)
        return {"input_ids": tokens, "attention_mask": attention_mask}


class PackedMemMapDatasetMegatron(PackedMemMapDatasetBase):
    def generate_megatron_index(self):
        # index_output_path = Path(index_output_path)

        self.index = []
        curr_offset = self.HEADER_SIZE_IN_BYTES
        curr_len = 0
        block_size_in_bytes = self.block_size * self.INT_SIZE_IN_BYTES
        for segment_offset, segment_len in tqdm(self.index_base):
            if curr_len + segment_len < block_size_in_bytes:
                curr_len += segment_len
            elif curr_len + segment_len == block_size_in_bytes:
                self.index.append((curr_offset, block_size_in_bytes))
                curr_len = 0
                curr_offset += block_size_in_bytes
            else:
                self.index.append((curr_offset, block_size_in_bytes))
                if segment_len > block_size_in_bytes:
                    curr_offset += block_size_in_bytes
                    curr_len = 0
                else:
                    curr_offset = segment_offset
                    curr_len = segment_len

    def __init__(self, raw_data_path: Path, block_size: int):
        """
        :param raw_data_path: Path to a packed binary file (*.pbin).
                              Use `llm_gym create_packed_data` to create one based on a jsonl-file.
        :param block_size: alias for max sequence length. The amount of tokens the model can handle.
        """
        super().__init__(raw_data_path=raw_data_path, block_size=block_size)

        # get number of total tokens in file
        total_tokens = self.data_len // self.INT_SIZE_IN_BYTES
        self.num_samples = total_tokens // self.block_size

        self.generate_megatron_index()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        offset, length = self.index[idx]
        tokens_as_byte_strings = np.memmap(
            self.raw_data_path,
            mode="r",
            offset=offset,
            shape=(length,),
        ).view(f"S{self.INT_SIZE_IN_BYTES}")
        tokens = [int.from_bytes(token, byteorder="big") for token in tokens_as_byte_strings]
        attention_mask = [1] * len(tokens)
        return {"input_ids": tokens, "attention_mask": attention_mask}
