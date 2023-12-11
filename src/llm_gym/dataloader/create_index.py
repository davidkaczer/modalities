import json
import os
import pickle as pkl
import queue
import threading
from pathlib import Path

import numpy as np
from tqdm import tqdm


# TODO: benchmark against pyspark
class IndexGenerator:
    def __init__(self, src_file: Path | str, chunksize: int = 4096, drop_faulty_entries: bool = False):
        """
        :param src_file: Path to a jsonl-file.
        :param chunksize: defines the size of byte chunks that are processed via a producer-consumer approach.
                          The producer reads chunks from the `src_file`, while the consumer creates index entries.
        :param drop_faulty_entries: Allow broken json entries in `src_file` by just skipping them.
                                    Otherwise, the index generation fails with an exception.
        """
        self.src_file = Path(src_file)
        self.chunksize = chunksize
        self.drop_faulty_entries = drop_faulty_entries
        with self.src_file.open(mode="r", encoding="utf-8") as fin:
            fin.seek(0, os.SEEK_END)
            char_num = fin.tell()
        self.chunks = char_num // self.chunksize
        self.reminder = char_num % self.chunksize
        self.chunk_queue = queue.Queue()
        self.index_map = []
        self.exception_buffer = []

    def run(self, dst_file: Path):
        self.exception_buffer = []
        reader = threading.Thread(target=self._reader_thread)
        reader.start()
        processor = threading.Thread(target=self._indexer_thread)
        processor.start()
        reader.join()
        processor.join()
        if self.exception_buffer:
            raise self.exception_buffer[0]
        print(f"Created index of length {len(self.index_map)}")
        dst_file.write_bytes(pkl.dumps(self.index_map))

    def _indexer_thread(self):
        def queue_generator():
            while True:
                chunk = self.chunk_queue.get()
                if chunk is None:
                    break
                yield chunk

        def process_line(last_index: int, curr_index: int):
            segment_len = curr_index - last_index
            try:  # check if line is a valid json
                string = np.memmap(self.src_file, mode="r", offset=last_index, shape=(segment_len,)).view("S1").tolist()
                string = [c.decode("iso-8859-1") for c in string]
                string = "".join(string)
                json.loads(string)
                self.index_map.append((last_index, segment_len))
            except Exception as low_level_err:
                if self.drop_faulty_entries:
                    print(f"faulty line at {last_index}-{curr_index}, skipping...")
                else:
                    print(f"{string=}")
                    err = ValueError(f"faulty line at {last_index}-{curr_index}")
                    err.__cause__ = low_level_err
                    self.exception_buffer.append(err)

        self.index_map = []
        last_index = 0
        for chunk_idx, chunk in tqdm(enumerate(queue_generator()), desc="Processed Chunks", total=self.chunks):
            for char_index, c in enumerate(chunk):
                curr_index = chunk_idx * self.chunksize + char_index
                if c == ord("\n"):
                    process_line(last_index, curr_index)
                    last_index = curr_index + 1
        # prevents automatically added "\n"-chars at the end of files getting interpreted as own sample
        if curr_index >= last_index:
            process_line(last_index, curr_index + 1)

    def _reader_thread(self):
        with open(self.src_file, "rb") as fin:
            while True:
                chunk = fin.read(self.chunksize)
                if self.exception_buffer:
                    raise RuntimeError(
                        "Exception found in exception buffer. Probably the indexer thread ran into an error..."
                    )
                if not chunk:
                    break
                self.chunk_queue.put(chunk)
        self.chunk_queue.put(None)
