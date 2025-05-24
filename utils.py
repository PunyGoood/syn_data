from typing import Any, Iterable, Literal, Mapping, Sequence, TypeVar
import os   
import time
import json
from pathlib import Path

_T = TypeVar("_T")

def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))

N_CORES = 1 if (count := os.cpu_count()) is None or count == 0 else count // 2

def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def read_jsonl(path: str | Path) -> list[Any]:
    """Read lines of JSON from a file (including '\n')."""
    with Path(path).open("r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str | Path, data: Sequence[Mapping], mode: str = "w"):
    # cannot use `dict` here as it is invariant
    with Path(path).open(mode) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")