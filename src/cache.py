"""Checksum-verified on-disk cache for derived arrays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


@dataclass(frozen=True)
class CacheProvenance:
    identifier: str
    transform: dict[str, object]
    source_checksum: str

    def __post_init__(self) -> None:
        if not self.identifier or not self.source_checksum:
            raise ValueError("identifier and source_checksum must be non-empty")


def _canonical(provenance: CacheProvenance) -> str:
    return json.dumps(asdict(provenance), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(array: np.ndarray, provenance_json: str) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    digest.update(provenance_json.encode("utf-8"))
    return digest.hexdigest()


class CacheStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not key or Path(key).name != key or key in {".", ".."}:
            raise ValueError("cache key must be a non-empty filename component")
        return self.root / f"{key}.npz"

    def write(self, key: str, value: np.ndarray, provenance: CacheProvenance) -> None:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("cached value must be a finite numeric array")
        provenance_json = _canonical(provenance)
        checksum = _digest(array, provenance_json)
        target = self._path(key)
        fd, temporary = tempfile.mkstemp(prefix=f".{key}.", suffix=".npz", dir=self.root)
        os.close(fd)
        try:
            np.savez_compressed(temporary, value=array, provenance=np.array(provenance_json), checksum=np.array(checksum))
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def read(self, key: str, provenance: CacheProvenance) -> np.ndarray:
        path = self._path(key)
        with np.load(path, allow_pickle=False) as stored:
            value = stored["value"]
            actual_json = str(stored["provenance"].item())
            checksum = str(stored["checksum"].item())
        if actual_json != _canonical(provenance):
            raise ValueError("cache provenance does not match the requested provenance")
        if checksum != _digest(value, actual_json):
            raise ValueError("cache checksum mismatch")
        return value
