import logging
import os
import glob
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class KIDNEY(ExtendedVisionDataset):
    """Kidney histopathology dataset stored as HDF5 shards.

    Supports two discovery modes:
      - Directory mode: ``root`` points to a folder containing ``shard_*.h5`` files.
      - File-list mode: ``root`` points to a ``.txt`` file with one HDF5 path per line.

    Each HDF5 shard is expected to contain:
      - ``patches``: uint8 array of shape (N, H, W, C) **or** variable-length
        byte-strings (JPEG/PNG encoded).  When raw uint8 arrays are stored the
        bytes returned by ``get_image_data`` are the raw pixel buffer; the
        ``ImageDataDecoder`` will handle both cases transparently.
      - ``labels`` (optional): integer array of shape (N,).

    HDF5 file handles are opened lazily per DataLoader worker to avoid the
    well-known ``h5py`` fork-safety issue.  Handles are closed when the dataset
    is garbage-collected.

    Args:
        root: Path to the shard directory **or** a ``.txt`` manifest file.
        extra: Unused, kept for API compatibility with other DINOv3 datasets.
        split: Optional split enum (TRAIN / VAL / TEST).  When ``root`` is a
            directory and ``split`` is given the class looks for shards inside
            ``<root>/<split.value>/``.
        transforms, transform, target_transform: Standard torchvision callables.
        shard_glob: Glob pattern used to discover shard files (default ``shard_*.h5``).
        max_retries: Number of retries on I/O errors before raising.
        shuffle_shards: Whether to shuffle the global index at construction.
            Useful for self-supervised pretraining; disable for deterministic
            evaluation.
        epoch_length: If > 0, overrides ``__len__`` so that samplers see a
            fixed virtual epoch size (useful for very large / variable-size
            datasets).  Set to 0 or -1 to use the true dataset size.
        prefetch_index: If True, the full index mapping is built at
            construction time.  Set to False to defer (useful when the dataset
            is instantiated on the main process but consumed by workers).
    """

    Target = int
    Split = _Split

    def __init__(
        self,
        *,
        root: str,
        extra: str = "",
        split: Optional["KIDNEY.Split"] = None,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shard_glob: str = "shard_*.h5",
        max_retries: int = 3,
        shuffle_shards: bool = True,
        epoch_length: int = 0,
        prefetch_index: bool = True,
    ) -> None:
        if h5py is None:
            raise ImportError(
                "h5py is required for the KIDNEY dataset. "
                "Install it with: pip install h5py"
            )

        # Resolve the actual shard directory
        shard_root = root
        self._is_filelist = root.endswith(".txt")

        if not self._is_filelist and split is not None:
            candidate = os.path.join(root, split.value)
            if os.path.isdir(candidate):
                shard_root = candidate

        super().__init__(
            root=shard_root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoders,
        )

        self._split = split
        self._shard_glob = shard_glob
        self._max_retries = max_retries
        self._shuffle_shards = shuffle_shards
        self._epoch_length = epoch_length if epoch_length > 0 else 0

        # Discover shard files
        self._hdf5_files: List[str] = self._discover_shards(shard_root)
        if len(self._hdf5_files) == 0:
            raise FileNotFoundError(
                f"No HDF5 shard files found. root={root}, "
                f"shard_root={shard_root}, glob={shard_glob}"
            )
        logger.info(f"KIDNEY: found {len(self._hdf5_files)} shard files under {shard_root}")

        # Global index mapping: list of (file_index, internal_index)
        self._index_mapping: List[Tuple[int, int]] = []
        self._shard_sizes: List[int] = []

        # Per-worker file handles (lazily initialised)
        self._file_handles: Optional[dict] = None

        if prefetch_index:
            self._build_index_mapping()


    def _discover_shards(self, shard_root: str) -> List[str]:
        """Return a sorted list of HDF5 file paths."""
        if self._is_filelist:
            # File-list mode (legacy DINOv2 approach)
            with open(shard_root, "r") as fh:
                lines = [l.strip() for l in fh if l.strip()]
            # Validate paths exist
            valid = [p for p in lines if os.path.isfile(p)]
            if len(valid) < len(lines):
                n_missing = len(lines) - len(valid)
                warnings.warn(
                    f"KIDNEY: {n_missing}/{len(lines)} shard paths in "
                    f"{shard_root} do not exist and will be skipped."
                )
            return sorted(valid)
        else:
            pattern = os.path.join(shard_root, self._shard_glob)
            return sorted(glob.glob(pattern))


    def _build_index_mapping(self) -> None:
        """Build a flat (file_idx, internal_idx) mapping over all shards."""
        self._index_mapping = []
        self._shard_sizes = []
        total = 0

        for file_idx, fpath in enumerate(self._hdf5_files):
            try:
                with h5py.File(fpath, "r") as h5f:
                    n = len(h5f["patches"])
            except Exception as exc:
                logger.warning(
                    f"KIDNEY: could not read shard {fpath}, skipping. Error: {exc}"
                )
                self._shard_sizes.append(0)
                continue

            self._shard_sizes.append(n)
            self._index_mapping.extend(
                [(file_idx, i) for i in range(n)]
            )
            total += n

        logger.info(
            f"KIDNEY: indexed {total:,d} patches across "
            f"{len(self._hdf5_files)} shards"
        )

        if self._shuffle_shards:
            rng = np.random.default_rng(seed=42)
            rng.shuffle(self._index_mapping)


    def _ensure_file_handles(self) -> None:
        """Open HDF5 files lazily (safe for forked DataLoader workers)."""
        if self._file_handles is not None:
            return
        self._file_handles = {}
        for idx, fpath in enumerate(self._hdf5_files):
            try:
                self._file_handles[idx] = h5py.File(fpath, "r", swmr=True)
            except Exception as exc:
                logger.warning(f"KIDNEY: failed to open {fpath}: {exc}")

    def _get_handle(self, file_idx: int) -> "h5py.File":
        self._ensure_file_handles()
        if file_idx not in self._file_handles:
            # Try to open on-the-fly
            fpath = self._hdf5_files[file_idx]
            self._file_handles[file_idx] = h5py.File(fpath, "r", swmr=True)
        return self._file_handles[file_idx]


    def get_image_data(self, index: int) -> bytes:
        """Return raw image bytes for the given global index."""
        if not self._index_mapping:
            self._build_index_mapping()

        file_idx, internal_idx = self._index_mapping[index]

        for attempt in range(self._max_retries):
            try:
                h5f = self._get_handle(file_idx)
                data = h5f["patches"][internal_idx]
                # data may be a numpy array (uint8 image) or bytes
                if isinstance(data, np.ndarray):
                    return data.tobytes()
                return bytes(data)
            except Exception as exc:
                logger.warning(
                    f"KIDNEY: read error on shard {self._hdf5_files[file_idx]} "
                    f"idx={internal_idx}, attempt {attempt + 1}/{self._max_retries}: {exc}"
                )
                # Invalidate handle and retry
                if file_idx in (self._file_handles or {}):
                    try:
                        self._file_handles[file_idx].close()
                    except Exception:
                        pass
                    del self._file_handles[file_idx]

        # All retries exhausted – fall back to a random other sample
        logger.error(
            f"KIDNEY: giving up on shard {self._hdf5_files[file_idx]} "
            f"idx={internal_idx} after {self._max_retries} retries. "
            f"Returning a random replacement sample."
        )
        fallback_idx = np.random.randint(0, len(self._index_mapping))
        return self.get_image_data(fallback_idx)

    def get_target(self, index: int) -> Any:
        """Return the label for the given global index, or None."""
        if not self._index_mapping:
            self._build_index_mapping()

        file_idx, internal_idx = self._index_mapping[index]

        try:
            h5f = self._get_handle(file_idx)
            if "labels" in h5f:
                return int(h5f["labels"][internal_idx])
        except Exception as exc:
            logger.warning(f"KIDNEY: could not read label at index {index}: {exc}")

        return None

    def __len__(self) -> int:
        if self._epoch_length > 0:
            return self._epoch_length
        if not self._index_mapping:
            self._build_index_mapping()
        return len(self._index_mapping)

    def __del__(self):
        """Close all open HDF5 file handles."""
        if self._file_handles:
            for h5f in self._file_handles.values():
                try:
                    h5f.close()
                except Exception:
                    pass
            self._file_handles = None


    def __repr__(self) -> str:
        split_str = self._split.value if self._split else "all"
        n = len(self)
        return (
            f"KIDNEY(root={self.root}, split={split_str}, "
            f"shards={len(self._hdf5_files)}, samples={n:,d}, "
            f"epoch_length={self._epoch_length})"
        )