"""Enhancement Dataset Manager — manages low/high quality image pair datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class EnhancementDatasetInfo:
    dataset_id: str
    name: str
    description: str
    source_dir: str          # low quality images
    target_dir: str          # high quality images
    task_type: str           # super_resolution / sharpening / denoising / deblurring
    sample_count: int = 0
    split_ratio: float = 0.8  # train fraction; (1 - split_ratio) = val
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    paired: bool = True


class EnhancementDatasetManager:
    """Registry and accessor for image enhancement datasets."""

    def __init__(self) -> None:
        self._registry: dict[str, EnhancementDatasetInfo] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, info: EnhancementDatasetInfo) -> None:
        self._registry[info.dataset_id] = info

    def get(self, dataset_id: str) -> Optional[EnhancementDatasetInfo]:
        return self._registry.get(dataset_id)

    def list_all(self) -> List[EnhancementDatasetInfo]:
        return list(self._registry.values())

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_samples(self, dataset_id: str) -> int:
        """Count image files in source_dir and update sample_count."""
        info = self._registry.get(dataset_id)
        if info is None:
            raise KeyError(f"Dataset '{dataset_id}' not found.")
        source = Path(info.source_dir)
        if not source.exists():
            return 0
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp")
        files = []
        for ext in extensions:
            files.extend(source.glob(ext))
        info.sample_count = len(files)
        return info.sample_count

    # ------------------------------------------------------------------
    # Pair retrieval
    # ------------------------------------------------------------------

    def get_pairs(
        self, dataset_id: str, split: str = "train"
    ) -> List[Tuple[str, str]]:
        """Return (source_path, target_path) pairs for a given split."""
        info = self._registry.get(dataset_id)
        if info is None:
            raise KeyError(f"Dataset '{dataset_id}' not found.")

        source_dir = Path(info.source_dir)
        target_dir = Path(info.target_dir)
        extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

        sources = sorted(
            p for p in source_dir.iterdir()
            if p.suffix.lower() in extensions
        ) if source_dir.exists() else []

        pairs: List[Tuple[str, str]] = []
        for src in sources:
            # Look for matching target by stem (same filename, any supported ext)
            tgt = None
            for ext in extensions:
                candidate = target_dir / (src.stem + ext)
                if candidate.exists():
                    tgt = candidate
                    break
            if tgt is not None:
                pairs.append((str(src), str(tgt)))

        # Split
        cut = int(len(pairs) * info.split_ratio)
        if split == "train":
            return pairs[:cut]
        else:
            return pairs[cut:]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_registry(self, path: str) -> None:
        data = {k: asdict(v) for k, v in self._registry.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def load_registry(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._registry = {
            k: EnhancementDatasetInfo(**v) for k, v in data.items()
        }
