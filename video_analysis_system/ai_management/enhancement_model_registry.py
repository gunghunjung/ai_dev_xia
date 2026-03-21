"""Enhancement Model Registry — versioned storage for trained enhancement models."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class EnhancementRegistryEntry:
    model_name: str
    version: str
    task_type: str          # super_resolution / sharpening / denoising / deblurring
    framework: str          # pytorch / onnx / tflite / sklearn
    file_path: str
    scale_factor: int = 1
    psnr: float = 0.0
    ssim: float = 0.0
    status: str = "registered"   # registered / validated / production / archived
    tags: List[str] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class EnhancementModelRegistry:
    """Stores and retrieves enhancement model entries with production promotion."""

    def __init__(self) -> None:
        self._entries: Dict[str, EnhancementRegistryEntry] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, entry: EnhancementRegistryEntry) -> None:
        self._entries[entry.entry_id] = entry

    def get(self, entry_id: str) -> Optional[EnhancementRegistryEntry]:
        return self._entries.get(entry_id)

    def find_by_name(self, name: str) -> Optional[EnhancementRegistryEntry]:
        """Return the first non-archived entry matching the model name."""
        for entry in self._entries.values():
            if entry.model_name == name and entry.status != "archived":
                return entry
        return None

    def list_by_task(self, task_type: str) -> List[EnhancementRegistryEntry]:
        return [
            e for e in self._entries.values()
            if e.task_type == task_type and e.status != "archived"
        ]

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def set_production(self, entry_id: str) -> None:
        """Promote entry to production; demote any existing production entry for same task."""
        target = self._get_or_raise(entry_id)

        # Demote current production entry for this task_type
        for entry in self._entries.values():
            if (
                entry.task_type == target.task_type
                and entry.status == "production"
                and entry.entry_id != entry_id
            ):
                entry.status = "validated"

        target.status = "production"

    def get_production(self, task_type: str) -> Optional[EnhancementRegistryEntry]:
        for entry in self._entries.values():
            if entry.task_type == task_type and entry.status == "production":
                return entry
        return None

    def archive(self, entry_id: str) -> None:
        self._get_or_raise(entry_id).status = "archived"

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def list_all(self) -> List[EnhancementRegistryEntry]:
        return list(self._entries.values())

    def top_by_psnr(self, task_type: str, n: int = 5) -> List[EnhancementRegistryEntry]:
        candidates = [
            e for e in self._entries.values()
            if e.task_type == task_type and e.status != "archived"
        ]
        return sorted(candidates, key=lambda e: e.psnr, reverse=True)[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {eid: asdict(e) for eid, e in self._entries.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._entries = {
            eid: EnhancementRegistryEntry(**v) for eid, v in data.items()
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_raise(self, entry_id: str) -> EnhancementRegistryEntry:
        entry = self._entries.get(entry_id)
        if entry is None:
            raise KeyError(f"Registry entry '{entry_id}' not found.")
        return entry
