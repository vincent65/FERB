from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RollbackManager:
    candidate_file: Path

    @property
    def backup_file(self) -> Path:
        return self.candidate_file.with_suffix(self.candidate_file.suffix + ".bak")

    def snapshot(self) -> None:
        self.backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.candidate_file, self.backup_file)

    def rollback(self) -> None:
        if self.backup_file.exists():
            shutil.copy2(self.backup_file, self.candidate_file)
