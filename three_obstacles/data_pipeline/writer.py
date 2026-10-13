"""Collect and atomically publish a validated raw physical-replay episode."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    MANIFEST_SCHEMA_VERSION,
    EpisodeValidationError,
    ValidationReport,
    canonical_json_bytes,
    sha256_file,
    validate_episode_records,
    validate_frame,
    validate_metadata,
)


class PhysicalReplayEpisodeWriter:
    """Writer that never overwrites an episode or publishes a partial result.

    RGB entries are references to camera-recorded files.  The writer does not
    fabricate, move, or modify images, and it does not import Gazebo or
    LeRobot.
    """

    def __init__(self, episode_dir: Path, metadata: Mapping[str, Any]) -> None:
        self.episode_dir = Path(episode_dir)
        self.metadata = dict(metadata)
        report = validate_metadata(self.metadata)
        if not report.valid:
            raise EpisodeValidationError("invalid metadata: " + "; ".join(report.errors))
        self._frames: list[dict[str, Any]] = []
        self._finalized = False

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def append_frame(self, frame: Mapping[str, Any]) -> None:
        if self._finalized:
            raise RuntimeError("cannot append to a finalized episode")
        record = dict(frame)
        report = validate_frame(
            record,
            self.metadata,
            expected_index=len(self._frames),
            previous=self._frames[-1] if self._frames else None,
        )
        if not report.valid:
            raise EpisodeValidationError("invalid frame: " + "; ".join(report.errors))
        self._frames.append(record)

    def validate(self, summary: Mapping[str, Any]) -> ValidationReport:
        return validate_episode_records(self.metadata, self._frames, summary)

    def finalize(self, summary: Mapping[str, Any]) -> Path:
        if self._finalized:
            raise RuntimeError("episode has already been finalized")
        report = self.validate(summary)
        if not report.valid:
            raise EpisodeValidationError("episode rejected: " + "; ".join(report.errors))
        if self.episode_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing episode directory: {self.episode_dir}")
        self.episode_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{self.episode_dir.name}.inprogress-", dir=self.episode_dir.parent))
        try:
            self._write_json(temporary / "metadata.json", self.metadata)
            self._write_jsonl(temporary / "frames.jsonl", self._frames)
            self._write_json(temporary / "result.json", dict(summary))
            self._write_json(
                temporary / "manifest.json",
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "episode_id": self.metadata["episode_id"],
                    "frame_count": len(self._frames),
                    "file_hashes": {
                        name: sha256_file(temporary / name)
                        for name in ("metadata.json", "frames.jsonl", "result.json")
                    },
                },
            )
            os.replace(temporary, self.episode_dir)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self._finalized = True
        return self.episode_dir

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.write_bytes(canonical_json_bytes(value) + b"\n")

    @staticmethod
    def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
        with path.open("wb") as stream:
            for record in records:
                stream.write(canonical_json_bytes(record) + b"\n")
