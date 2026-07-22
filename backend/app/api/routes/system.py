"""System info endpoints — disk space, path recommendations."""

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(tags=["system"])


class DiskInfo(BaseModel):
    path: str
    total_gb: float
    free_gb: float
    used_pct: float
    writable: bool


class DiskInfoResponse(BaseModel):
    recommended: str
    candidates: list[DiskInfo]


CANDIDATE_PATHS = [
    "/root/models/her-axera",
    "/opt/models/her-axera",
    "/home/models/her-axera",
    "/data/models/her-axera",
    "/tmp/models/her-axera",
]


def _get_disk_info(path: str) -> DiskInfo:
    p = Path(path)
    # Walk up to find existing parent
    test_path = p
    while test_path != test_path.parent and not test_path.exists():
        test_path = test_path.parent
    try:
        usage = shutil.disk_usage(str(test_path))
        writable = os.access(str(test_path), os.W_OK)
        return DiskInfo(
            path=path,
            total_gb=round(usage.total / (1024**3), 1),
            free_gb=round(usage.free / (1024**3), 1),
            used_pct=round((usage.used / usage.total) * 100, 1),
            writable=writable,
        )
    except OSError:
        return DiskInfo(path=path, total_gb=0, free_gb=0, used_pct=100, writable=False)


@router.get("/system/disk", response_model=DiskInfoResponse)
def get_disk_info() -> DiskInfoResponse:
    """Return disk space info for candidate model storage paths."""
    candidates = [_get_disk_info(p) for p in CANDIDATE_PATHS]

    # Recommended: pick first writable path with > 1GB free
    recommended = CANDIDATE_PATHS[0]
    for c in candidates:
        if c.writable and c.free_gb > 1:
            recommended = c.path
            break

    return DiskInfoResponse(recommended=recommended, candidates=candidates)
