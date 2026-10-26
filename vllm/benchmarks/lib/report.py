# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared measurement envelope for benchmark and evaluation clients."""

import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any


def json_safe(value: Any) -> Any:
    """Represent non-finite values explicitly, without nonstandard JSON tokens."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        # Multimodal bucket CLI settings use tuple keys, spelled as tuples in
        # their JSON input as well (e.g. "(224, 224, 1)").
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def make_report(
    *,
    kind: str,
    name: str,
    settings: dict[str, Any],
    metrics: dict[str, Any],
    records: dict[str, Any],
    status: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report; server provenance is unknown unless supplied by a runner.

    The settings hash identifies client settings, not full run comparability.
    A report is an observation, not an attestation or a correctness certificate.
    """
    settings = json_safe(settings)
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return json_safe(
        {
            "schema": "byog.measurement/v1",
            "kind": kind,
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            "status": status,
            "client": {"vllm_version": importlib.metadata.version("vllm")},
            "server": provenance,
            "settings": settings,
            "settings_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "metrics": metrics,
            "records": records,
        }
    )
