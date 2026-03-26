from __future__ import annotations

from pathlib import Path

import httpx

from .config import HF_API_TOKEN, HF_MODEL_URL
from .data_access import bucket_by_checkout_count


QUERIES = ["checkout counter", "cash register"]


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    return headers


def estimate_checkouts_from_image(image_path: Path) -> tuple[int, str, str]:
    if not image_path.exists():
        return 0, bucket_by_checkout_count(0), "arquivo_nao_encontrado"

    payload = {
        "inputs": {
            "image": image_path.read_bytes(),
            "parameters": {"candidate_labels": QUERIES},
        }
    }

    try:
        with httpx.Client(timeout=40) as client:
            response = client.post(HF_MODEL_URL, headers=_build_headers(), json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return 0, bucket_by_checkout_count(0), "falha_api"

    detections: list[dict] = data if isinstance(data, list) else []
    count = 0
    for item in detections:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if any(query in label for query in QUERIES) and score >= 0.2:
            count += 1

    return count, bucket_by_checkout_count(count), "ok"