from __future__ import annotations

import base64
from pathlib import Path
from statistics import quantiles

import cv2
import httpx

from .config import HF_API_TOKEN, HF_MODEL_URL
from .data_access import bucket_by_checkout_count


QUERIES = ["checkout counter", "cash register"]
TRAINING_DIR = Path(__file__).resolve().parent.parent / "treino IA"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}
DEFAULT_ORANGE_LOWER = (5, 80, 80)
DEFAULT_ORANGE_UPPER = (25, 255, 255)
_ORANGE_RANGE_CACHE: tuple[tuple[int, int, int], tuple[int, int, int], bool] | None = None


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    return headers


def _load_training_images() -> list[Path]:
    if not TRAINING_DIR.exists() or not TRAINING_DIR.is_dir():
        return []

    images: list[Path] = []

    for image_path in TRAINING_DIR.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images.append(image_path)

    return images


def _learn_orange_hsv_range() -> tuple[tuple[int, int, int], tuple[int, int, int], bool]:
    global _ORANGE_RANGE_CACHE
    if _ORANGE_RANGE_CACHE is not None:
        return _ORANGE_RANGE_CACHE

    train_images = _load_training_images()
    if not train_images:
        _ORANGE_RANGE_CACHE = (DEFAULT_ORANGE_LOWER, DEFAULT_ORANGE_UPPER, False)
        return _ORANGE_RANGE_CACHE

    hues: list[int] = []
    sats: list[int] = []
    vals: list[int] = []

    for image_path in train_images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        # Pixels quentes e saturados (faixa típica do laranja das placas de checkout).
        mask = (pixels[:, 0] >= 3) & (pixels[:, 0] <= 35) & (pixels[:, 1] >= 70) & (pixels[:, 2] >= 70)
        selected = pixels[mask]
        if selected.size == 0:
            continue

        hues.extend(selected[:, 0].tolist())
        sats.extend(selected[:, 1].tolist())
        vals.extend(selected[:, 2].tolist())

    if len(hues) < 120:
        _ORANGE_RANGE_CACHE = (DEFAULT_ORANGE_LOWER, DEFAULT_ORANGE_UPPER, False)
        return _ORANGE_RANGE_CACHE

    q_h = quantiles(hues, n=10)
    q_s = quantiles(sats, n=10)
    q_v = quantiles(vals, n=10)

    lower = (
        max(0, int(q_h[0]) - 3),
        max(40, int(q_s[0]) - 25),
        max(40, int(q_v[0]) - 25),
    )
    upper = (
        min(179, int(q_h[-1]) + 3),
        min(255, int(q_s[-1]) + 20),
        min(255, int(q_v[-1]) + 20),
    )

    _ORANGE_RANGE_CACHE = (lower, upper, True)
    return _ORANGE_RANGE_CACHE


def _count_checkout_signs(image, image_area: int) -> int:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper, _ = _learn_orange_hsv_range()

    orange_mask = cv2.inRange(hsv, lower, upper)
    orange_mask = cv2.morphologyEx(
        orange_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    sign_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sign_centers: list[int] = []
    min_sign_area = image_area * 0.00002
    max_sign_area = image_area * 0.003
    for contour in sign_contours:
        area = cv2.contourArea(contour)
        if area < min_sign_area or area > max_sign_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.45 or aspect_ratio > 2.8:
            continue

        fill_ratio = area / float(w * h)
        if fill_ratio < 0.2:
            continue

        if y < int(image.shape[0] * 0.1) or y > int(image.shape[0] * 0.78):
            continue

        sign_centers.append(x + w // 2)

    sign_centers.sort()
    unique_sign_centers: list[int] = []
    min_sep = max(22, int(image.shape[1] * 0.018))
    for center_x in sign_centers:
        if not unique_sign_centers or abs(center_x - unique_sign_centers[-1]) > min_sep:
            unique_sign_centers.append(center_x)

    return len(unique_sign_centers)


def _estimate_with_local_cv_raw(image_path: Path) -> tuple[int, str]:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, "erro_leitura_imagem"

    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side > 1400:
        scale = 1400 / max_side
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    image_area = image.shape[0] * image.shape[1]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sign_count = _count_checkout_signs(image, image_area)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[list[int]] = []
    scores: list[float] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.0015 or area > image_area * 0.2:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0:
            continue

        aspect_ratio = w / h
        if aspect_ratio < 0.3 or aspect_ratio > 3.2:
            continue

        fill_ratio = area / float(w * h)
        if fill_ratio < 0.35:
            continue

        boxes.append([x, y, w, h])
        scores.append(float(area))

    if not boxes:
        return max(0, min(sign_count, 60)), "ok_local_cv_signs"

    selected = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=0.35)
    contour_count = len(selected) if selected is not None and len(selected) > 0 else len(boxes)
    raw_count = max(contour_count, sign_count)
    return max(0, min(int(raw_count), 60)), "ok_local_cv"


def _estimate_with_local_cv(image_path: Path) -> tuple[int, str, str]:
    raw_count, raw_status = _estimate_with_local_cv_raw(image_path)
    if raw_status == "erro_leitura_imagem":
        return 0, bucket_by_checkout_count(0), raw_status
    return raw_count, bucket_by_checkout_count(raw_count), raw_status


def _estimate_with_hf(image_path: Path) -> tuple[int, str, str]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "inputs": image_b64,
        "parameters": {"candidate_labels": QUERIES},
        "options": {"wait_for_model": True},
    }

    try:
        with httpx.Client(timeout=40) as client:
            response = client.post(HF_MODEL_URL, headers=_build_headers(), json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception:
        return 0, bucket_by_checkout_count(0), "falha_api_hf"

    detections: list[dict] = data if isinstance(data, list) else []
    count = 0
    for item in detections:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if any(query in label for query in QUERIES) and score >= 0.2:
            count += 1

    return count, bucket_by_checkout_count(count), "ok_hf"


def estimate_checkouts_from_image(image_path: Path) -> tuple[int, str, str]:
    if not image_path.exists():
        return 0, bucket_by_checkout_count(0), "arquivo_nao_encontrado"

    # Prioriza leitura local por visao computacional; usa HF como apoio quando disponivel.
    local_count, local_bucket, local_status = _estimate_with_local_cv(image_path)

    if HF_API_TOKEN:
        hf_count, _, hf_status = _estimate_with_hf(image_path)
        if hf_status == "ok_hf":
            fused_count = max(local_count, hf_count)
            return fused_count, bucket_by_checkout_count(fused_count), f"ok_fusao_local_hf({local_status})"
        return local_count, local_bucket, f"{local_status}_hf_indisponivel"

    return local_count, local_bucket, f"{local_status}_sem_token"