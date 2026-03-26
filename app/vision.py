from __future__ import annotations

import base64
import csv
from pathlib import Path
import re
from statistics import median

import cv2
import httpx

from .config import HF_API_TOKEN, HF_MODEL_URL
from .data_access import bucket_by_checkout_count


QUERIES = ["checkout counter", "cash register"]
TRAINING_DIR = Path(__file__).resolve().parent.parent / "treino IA"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    return headers


def _extract_label_from_filename(name: str) -> int | None:
    text = name.lower()
    patterns = [
        r"(?:checkout|checkouts|caixa|caixas|cx|qtd|qtde)[_\- ]*(\d{1,2})",
        r"(\d{1,2})[_\- ]*(?:checkout|checkouts|caixa|caixas|cx)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if 0 <= value <= 60:
                return value
    return None


def _load_labels_csv() -> dict[str, int]:
    labels_path = TRAINING_DIR / "labels.csv"
    if not labels_path.exists():
        return {}

    labels: dict[str, int] = {}
    try:
        with labels_path.open("r", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                filename = str(row.get("arquivo") or row.get("file") or "").strip()
                value_text = str(row.get("checkouts") or row.get("qtd") or row.get("count") or "").strip()
                if not filename or not value_text:
                    continue
                if value_text.isdigit():
                    value = int(value_text)
                    if 0 <= value <= 60:
                        labels[filename.lower()] = value
    except Exception:
        return {}
    return labels


def _load_training_samples() -> list[tuple[Path, int]]:
    if not TRAINING_DIR.exists() or not TRAINING_DIR.is_dir():
        return []

    csv_labels = _load_labels_csv()
    samples: list[tuple[Path, int]] = []

    for image_path in TRAINING_DIR.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = csv_labels.get(image_path.name.lower())
        if label is None:
            label = _extract_label_from_filename(image_path.name)
        if label is None:
            label = _extract_label_from_filename(image_path.parent.name)

        if label is not None:
            samples.append((image_path, label))

    return samples


def _calibrate_with_training(raw_count: int) -> tuple[int, str]:
    samples = _load_training_samples()
    if len(samples) < 2:
        return raw_count, "ok_local_cv_sem_treino"

    predicted: list[int] = []
    actual: list[int] = []

    for sample_path, label in samples:
        predicted_count, _ = _estimate_with_local_cv_raw(sample_path)
        predicted.append(max(predicted_count, 1))
        actual.append(label)

    ratios = [a / p for p, a in zip(predicted, actual) if p > 0]
    offsets = [a - p for p, a in zip(predicted, actual)]
    if not ratios:
        return raw_count, "ok_local_cv_sem_treino"

    scale = median(ratios)
    offset = median(offsets)
    calibrated = int(round((raw_count * scale) + (offset * 0.35)))
    calibrated = max(0, min(calibrated, 60))
    return calibrated, "ok_local_cv_calibrado"


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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Heuristica 1: contar placas laranja de identificacao de checkout (ex.: 04, 05, 06).
    orange_mask = cv2.inRange(hsv, (5, 80, 80), (25, 255, 255))
    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    sign_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sign_centers: list[int] = []
    min_sign_area = image_area * 0.00002
    max_sign_area = image_area * 0.0025
    for contour in sign_contours:
        area = cv2.contourArea(contour)
        if area < min_sign_area or area > max_sign_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.5 or aspect_ratio > 2.5:
            continue

        fill_ratio = area / float(w * h)
        if fill_ratio < 0.2:
            continue

        # Ignora cantos superiores e inferiores extremos, onde geralmente ha ruido.
        if y < int(image.shape[0] * 0.12) or y > int(image.shape[0] * 0.75):
            continue

        sign_centers.append(x + w // 2)

    sign_centers.sort()
    unique_sign_centers: list[int] = []
    min_sep = max(22, int(image.shape[1] * 0.02))
    for center_x in sign_centers:
        if not unique_sign_centers or abs(center_x - unique_sign_centers[-1]) > min_sep:
            unique_sign_centers.append(center_x)

    sign_count = len(unique_sign_centers)

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

    calibrated_count, calib_status = _calibrate_with_training(raw_count)
    if calib_status == "ok_local_cv_sem_treino":
        return raw_count, bucket_by_checkout_count(raw_count), raw_status
    return calibrated_count, bucket_by_checkout_count(calibrated_count), calib_status


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

    # Prefer Hugging Face when token is configured; fallback to local CV otherwise.
    if HF_API_TOKEN:
        count, bucket, status = _estimate_with_hf(image_path)
        if status == "ok_hf":
            return count, bucket, status

    local_count, local_bucket, local_status = _estimate_with_local_cv(image_path)
    if HF_API_TOKEN:
        return local_count, local_bucket, f"{local_status}_fallback_hf"
    return local_count, local_bucket, f"{local_status}_sem_token"