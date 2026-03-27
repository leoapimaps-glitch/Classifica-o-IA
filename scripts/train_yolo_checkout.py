from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = BASE_DIR / "treino IA" / "dataset_yolo" / "data.yaml"
RUNS_DIR = BASE_DIR / "treino IA" / "runs"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train() -> None:
    if not DATASET_YAML.exists():
        raise RuntimeError(
            "Dataset nao encontrado. Rode antes: python scripts/export_yolo_dataset.py"
        )

    model = YOLO("yolov8n.pt")
    result = model.train(
        data=str(DATASET_YAML),
        epochs=60,
        imgsz=960,
        batch=8,
        project=str(RUNS_DIR),
        name="checkout_detector",
        patience=12,
        workers=2,
        verbose=True,
    )

    best_path = Path(result.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise RuntimeError("Treino concluido, mas arquivo best.pt nao foi encontrado.")

    target = MODELS_DIR / "checkout_yolo.pt"
    target.write_bytes(best_path.read_bytes())
    print(f"Modelo salvo em: {target}")


if __name__ == "__main__":
    train()
