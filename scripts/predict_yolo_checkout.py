from __future__ import annotations

from pathlib import Path
import sys

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "checkout_yolo.pt"


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python scripts/predict_yolo_checkout.py caminho_da_imagem")
        return

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Imagem nao encontrada: {image_path}")
        return

    if not MODEL_PATH.exists():
        print(f"Modelo nao encontrado em {MODEL_PATH}. Rode o treino antes.")
        return

    model = YOLO(str(MODEL_PATH))
    result = model.predict(source=str(image_path), conf=0.30, verbose=False)[0]
    boxes = result.boxes
    count = int(len(boxes)) if boxes is not None else 0
    print(f"Checkouts detectados: {count}")


if __name__ == "__main__":
    main()
