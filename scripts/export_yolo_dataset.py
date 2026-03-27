from __future__ import annotations

from pathlib import Path
import random
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_IMAGES = BASE_DIR / "treino IA" / "images"
TRAINING_LABELS = BASE_DIR / "treino IA" / "labels"
DATASET_DIR = BASE_DIR / "treino IA" / "dataset_yolo"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}


def _safe_unlink(path: Path) -> None:
    if path.exists():
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)


def export_dataset(train_ratio: float = 0.85) -> None:
    if not TRAINING_IMAGES.exists() or not TRAINING_LABELS.exists():
        raise RuntimeError("Pasta de treino nao encontrada. Use o ajuste manual no painel admin primeiro.")

    images = [p for p in TRAINING_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
    pairs: list[tuple[Path, Path]] = []
    for image in images:
        label = TRAINING_LABELS / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))

    if len(pairs) < 5:
        raise RuntimeError("Poucas amostras para treino. Anote pelo menos 5 imagens no admin.")

    random.shuffle(pairs)
    train_size = max(1, int(len(pairs) * train_ratio))
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:] or pairs[-1:]

    _safe_unlink(DATASET_DIR)
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    def copy_split(split_pairs: list[tuple[Path, Path]], split_name: str) -> None:
        for image, label in split_pairs:
            shutil.copy2(image, DATASET_DIR / "images" / split_name / image.name)
            shutil.copy2(label, DATASET_DIR / "labels" / split_name / label.name)

    copy_split(train_pairs, "train")
    copy_split(val_pairs, "val")

    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {DATASET_DIR.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: checkout",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Dataset YOLO exportado em: {DATASET_DIR}")
    print(f"Treino: {len(train_pairs)} | Validacao: {len(val_pairs)}")


if __name__ == "__main__":
    export_dataset()
