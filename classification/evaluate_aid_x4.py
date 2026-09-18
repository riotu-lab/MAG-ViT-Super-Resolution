#!/usr/bin/env python3
"""Evaluate one fixed AID classifier on LR x4, MAG-ViT x4, and FunSR x4.

The classifier checkpoint is loaded exactly once. This is important for a fair
comparison: only the input image source changes between the three conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


AID_CLASSES = [
    "Airport", "BareLand", "BaseballField", "Beach", "Bridge", "Center",
    "Church", "Commercial", "DenseResidential", "Desert", "Farmland",
    "Forest", "Industrial", "Meadow", "MediumResidential", "Mountain",
    "Park", "Parking", "Playground", "Pond", "Port", "RailwayStation",
    "Resort", "River", "School", "SparseResidential", "Square", "Stadium",
    "StorageTanks", "Viaduct",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one fixed classifier checkpoint and evaluate the same model "
            "on LR x4, MAG-ViT x4, and FunSR x4 images."
        )
    )
    parser.add_argument("--model", choices=("maxvit_t", "resnext101_32x8d"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lr-x4", type=Path, required=True)
    parser.add_argument("--magvit-x4", type=Path, required=True)
    parser.add_argument("--funsr-x4", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="For example: cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--expected-images", type=int, default=2000,
                        help="Fail unless every condition has this many images; use 0 to disable")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_label(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


class AIDTestDataset(Dataset):
    """Read either class subfolders or a flat folder with class-prefixed names."""

    def __init__(self, root: Path, class_names: list[str], transform) -> None:
        self.root = root.expanduser().resolve()
        self.class_names = class_names
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")

        normalized_to_index = {
            normalize_label(class_name): index for index, class_name in enumerate(class_names)
        }
        image_paths = sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise RuntimeError(f"No supported images found in {self.root}")

        self.samples: list[tuple[Path, int]] = []
        errors: list[str] = []
        for path in image_paths:
            # Class-subfolder form: root/Airport/image.png
            relative = path.relative_to(self.root)
            if len(relative.parts) > 1:
                label_text = relative.parts[0]
            else:
                # Flat form used in the original scripts: airport_1.png
                label_text = path.stem.split("_", 1)[0]

            normalized = normalize_label(label_text)
            if normalized not in normalized_to_index:
                errors.append(str(path))
                continue
            self.samples.append((path, normalized_to_index[normalized]))

        if errors:
            preview = "\n".join(errors[:10])
            raise RuntimeError(
                f"Could not infer labels for {len(errors)} image(s) in {self.root}. "
                f"First examples:\n{preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label, str(path)


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added
        return torch.load(path, map_location="cpu")


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def build_model(model_name: str, num_classes: int) -> nn.Module:
    # weights=None is intentional: the checkpoint contains the trained weights.
    if model_name == "maxvit_t":
        model = models.maxvit_t(weights=None)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, num_classes)
        return model
    if model_name == "resnext101_32x8d":
        model = models.resnext101_32x8d(weights=None)
        # Preserve the exact head structure used by the original training code;
        # its checkpoint keys are fc.0.weight and fc.0.bias.
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_classes))
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def load_classifier(checkpoint_path: Path, requested_model: str):
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        saved_model = checkpoint.get("model_name")
        class_names = list(checkpoint.get("class_names", AID_CLASSES))
        num_classes = int(checkpoint.get("num_classes", len(class_names)))
    elif isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
        saved_model = None
        class_names = list(AID_CLASSES)
        num_classes = len(class_names)
    else:
        raise RuntimeError(
            "Unsupported checkpoint format. Expected a state_dict or a dictionary containing state_dict."
        )

    if saved_model and saved_model != requested_model:
        raise RuntimeError(
            f"Checkpoint says model_name={saved_model!r}, but --model={requested_model!r}."
        )
    if num_classes != len(class_names):
        raise RuntimeError(
            f"Checkpoint num_classes={num_classes}, but it contains {len(class_names)} class names."
        )
    if set(map(normalize_label, class_names)) != set(map(normalize_label, AID_CLASSES)):
        raise RuntimeError("Checkpoint class_names do not match the 30 AID classes.")

    model = build_model(requested_model, num_classes)
    model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    return model, class_names, checkpoint_path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []

    model.eval()
    for images, targets, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(loss_function(logits, targets).item())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(targets.cpu().numpy())
        paths.extend(batch_paths)

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels)
    return total_loss / len(y_true), y_pred, y_true, paths


def write_predictions(
    output_path: Path,
    paths: Iterable[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "true_idx", "true_class", "pred_idx", "pred_class", "correct"])
        for path, true_idx, pred_idx in zip(paths, labels, predictions):
            writer.writerow([
                path,
                int(true_idx),
                class_names[int(true_idx)],
                int(pred_idx),
                class_names[int(pred_idx)],
                int(true_idx == pred_idx),
            ])


def evaluate_condition(
    name: str,
    root: Path,
    model: nn.Module,
    class_names: list[str],
    transform,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    expected_images: int,
) -> dict:
    dataset = AIDTestDataset(root, class_names, transform)
    if expected_images and len(dataset) != expected_images:
        raise RuntimeError(
            f"{name} contains {len(dataset)} images; expected {expected_images}. "
            "Use --expected-images 0 only if this difference is intentional."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    loss, predictions, labels, paths = predict(model, loader, device)
    correct = int((predictions == labels).sum())
    accuracy = correct / len(labels)

    condition_dir = output_root / name
    condition_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        condition_dir / "per_image_predictions.csv",
        paths,
        labels,
        predictions,
        class_names,
    )

    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    np.save(condition_dir / "confusion_matrix.npy", matrix)
    np.savetxt(condition_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    report = classification_report(
        labels,
        predictions,
        labels=range(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (condition_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    result = {
        "condition": name,
        "dataset_root": str(dataset.root),
        "num_images": len(dataset),
        "correct": correct,
        "loss": loss,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
    }
    (condition_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}), but CUDA is unavailable.")
    device = torch.device(args.device)

    model, class_names, checkpoint_path = load_classifier(args.checkpoint, args.model)
    model.to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    conditions = [
        ("LR_x4", args.lr_x4),
        ("MAGViT_x4", args.magvit_x4),
        ("FunSR_x4", args.funsr_x4),
    ]

    results = []
    for name, root in conditions:
        result = evaluate_condition(
            name=name,
            root=root,
            model=model,
            class_names=class_names,
            transform=preprocessing,
            output_root=args.output_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            expected_images=args.expected_images,
        )
        results.append(result)
        print(
            f"{name}: {result['correct']}/{result['num_images']} "
            f"= {result['accuracy_percent']:.4f}%"
        )

    checkpoint_hash = sha256_file(checkpoint_path)
    summary_path = args.output_dir / "table_viii_x4_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "classifier", "checkpoint", "checkpoint_sha256", "condition",
            "dataset_root", "num_images", "correct", "loss", "accuracy",
            "accuracy_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "classifier": args.model,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                **result,
            })

    run_metadata = {
        "classifier": args.model,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "expected_images": args.expected_images,
        "class_names": class_names,
        "preprocessing": {
            "resize": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
