from __future__ import annotations
from pathlib import Path
import argparse
import csv
import gc
import json
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from preprocess_config import load_classification_fabric_ids

RANDOM_SEED = 20260730
EXPECTED_SOURCE_IMAGES_PER_FABRIC = 10
TRAIN_SOURCE_IMAGES_PER_FABRIC = 6
VAL_SOURCE_IMAGES_PER_FABRIC = 2
TEST_SOURCE_IMAGES_PER_FABRIC = 2
OUTPUT_IMAGES_PER_FABRIC = {
    "train": 600,
    "val": 200,
    "test": 200,
}
AUGMENT_SPLITS = {
    "train": True,
    "val": True,
    "test": True,
}
OUTPUT_IMAGE_SIZE = (224, 224)
WORKING_IMAGE_SIZE = (256, 256)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RESAMPLE_METHOD = "bilinear"
HORIZONTAL_FLIP_PROBABILITY = 0.5
VERTICAL_FLIP_PROBABILITY = 0.5
RANDOM_CROP_SCALE_RANGE = (0.80, 1.00)
RANDOM_CROP_ASPECT_RANGE = (0.90, 1.10)
RANDOM_ROTATION_DEGREES = 10.0
BRIGHTNESS_FACTOR_RANGE = (0.90, 1.10)
CONTRAST_FACTOR_RANGE = (0.90, 1.10)
SATURATION_FACTOR_RANGE = (0.90, 1.10)
GAUSSIAN_BLUR_PROBABILITY = 0.15
GAUSSIAN_BLUR_RADIUS_RANGE = (0.10, 1.00)
GAUSSIAN_NOISE_PROBABILITY = 0.50
GAUSSIAN_NOISE_STD_RANGE_255 = (1.0, 5.0)
PT_USE_NEW_ZIPFILE_SERIALIZATION = True
STRICT_VALIDATION = True
OVERWRITE_OUTPUTS = False
VISUAL_MANIFEST_NAME = "visual_source_split_manifest.csv"
VISUAL_SUMMARY_NAME = "visual_preprocessing_summary.json"

SPLITS = ("train", "val", "test")

@dataclass(frozen=True)
class SourceImage:
    fabric_id: int
    source_index: int
    path: Path
    split: str

def natural_key(text: str) -> tuple[Any, ...]:
    import re

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )

def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

def discover_and_split_sources(
    classification_root: Path,
    fabric_ids: list[int],
) -> list[SourceImage]:
    if not classification_root.is_dir():
        raise FileNotFoundError(
            f"visual data root directory does not exist：{classification_root}"
        )
    configured_total = (
        TRAIN_SOURCE_IMAGES_PER_FABRIC
        + VAL_SOURCE_IMAGES_PER_FABRIC
        + TEST_SOURCE_IMAGES_PER_FABRIC
    )
    if configured_total != EXPECTED_SOURCE_IMAGES_PER_FABRIC:
        raise ValueError(
            "configured Visual source-image split counts do not sum to the expected count"
        )
    if set(OUTPUT_IMAGES_PER_FABRIC) != set(SPLITS):
        raise ValueError("OUTPUT_IMAGES_PER_FABRIC must define train/val/test")
    if sum(OUTPUT_IMAGES_PER_FABRIC.values()) != 1_000:
        raise ValueError(
            "per-fabric Visual total output count must be 1000"
        )
    if set(AUGMENT_SPLITS) != set(SPLITS):
        raise ValueError("AUGMENT_SPLITS must define train/val/test")

    records: list[SourceImage] = []
    for fabric_id in fabric_ids:
        fabric_dir = classification_root / f"fabric_{fabric_id:03d}"
        if not fabric_dir.is_dir():
            raise FileNotFoundError(
                f"fabric {fabric_id} directory does not exist：{fabric_dir}"
            )
        paths = sorted(
            (
                path
                for path in fabric_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: natural_key(path.name),
        )
        if (
            STRICT_VALIDATION
            and len(paths) != EXPECTED_SOURCE_IMAGES_PER_FABRIC
        ):
            raise RuntimeError(
                f"fabric {fabric_id}：expected "
                f"{EXPECTED_SOURCE_IMAGES_PER_FABRIC} source images，"
                f"actually found {len(paths)} "
            )
        rng = random.Random(RANDOM_SEED + fabric_id)
        rng.shuffle(paths)
        test_end = TEST_SOURCE_IMAGES_PER_FABRIC
        val_end = test_end + VAL_SOURCE_IMAGES_PER_FABRIC
        split_paths = {
            "test": paths[:test_end],
            "val": paths[test_end:val_end],
            "train": paths[val_end:],
        }
        source_index_by_path = {
            path: index
            for index, path in enumerate(
                sorted(paths, key=lambda item: natural_key(item.name))
            )
        }
        for split in SPLITS:
            for path in split_paths[split]:
                records.append(
                    SourceImage(
                        fabric_id=fabric_id,
                        source_index=source_index_by_path[path],
                        path=path,
                        split=split,
                    )
                )
    return sorted(
        records,
        key=lambda row: (
            SPLITS.index(row.split),
            row.fabric_id,
            row.source_index,
        ),
    )

def pil_resampling() -> int:
    return {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[RESAMPLE_METHOD]

def random_resized_crop(
    image: Image.Image, rng: np.random.Generator
) -> Image.Image:
    width, height = image.size
    area = width * height
    for _ in range(10):
        scale = rng.uniform(*RANDOM_CROP_SCALE_RANGE)
        aspect = rng.uniform(*RANDOM_CROP_ASPECT_RANGE)
        crop_width = int(round(np.sqrt(area * scale * aspect)))
        crop_height = int(round(np.sqrt(area * scale / aspect)))
        if 1 <= crop_width <= width and 1 <= crop_height <= height:
            left = int(rng.integers(0, width - crop_width + 1))
            top = int(rng.integers(0, height - crop_height + 1))
            return image.crop(
                (left, top, left + crop_width, top + crop_height)
            ).resize(WORKING_IMAGE_SIZE, resample=pil_resampling())
    return ImageOps.fit(
        image,
        WORKING_IMAGE_SIZE,
        method=pil_resampling(),
        centering=(0.5, 0.5),
    )

def augment_image(path: Path, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    with Image.open(path) as source:
        image = random_resized_crop(source.convert("RGB"), rng)
    if rng.random() < HORIZONTAL_FLIP_PROBABILITY:
        image = ImageOps.mirror(image)
    if rng.random() < VERTICAL_FLIP_PROBABILITY:
        image = ImageOps.flip(image)
    angle = float(
        rng.uniform(-RANDOM_ROTATION_DEGREES, RANDOM_ROTATION_DEGREES)
    )
    border_pixels = np.concatenate(
        (
            np.asarray(image)[0, :, :],
            np.asarray(image)[-1, :, :],
            np.asarray(image)[:, 0, :],
            np.asarray(image)[:, -1, :],
        ),
        axis=0,
    )
    fill_color = tuple(
        int(value) for value in np.median(border_pixels, axis=0)
    )
    image = image.rotate(
        angle,
        resample=pil_resampling(),
        fillcolor=fill_color,
    )
    image = ImageEnhance.Brightness(image).enhance(
        float(rng.uniform(*BRIGHTNESS_FACTOR_RANGE))
    )
    image = ImageEnhance.Contrast(image).enhance(
        float(rng.uniform(*CONTRAST_FACTOR_RANGE))
    )
    image = ImageEnhance.Color(image).enhance(
        float(rng.uniform(*SATURATION_FACTOR_RANGE))
    )
    if rng.random() < GAUSSIAN_BLUR_PROBABILITY:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=float(rng.uniform(*GAUSSIAN_BLUR_RADIUS_RANGE))
            )
        )
    image = ImageOps.fit(
        image,
        OUTPUT_IMAGE_SIZE,
        method=pil_resampling(),
        centering=(0.5, 0.5),
    )
    array = np.asarray(image, dtype=np.float32)
    if rng.random() < GAUSSIAN_NOISE_PROBABILITY:
        noise_std = float(rng.uniform(*GAUSSIAN_NOISE_STD_RANGE_255))
        array += rng.normal(0.0, noise_std, size=array.shape)
    array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return np.transpose(array, (2, 0, 1))

def deterministic_resize(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            OUTPUT_IMAGE_SIZE, resample=pil_resampling()
        )
        array = np.asarray(image, dtype=np.uint8)
    return np.transpose(array, (2, 0, 1))

def augmentation_seed(
    split: str,
    fabric_id: int,
    source_index: int,
    augmentation_index: int,
) -> int:
    split_offset = {"train": 0, "val": 100_000_019, "test": 200_000_033}[split]
    return (
        RANDOM_SEED
        + split_offset
        + fabric_id * 1_000_003
        + source_index * 10_007
        + augmentation_index
    )

def output_plan(
    records: list[SourceImage], split: str
) -> list[tuple[SourceImage, int]]:
    split_records = [row for row in records if row.split == split]
    by_fabric: dict[int, list[SourceImage]] = {}
    for record in split_records:
        by_fabric.setdefault(record.fabric_id, []).append(record)
    plan: list[tuple[SourceImage, int]] = []
    for fabric_id in sorted(by_fabric):
        sources = sorted(
            by_fabric[fabric_id], key=lambda row: row.source_index
        )
        for output_index in range(OUTPUT_IMAGES_PER_FABRIC[split]):
            source = sources[output_index % len(sources)]
            augmentation_index = output_index // len(sources)
            plan.append((source, augmentation_index))
    return plan

def atomic_torch_save(payload: dict[str, Any], path: Path, overwrite: bool) -> None:
    import torch

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output file already exists：{path}。use this option to replace them --overwrite。"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(
        payload,
        temporary,
        _use_new_zipfile_serialization=PT_USE_NEW_ZIPFILE_SERIALIZATION,
    )
    os.replace(temporary, path)

def process_split(
    records: list[SourceImage],
    split: str,
    output_root: Path,
    overwrite: bool,
) -> int:
    import torch

    plan = output_plan(records, split)
    total = len(plan)
    channels, height, width = 3, OUTPUT_IMAGE_SIZE[1], OUTPUT_IMAGE_SIZE[0]
    output_root.mkdir(parents=True, exist_ok=True)
    descriptor, mmap_name = tempfile.mkstemp(
        dir=output_root,
        prefix=f".visual_{split}.",
        suffix=".uint8.mmap",
    )
    os.close(descriptor)
    mmap_path = Path(mmap_name)
    images = np.memmap(
        mmap_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total, channels, height, width),
    )
    fabric_ids = np.empty(total, dtype=np.int32)
    source_indices = np.empty(total, dtype=np.int16)
    augmentation_indices = np.empty(total, dtype=np.int32)
    source_paths: list[str] = []
    try:
        for index, (record, augmentation_index) in enumerate(plan):
            if AUGMENT_SPLITS[split]:
                seed = augmentation_seed(
                    split,
                    record.fabric_id,
                    record.source_index,
                    augmentation_index,
                )
                images[index] = augment_image(record.path, seed)
            else:
                images[index] = deterministic_resize(record.path)
            fabric_ids[index] = record.fabric_id
            source_indices[index] = record.source_index
            augmentation_indices[index] = augmentation_index
            source_paths.append(str(record.path))
            if (index + 1) % 500 == 0 or index + 1 == total:
                print(
                    f"[Visual:{split}] processed {index + 1}/{total} images",
                    flush=True,
                )
        images.flush()
        payload = {
            "images": torch.from_numpy(images),
            "labels": {"fabric_id": torch.from_numpy(fabric_ids)},
            "metadata": {
                "source_index": torch.from_numpy(source_indices),
                "augmentation_index": torch.from_numpy(
                    augmentation_indices
                ),
                "source_paths": source_paths,
            },
            "config": {
                "random_seed": RANDOM_SEED,
                "image_size": list(OUTPUT_IMAGE_SIZE),
                "image_dtype": "uint8",
                "image_layout": "NCHW_RGB",
                "source_split_counts_per_fabric": {
                    "train": TRAIN_SOURCE_IMAGES_PER_FABRIC,
                    "val": VAL_SOURCE_IMAGES_PER_FABRIC,
                    "test": TEST_SOURCE_IMAGES_PER_FABRIC,
                },
                "output_images_per_fabric": OUTPUT_IMAGES_PER_FABRIC,
                "augmented_splits": AUGMENT_SPLITS,
                "evaluation_unit": "source_image",
            },
        }
        atomic_torch_save(
            payload,
            output_root / f"visual_{split}.pt",
            overwrite=overwrite,
        )
        del payload
    finally:
        del images
        gc.collect()
        mmap_path.unlink(missing_ok=True)
    return total

def write_manifest(records: list[SourceImage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["fabric_id", "source_index", "split", "source_path"]
        )
        for record in records:
            writer.writerow(
                [
                    record.fabric_id,
                    record.source_index,
                    record.split,
                    record.path,
                ]
            )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)

def statistics(records: list[SourceImage]) -> dict[str, Any]:
    fabric_ids = sorted({record.fabric_id for record in records})
    return {
        "num_fabrics": len(fabric_ids),
        "fabric_ids": fabric_ids,
        "source_images": len(records),
        "source_images_per_split": {
            split: sum(record.split == split for record in records)
            for split in SPLITS
        },
        "planned_output_images_per_split": {
            split: len(output_plan(records, split)) for split in SPLITS
        },
        "random_seed": RANDOM_SEED,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="offline-preprocess and augment Visual modality。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan only、validate and print statistics，do not write files。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_OUTPUTS,
        help="overwrite existing generated results using atomic replacement。",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_reproducible_seed(RANDOM_SEED)
    fabric_ids = load_classification_fabric_ids(args.config)
    records = discover_and_split_sources(
        args.classification_root, fabric_ids
    )
    stats = statistics(records)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("read-only validation completed：no files were written。", flush=True)
        return

    targets = [args.output_root / f"visual_{split}.pt" for split in SPLITS]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "the following outputs already exist，use this option to replace them --overwrite："
            + ", ".join(str(path) for path in existing)
        )
    write_manifest(
        records, args.output_root / VISUAL_MANIFEST_NAME
    )
    stats["written_output_images_per_split"] = {
        split: process_split(
            records, split, args.output_root, args.overwrite
        )
        for split in SPLITS
    }
    (args.output_root / VISUAL_SUMMARY_NAME).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"processing completed，output directory：{args.output_root}", flush=True)

if __name__ == "__main__":
    main()
