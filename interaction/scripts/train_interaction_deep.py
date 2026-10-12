from __future__ import annotations
import argparse
import copy
import gc
import sys
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import lightning as L
import pandas as pd
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from interaction.offline_adapter import (
    OfflineInteractionDataset,
    build_label_mapping,
    load_interaction_manifest,
    task_split_from_manifest,
)
from src.networks.multimodal_late_fusion_net import MultiModalLateFusionNet
from src.data.offline_dataset import offline_collate_fn
from src.training.lightning_module import FabricClassificationLightningModule
from src.training.lightning_utils import (
    apply_encoder_override,
    build_callbacks,
    build_loggers,
    compute_num_devices,
    find_resume_checkpoint,
    metric_from_test_result,
    prepare_run_dirs,
    seed_list,
    torch_precision_setting,
)
from src.training.train_utils import compute_class_weights
from src.utils.config import load_yaml
from src.utils.io import save_json, write_csv
from src.utils.logger import setup_logger

METRICS = (
    "accuracy_window",
    "precision_window",
    "recall_window",
    "f1_window",
    "accuracy_trial",
    "precision_trial",
    "recall_trial",
    "f1_trial",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train Interaction deep model")
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def load_configs(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    deep_config = load_yaml(path)
    dataset_config = load_yaml(resolve(deep_config["dataset_config"]))
    merged = copy.deepcopy(dataset_config)
    merged.update(deep_config)
    for key in ("audio", "accel", "gelsight", "window"):
        if key in dataset_config:
            merged[key] = dataset_config[key]
    return deep_config, merged

def encoder_name_map(config: dict[str, Any]) -> dict[str, str]:
    modalities = config.get("model", {}).get("modalities", {})
    defaults = {
        "audio": "lstm",
        "accel": "tcn",
        "gelsight": "efficientnet_b0",
    }
    return {
        modality: str(
            modalities.get(modality, {}).get("encoder", default)
        ).lower()
        for modality, default in defaults.items()
    }

def experiment_name(
    task_name: str, combo: list[str], encoders: dict[str, str]
) -> str:
    encoder_text = "-".join(encoders[name] for name in combo)
    modality_text = "".join(
        {"audio": "Audio", "accel": "Accel", "gelsight": "GelSight"}[name]
        for name in combo
    )
    return (
        f"DL-Interaction-{task_name}-{encoder_text}-"
        f"LateConcat-{modality_text}"
    )

def make_dataloaders(
    input_root: Path,
    split_df: pd.DataFrame,
    label_to_index: dict[str, int],
    modalities: list[str],
    config: dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = OfflineInteractionDataset(
            input_root=input_root,
            split=split,
            split_df=split_df,
            label_to_index=label_to_index,
            modalities=modalities,
            config=config,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=offline_collate_fn,
        )
    return loaders

def result_row(
    task_name: str,
    seed: int,
    name: str,
    combo: list[str],
    encoders: dict[str, str],
    val_f1_trial: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "seed": seed,
        "evaluation_mode": "fixed_trial_split",
        "experiment_name": name,
        "modality_combo": "+".join(combo),
        "encoders": "; ".join(
            f"{modality}:{encoders[modality]}" for modality in combo
        ),
        "fusion": "none" if len(combo) == 1 else "late_concat",
        "val_f1_trial": val_f1_trial,
        **{
            metric: metric_from_test_result(metrics, metric)
            for metric in METRICS
        },
    }

def upsert(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = (str(row["task_name"]), int(row["seed"]), str(row["experiment_name"]))
    rows[:] = [
        old
        for old in rows
        if (
            str(old["task_name"]),
            int(old["seed"]),
            str(old["experiment_name"]),
        )
        != key
    ]
    rows.append(row)

def save_progress(
    rows: list[dict[str, Any]], output_path: Path, all_path: Path
) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["task_name", "seed", "experiment_name"]
        ).reset_index(drop=True)
    write_csv(frame, output_path)
    task_name = str(rows[0]["task_name"]) if rows else ""
    if all_path.exists():
        old = pd.read_csv(all_path)
        if {"task_name", "method_type"}.issubset(old.columns):
            old = old[
                ~(
                    (old["task_name"].astype(str) == task_name)
                    & (old["method_type"] == "deep_learning")
                )
            ]
    else:
        old = pd.DataFrame()
    current = frame.copy()
    if not current.empty:
        current["method_type"] = "deep_learning"
        current["model"] = current["encoders"]
    write_csv(pd.concat([old, current], ignore_index=True), all_path)

def clear_runtime_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main() -> None:
    args = parse_args()
    deep_config, config = load_configs(args.config)
    task_name = str(deep_config["task_name"])
    target_label = str(deep_config["target_label"])
    input_root, manifest = load_interaction_manifest(config, PROJECT_ROOT)
    split_df = task_split_from_manifest(manifest, task_name, target_label)
    label_to_index, class_names = build_label_mapping(split_df)
    semantic_names = (
        split_df[["label", "task_label_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["task_label_name"]
        .astype(str)
        .to_dict()
    )

    checkpoint_root = resolve(
        deep_config.get("checkpoint_dir", "checkpoints/interaction")
    )
    save_json(
        {
            "task_name": task_name,
            "target_label": target_label,
            "label_to_index": label_to_index,
            "class_names": class_names,
            "semantic_class_names": semantic_names,
        },
        checkpoint_root / f"label_mapping_{task_name}.json",
    )

    combos = [
        [str(value).lower() for value in combo]
        for combo in deep_config.get("modality_combos", [])
    ]
    if not combos:
        raise ValueError("modality_combos cannot be empty")
    if any("visual" in combo for combo in combos):
        raise ValueError("Interaction does not use Visual modality")
    encoder_defaults = encoder_name_map(config)
    output_path = resolve(deep_config["output_csv"])
    all_path = resolve(
        deep_config.get(
            "all_results_csv",
            "interaction/results/tables/all_interaction_results.csv",
        )
    )
    seeds = seed_list(deep_config.get("seed", [0, 1, 2, 3, 4]))
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if {"seed", "task_name", "experiment_name"}.issubset(existing.columns):
            rows = existing[existing["seed"].astype(int).isin(seeds)].to_dict("records")
        else:
            legacy = output_path.with_name(f"{output_path.stem}_legacy.csv")
            if not legacy.exists():
                write_csv(existing, legacy)
            rows: list[dict[str, Any]] = []
    else:
        rows = []

    logger = setup_logger(
        f"train_interaction_deep_{task_name}",
        PROJECT_ROOT
        / "interaction/results/logs"
        / task_name
        / f"train_deep_{task_name}.log",
    )
    train_labels = split_df[split_df["split"] == "train"][["label"]]
    class_weights = (
        compute_class_weights(train_labels, label_to_index)
        if bool(deep_config.get("class_weighted_loss", True))
        else None
    )

    for seed in seeds:
        L.seed_everything(seed, workers=True)
        for combo in combos:
            encoders = {name: encoder_defaults[name] for name in combo}
            name = experiment_name(task_name, combo, encoders)
            key = (task_name, seed, name)
            existing_keys = {
                (
                    str(row["task_name"]),
                    int(row["seed"]),
                    str(row["experiment_name"]),
                )
                for row in rows
            }
            if key in existing_keys:
                logger.info("skipping existing result：experiment=%s seed=%s", name, seed)
                continue

            run_config = apply_encoder_override(config, encoders)
            loaders = make_dataloaders(
                input_root=input_root,
                split_df=split_df,
                label_to_index=label_to_index,
                modalities=combo,
                config=run_config,
                batch_size=int(deep_config.get("batch_size", 32)),
                num_workers=int(deep_config.get("num_workers", 16)),
            )
            model = MultiModalLateFusionNet(
                combo, len(class_names), run_config
            )
            module = FabricClassificationLightningModule(
                model=model,
                num_classes=len(class_names),
                learning_rate=float(deep_config.get("learning_rate", 3e-4)),
                weight_decay=float(deep_config.get("weight_decay", 1e-4)),
                scheduler_name=str(
                    deep_config.get("scheduler", "reduce_on_plateau")
                ),
                scheduler_patience=int(
                    deep_config.get("scheduler_patience", 5)
                ),
                class_weights=class_weights,
                average=str(deep_config.get("average", "macro")),
                trial_aggregation=str(
                    deep_config.get("trial_aggregation", "average_logits")
                ),
            )
            run_dirs = prepare_run_dirs(
                PROJECT_ROOT, name, seed, run_config
            )
            callbacks = build_callbacks(
                run_dirs["checkpoint_dir"], deep_config
            )
            loggers = build_loggers(PROJECT_ROOT, name, seed, deep_config)
            resume = find_resume_checkpoint(
                run_dirs["checkpoint_dir"], deep_config
            )
            trainer = Trainer(
                accelerator=str(deep_config.get("accelerator", "auto")),
                devices=compute_num_devices(deep_config),
                max_epochs=int(deep_config.get("epochs", 100)),
                precision=torch_precision_setting(deep_config),
                deterministic=bool(deep_config.get("deterministic", True)),
                logger=loggers,
                callbacks=callbacks,
                default_root_dir=str(PROJECT_ROOT),
                log_every_n_steps=int(
                    deep_config.get("log_every_n_steps", 10)
                ),
                enable_checkpointing=True,
            )
            logger.info("starting training：experiment=%s seed=%s", name, seed)
            trainer.fit(
                module,
                train_dataloaders=loaders["train"],
                val_dataloaders=loaders["val"],
                ckpt_path=resume,
            )
            best_score = trainer.checkpoint_callback.best_model_score
            if best_score is None:
                raise RuntimeError(
                    f"{name} seed={seed} did not produce val_f1_trial"
                )
            val_f1_trial = float(
                best_score.detach().cpu().item()
                if isinstance(best_score, torch.Tensor)
                else best_score
            )
            test_metrics = trainer.test(
                module, dataloaders=loaders["test"], ckpt_path="best"
            )[0]
            row = result_row(
                task_name,
                seed,
                name,
                combo,
                encoders,
                val_f1_trial,
                test_metrics,
            )
            upsert(rows, row)
            save_progress(rows, output_path, all_path)
            logger.info(
                "completed：experiment=%s seed=%s trial_f1=%.4f",
                name,
                seed,
                row["f1_trial"],
            )
            del trainer, module, model, loaders, callbacks, loggers
            clear_runtime_cache()

    save_progress(rows, output_path, all_path)

if __name__ == "__main__":
    main()
