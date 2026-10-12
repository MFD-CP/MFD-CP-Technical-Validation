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
from src.networks.single_modal_net import SingleModalNet
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
from src.training.train_utils import (
    build_label_mapping,
    compute_class_weights,
    load_classification_data,
    make_dataloaders,
)
from src.utils.config import load_yaml
from src.utils.io import save_json, write_csv
from src.utils.logger import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="single-modal deep-learning YAML configuration path")
    return parser.parse_args()

def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def _load_configs(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    deep_cfg = load_yaml(path)
    dataset_cfg = load_yaml(_resolve_project_path(deep_cfg["dataset_config"]))
    merged = copy.deepcopy(dataset_cfg)
    merged.update(deep_cfg)
    for key in ["audio", "accel", "gelsight", "visual", "window"]:
        if key in dataset_cfg:
            merged[key] = dataset_cfg[key]
    return deep_cfg, merged

def _build_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = config.get("experiments")
    if experiments:
        return experiments

    rows: list[dict[str, Any]] = []
    for modality in config.get("modalities", ["audio", "accel", "gelsight", "visual"]):
        modality = str(modality).lower()
        encoder = config.get("model", {}).get("modalities", {}).get(modality, {}).get("encoder")
        if encoder is None:
            encoder = "tcn" if modality in {"audio", "accel"} else "resnet18"
        rows.append(
            {
                "name": f"DL-Single-{modality}-{encoder}",
                "modality": modality,
                "encoder": encoder,
            }
        )
    return rows

def _result_row(seed: int, experiment: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "experiment_name": experiment["name"],
        "modality": experiment["modality"],
        "encoder": str(experiment["encoder"]),
        "accuracy_window": metric_from_test_result(metrics, "accuracy_window"),
        "precision_window": metric_from_test_result(metrics, "precision_window"),
        "recall_window": metric_from_test_result(metrics, "recall_window"),
        "f1_window": metric_from_test_result(metrics, "f1_window"),
        "accuracy_trial": metric_from_test_result(metrics, "accuracy_trial"),
        "precision_trial": metric_from_test_result(metrics, "precision_trial"),
        "recall_trial": metric_from_test_result(metrics, "recall_trial"),
        "f1_trial": metric_from_test_result(metrics, "f1_trial"),
    }

def _upsert_result(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = (int(row["seed"]), str(row["experiment_name"]))
    rows[:] = [
        old
        for old in rows
        if (int(old["seed"]), str(old["experiment_name"])) != key
    ]
    rows.append(row)

def _save_progress(
    rows: list[dict[str, Any]],
    output_path: Path,
    all_path: Path,
    experiment_names: set[str],
) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["seed", "experiment_name"]).reset_index(drop=True)
    write_csv(frame, output_path)

    if all_path.exists():
        old = pd.read_csv(all_path)
        old = old[~old["experiment_name"].isin(experiment_names)]
    else:
        old = pd.DataFrame()
    current = pd.DataFrame(
        [
            {
                "seed": int(row["seed"]),
                "experiment_name": row["experiment_name"],
                "method_type": "deep_learning",
                "model": row["encoder"],
                "modality_combo": row["modality"],
                "fusion": "none",
                **{key: row[key] for key in [
                    "accuracy_window", "precision_window", "recall_window", "f1_window",
                    "accuracy_trial", "precision_trial", "recall_trial", "f1_trial",
                ]},
            }
            for row in rows
        ]
    )
    write_csv(pd.concat([old, current], ignore_index=True), all_path)

def _clear_runtime_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main() -> None:
    args = parse_args()
    deep_cfg, cfg = _load_configs(args.config)
    logger = setup_logger("train_deep_single", PROJECT_ROOT / "class/results/logs/train_deep_single.log")

    dataset_root, metadata, split_df = load_classification_data(
        cfg, PROJECT_ROOT
    )

    label_to_index, class_names = build_label_mapping(metadata)
    checkpoint_dir = PROJECT_ROOT / deep_cfg.get("checkpoint_dir", "checkpoints/class")
    save_json(
        {"label_to_index": label_to_index, "class_names": class_names},
        checkpoint_dir / "label_mapping.json",
    )

    experiments = _build_experiments(deep_cfg)
    experiment_names = {str(experiment["name"]) for experiment in experiments}
    output_path = _resolve_project_path(
        deep_cfg.get("output_csv", "class/results/tables/single_modal_deep.csv")
    )
    all_path = _resolve_project_path(
        deep_cfg.get("all_results_csv", "class/results/tables/all_results.csv")
    )
    if output_path.exists():
        rows = pd.read_csv(output_path).to_dict("records")
    else:
        rows: list[dict[str, Any]] = []

    for seed in seed_list(deep_cfg.get("seed", 42)):
        L.seed_everything(seed, workers=True)
        for experiment in experiments:
            result_key = (seed, str(experiment["name"]))
            saved_keys = {
                (int(row["seed"]), str(row["experiment_name"]))
                for row in rows
            }
            if result_key in saved_keys:
                logger.info(
                    "skipping existing result: experiment=%s seed=%s",
                    experiment["name"],
                    seed,
                )
                continue

            modality = str(experiment["modality"]).lower()
            encoder = str(experiment["encoder"]).lower()
            run_cfg = apply_encoder_override(cfg, {modality: encoder})

            loaders = make_dataloaders(
                metadata=metadata,
                split_df=split_df,
                dataset_root=dataset_root,
                modalities=[modality],
                config=run_cfg,
                label_to_index=label_to_index,
                batch_size=int(deep_cfg.get("batch_size", 32)),
                num_workers=int(deep_cfg.get("num_workers", 4)),
            )

            class_weights = None
            if bool(deep_cfg.get("class_weighted_loss", True)):
                train_meta = metadata.merge(split_df[["trial_id", "split"]], on="trial_id")
                train_meta = train_meta[train_meta["split"] == "train"]
                class_weights = compute_class_weights(train_meta, label_to_index)

            model = SingleModalNet(modality, len(class_names), run_cfg)
            lit_module = FabricClassificationLightningModule(
                model=model,
                num_classes=len(class_names),
                learning_rate=float(deep_cfg.get("learning_rate", 3e-4)),
                weight_decay=float(deep_cfg.get("weight_decay", 1e-4)),
                scheduler_name=str(deep_cfg.get("scheduler", "reduce_on_plateau")),
                scheduler_patience=int(deep_cfg.get("scheduler_patience", 5)),
                class_weights=class_weights,
                average=str(deep_cfg.get("average", "macro")),
                trial_aggregation=str(deep_cfg.get("trial_aggregation", "average_logits")),
            )

            run_dirs = prepare_run_dirs(PROJECT_ROOT, experiment["name"], seed, run_cfg)
            callbacks = build_callbacks(run_dirs["checkpoint_dir"], deep_cfg)
            loggers = build_loggers(PROJECT_ROOT, experiment["name"], seed, deep_cfg)
            resume_ckpt = find_resume_checkpoint(run_dirs["checkpoint_dir"], deep_cfg)

            trainer = Trainer(
                accelerator=str(deep_cfg.get("accelerator", "auto")),
                devices=compute_num_devices(deep_cfg),
                max_epochs=int(deep_cfg.get("epochs", 100)),
                precision=torch_precision_setting(deep_cfg),
                deterministic=bool(deep_cfg.get("deterministic", True)),
                logger=loggers,
                callbacks=callbacks,
                default_root_dir=str(PROJECT_ROOT),
                log_every_n_steps=int(deep_cfg.get("log_every_n_steps", 10)),
                enable_checkpointing=True,
            )
            logger.info(
                "starting training: experiment=%s seed=%s resume=%s",
                experiment["name"],
                seed,
                resume_ckpt,
            )
            trainer.fit(
                lit_module,
                train_dataloaders=loaders["train"],
                val_dataloaders=loaders["val"],
                ckpt_path=resume_ckpt,
            )
            test_metrics = trainer.test(
                lit_module,
                dataloaders=loaders["test"],
                ckpt_path="best",
            )[0]

            row = _result_row(seed, experiment, test_metrics)
            _upsert_result(rows, row)
            _save_progress(rows, output_path, all_path, experiment_names)
            logger.info(
                "completed: experiment=%s seed=%s trial_f1=%.4f",
                experiment["name"],
                seed,
                row["f1_trial"],
            )
            del trainer, lit_module, model, loaders, callbacks, loggers
            _clear_runtime_cache()

    _save_progress(rows, output_path, all_path, experiment_names)
    logger.info("results saved: %s", output_path)

if __name__ == "__main__":
    main()
