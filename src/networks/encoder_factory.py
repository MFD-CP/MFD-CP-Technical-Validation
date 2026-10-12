from __future__ import annotations
from copy import deepcopy
from typing import Any
from torch import nn
from src.networks.image.image_encoder import ImageEncoder
from src.networks.temporal.lstm_encoder import LSTMEncoder
from src.networks.temporal.tcn_encoder import TCNEncoder

TEMPORAL_ENCODERS = {"tcn", "lstm"}
IMAGE_ENCODERS = {"efficientnet_b0"}

def _get_model_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = config or {}
    return cfg.get("model", cfg)

def get_modality_encoder_name(config: dict[str, Any], modality: str) -> str:
    model_cfg = _get_model_cfg(config)
    modality = modality.lower()

    override = model_cfg.get("encoder_by_modality", {}).get(modality)
    if isinstance(override, str):
        return override.lower()
    if isinstance(override, dict) and "type" in override:
        return str(override["type"]).lower()

    modality_cfg = model_cfg.get("modalities", {}).get(modality, {})
    if "encoder" in modality_cfg:
        return str(modality_cfg["encoder"]).lower()

    if modality in {"audio", "accel"}:
        return str(model_cfg.get("temporal_encoder", "tcn")).lower()
    if modality in {"gelsight", "visual"}:
        return str(model_cfg.get("image_encoder", "efficientnet_b0")).lower()
    raise ValueError(f"unknown modality: {modality}")

def _encoder_specific_cfg(model_cfg: dict[str, Any], encoder_name: str, modality: str) -> dict[str, Any]:
    encoder_name = encoder_name.lower()
    modality = modality.lower()
    params: dict[str, Any] = {}

    if encoder_name in TEMPORAL_ENCODERS:
        params.update(deepcopy(model_cfg.get("temporal", {}).get(encoder_name, {})))
    elif encoder_name in IMAGE_ENCODERS:
        params.update(deepcopy(model_cfg.get("image", {}).get(encoder_name, {})))

    override = model_cfg.get("encoder_by_modality", {}).get(modality)
    if isinstance(override, dict):
        params.update(deepcopy(override.get("params", {})))

    modality_cfg = model_cfg.get("modalities", {}).get(modality, {})
    if isinstance(modality_cfg, dict):
        params.update(deepcopy(modality_cfg.get("params", {})))
    return params

def build_encoder(
    modality: str,
    config: dict[str, Any],
    output_dim: int | None = None,
) -> tuple[nn.Module, str]:
    model_cfg = _get_model_cfg(config)
    modality = modality.lower()
    feature_dim = int(output_dim or model_cfg.get("feature_dim", 256))
    encoder_name = get_modality_encoder_name(config, modality)
    params = _encoder_specific_cfg(model_cfg, encoder_name, modality)

    if modality == "audio":
        in_channels = int(config.get("audio", {}).get("n_mels", 64))
    elif modality == "accel":
        in_channels = int(config.get("accel", {}).get("channels", 4))
    elif modality in {"gelsight", "visual"}:
        in_channels = 3
    else:
        raise ValueError(f"unknown modality: {modality}")

    if encoder_name == "tcn":
        encoder = TCNEncoder(
            in_channels=in_channels,
            hidden_channels=params.get("hidden_channels", [64, 128, 256]),
            output_dim=feature_dim,
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.2)),
            dilations=params.get("dilations", [1, 2, 4, 8]),
        )
    elif encoder_name == "lstm":
        encoder = LSTMEncoder(
            in_channels=in_channels,
            hidden_size=int(params.get("hidden_size", 128)),
            num_layers=int(params.get("num_layers", 2)),
            output_dim=feature_dim,
            dropout=float(params.get("dropout", 0.2)),
            bidirectional=bool(params.get("bidirectional", True)),
        )
    elif encoder_name in IMAGE_ENCODERS:
        encoder = ImageEncoder(
            architecture=encoder_name,
            output_dim=feature_dim,
            pretrained=bool(params.get("pretrained", True)),
            dropout=float(params.get("dropout", 0.0)),
        )
    else:
        raise ValueError(f"unsupported encoder: {encoder_name}")

    return encoder, encoder_name
