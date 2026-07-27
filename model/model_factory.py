import argparse
import sys
from pathlib import Path


_LEGACY_ROOT = Path(__file__).resolve().parents[1] / "zmh_train"


def _ensure_legacy_path():
    legacy_root = str(_LEGACY_ROOT)
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)


def _resolve_swin_variant(params, swin_variant):
    if swin_variant is not None:
        return swin_variant

    variant = params.get("swin_variant", "v1")
    if variant not in {"v1", "v2"}:
        raise ValueError(f"Unsupported swin_variant: {variant}")
    return variant


def build_model(params, swin_variant=None):
    model_type = params.get("model_type", "swin")

    if model_type == "swin":
        variant = _resolve_swin_variant(params, swin_variant)
        if variant == "v2":
            from model.swinunet_v2 import SwinTransformer
        else:
            from model.swinunet import SwinTransformer
        return SwinTransformer(params)

    if model_type == "resnet":
        from model.resnet_baseline import ResNetBaseline
        return ResNetBaseline(params)

    if model_type == "convlstm":
        from model.convlstm_baseline import ConvLSTMBaseline
        return ConvLSTMBaseline(params)

    if model_type == "fno":
        from model.fno_baseline import FNOBaseline
        return FNOBaseline(params)

    if model_type == "legacy_resnet":
        _ensure_legacy_path()
        from model_pre_baselines.resnet.resnet import ResNet

        return ResNet(
            in_channels=params.get("in_channels", params["net_image"]["N_in_channels"]),
            out_channels=params.get("out_channels", params["net_image"]["N_out_channels"]),
            n_blocks=params.get("n_blocks", 12),
            hidden_channels=params.get("hidden_channels", 64),
            dropout=params.get("dropout", 0.1),
            activation=params.get("activation", "leaky"),
            norm=params.get("norm", True),
        )

    if model_type == "afnonet":
        _ensure_legacy_path()
        from model_pre_baselines.afnonet.afnonet import AFNONet

        patch_size = params.get("patch_size", 8)
        num_blocks = params.get("num_blocks", 16)
        afno_params = argparse.Namespace(
            patch_size=patch_size,
            N_in_channels=params.get("in_channels", params["net_image"]["N_in_channels"]),
            N_out_channels=params.get("out_channels", params["net_image"]["N_out_channels"]),
            num_blocks=num_blocks,
        )

        return AFNONet(
            params=afno_params,
            img_size=(params["img_size"][0], params["img_size"][1]),
            patch_size=(patch_size, patch_size),
            embed_dim=params.get("embed_dim", 128),
            depth=params.get("depth", 12),
            num_blocks=num_blocks,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def _forward_legacy_resnet_with_feature(model, x):
    if len(x.shape) == 5:
        x = x.flatten(1, 2)

    x = model.image_proj(x)
    for block in model.blocks:
        x = block(x)

    feature = model.activation(model.norm(x))
    prediction = model.final(feature)
    return feature, prediction


def _forward_afnonet_with_feature(model, x):
    tokens = model.forward_features(x)
    bsz, h_tokens, w_tokens, channels = tokens.shape

    feature = tokens.permute(0, 3, 1, 2).contiguous()

    prediction = model.head(tokens)
    patch_h, patch_w = model.patch_size
    prediction = prediction.view(
        bsz,
        h_tokens,
        w_tokens,
        patch_h,
        patch_w,
        model.out_chans,
    )
    prediction = prediction.permute(0, 5, 1, 3, 2, 4).contiguous()
    prediction = prediction.view(
        bsz,
        model.out_chans,
        h_tokens * patch_h,
        w_tokens * patch_w,
    )
    return feature, prediction


def forward_with_feature(model, params, x):
    model_type = params.get("model_type", "swin")

    if model_type == "legacy_resnet":
        return _forward_legacy_resnet_with_feature(model, x)

    if model_type == "afnonet":
        return _forward_afnonet_with_feature(model, x)

    return extract_feature_and_prediction(model(x))


def get_feature_channels(params):
    model_type = params.get("model_type", "swin")

    if model_type == "swin":
        variant = _resolve_swin_variant(params, None)
        embed_dim = params["net_image"].get("embed_dim", 96)
        return embed_dim if variant == "v2" else embed_dim * 2

    if model_type == "resnet":
        return params.get("resnet_base_channels", 32) * 6

    if model_type == "legacy_resnet":
        return params.get("hidden_channels", 64) * 2

    if model_type == "fno":
        return params.get("fno_width", 48)

    if model_type == "afnonet":
        return params.get("embed_dim", 128)

    raise ValueError(
        f"Feature channel inference is not implemented for model_type={model_type}"
    )


def extract_feature_and_prediction(model_output):
    if isinstance(model_output, tuple):
        if len(model_output) >= 2:
            return model_output[0], model_output[1]
        if len(model_output) == 1:
            return None, model_output[0]
        raise ValueError("Model returned an empty tuple.")
    return None, model_output


def extract_prediction(model_output):
    _, prediction = extract_feature_and_prediction(model_output)
    return prediction


def describe_model_type(model_type):
    names = {
        "swin": "SwinTransformer",
        "resnet": "ResNet Baseline",
        "legacy_resnet": "Legacy ResNet Baseline",
        "convlstm": "ConvLSTM Baseline",
        "fno": "FNO Baseline",
        "afnonet": "AFNONet Baseline",
    }
    return names.get(model_type, model_type)
