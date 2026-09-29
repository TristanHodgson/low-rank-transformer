import os
import time

import torch
import torch.cuda as cuda
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from modules.model import Model


app = FastAPI(title="Low-Rank Transformer Inference API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}


class InferenceRequest(BaseModel):
    encrypted_tokens: List[List[int]]  # Shape: (batch_size, seq_len)


def replace_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """
    Replace a nested module by its dotted name.

    Examples:
        blocks.0.sa.q_proj
        blocks.0.ffwd.net.0
    """
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)

    # Sequential/ModuleList children are named "0", "1", etc.
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def adapt_model_to_checkpoint(
    model: Model,
    state_dict: dict,
) -> int:
    """
    Modify `model` so its module structure matches a factorised checkpoint.

    A normal Linear layer:

        Linear(in_features, out_features)

    has checkpoint keys such as:

        blocks.0.sa.q_proj.weight
        blocks.0.sa.q_proj.bias

    After low-rank factorisation it becomes:

        Sequential(
            Linear(in_features, rank, bias=False),
            Linear(rank, out_features, bias=True),
        )

    with checkpoint keys such as:

        blocks.0.sa.q_proj.0.weight
        blocks.0.sa.q_proj.1.weight
        blocks.0.sa.q_proj.1.bias

    Crucially, this function DOES NOT reconstruct the original dense
    weight matrix. The two smaller Linear layers remain separate during
    inference, so low-rank models actually execute two smaller GEMMs.

    Returns:
        Number of Linear modules converted to low-rank factorisations.
    """
    factorised_count = 0

    # Make a fixed list because we modify the model while iterating.
    linear_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    for name, module in linear_modules:
        first_weight_key = f"{name}.0.weight"
        second_weight_key = f"{name}.1.weight"

        # If these keys do not exist, this layer was not factorised.
        if (
            first_weight_key not in state_dict
            or second_weight_key not in state_dict
        ):
            continue

        first_weight = state_dict[first_weight_key]
        second_weight = state_dict[second_weight_key]

        # nn.Linear stores weights as:
        #     (out_features, in_features)
        #
        # First factor:
        #     Linear(original_in_features, rank)
        #
        # so:
        #     first_weight.shape == (rank, original_in_features)
        rank = first_weight.shape[0]

        expected_first_shape = (rank, module.in_features)
        expected_second_shape = (module.out_features, rank)

        if tuple(first_weight.shape) != expected_first_shape:
            raise RuntimeError(
                f"Invalid low-rank checkpoint shape for '{name}'. "
                f"{first_weight_key} has shape "
                f"{tuple(first_weight.shape)}, expected "
                f"{expected_first_shape}."
            )

        if tuple(second_weight.shape) != expected_second_shape:
            raise RuntimeError(
                f"Invalid low-rank checkpoint shape for '{name}'. "
                f"{second_weight_key} has shape "
                f"{tuple(second_weight.shape)}, expected "
                f"{expected_second_shape}."
            )

        first_bias_key = f"{name}.0.bias"
        second_bias_key = f"{name}.1.bias"

        first_has_bias = first_bias_key in state_dict
        second_has_bias = second_bias_key in state_dict

        factorised_layer = nn.Sequential(
            nn.Linear(
                module.in_features,
                rank,
                bias=first_has_bias,
            ),
            nn.Linear(
                rank,
                module.out_features,
                bias=second_has_bias,
            ),
        )

        replace_module(model, name, factorised_layer)
        factorised_count += 1

    return factorised_count


def load_transformer_checkpoint(weights_path: str) -> Model:
    """
    Instantiate the transformer architecture and load either a normal
    full-rank checkpoint or a genuinely factorised low-rank checkpoint.
    """
    # Load onto CPU first. This allows us to inspect the checkpoint and
    # construct the correct architecture before moving everything to CUDA.
    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )

    model = Model(
        vocab_size=32,
        seq_len=32,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_layers=12,
    )

    factorised_count = adapt_model_to_checkpoint(
        model,
        state_dict,
    )

    # Keep strict=True. If the saved architecture does not match the model
    # we want startup to fail rather than silently benchmark bad weights.
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    if factorised_count > 0:
        print(
            f"[INIT] Loaded factorised checkpoint with "
            f"{factorised_count} low-rank Linear layers."
        )
    else:
        print("[INIT] Loaded full-rank checkpoint.")

    return model


@app.on_event("startup")
async def startup_event():
    weights_dir = os.environ.get("WEIGHTS_DIR", "/app/weights")

    full_path = os.path.join(weights_dir, "full_rank.pth")
    comp_path = os.path.join(weights_dir, "compressed.pth")

    if os.path.exists(full_path):
        models["full"] = load_transformer_checkpoint(full_path)
        print(f"[INIT] Loaded Full-Rank model from {full_path}")
    else:
        print(f"[WARNING] Full-Rank model not found at {full_path}")

    if os.path.exists(comp_path):
        models["compressed"] = load_transformer_checkpoint(comp_path)
        print(f"[INIT] Loaded Compressed model from {comp_path}")
    else:
        print(f"[WARNING] Compressed model not found at {comp_path}")

    # CUDA warmup.
    #
    # This is deliberately outside the measured request timings so initial
    # CUDA kernel setup does not contaminate throughput measurements.
    if models:
        dummy_input = torch.randint(
            0,
            32,
            (32, 32),
            device=device,
        )

        with torch.inference_mode():
            for model in models.values():
                for _ in range(10):
                    _ = model(dummy_input)

        if device.type == "cuda":
            cuda.synchronize()

        print("[INIT] GPU Warmup Complete.")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_models": list(models.keys()),
        "device": str(device),
    }


@app.post("/predict/{model_name}")
async def predict(model_name: str, req: InferenceRequest):
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not loaded.",
        )

    model = models[model_name]

    try:
        input_tensor = torch.tensor(
            req.encrypted_tokens,
            dtype=torch.long,
            device=device,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid encrypted_tokens: {exc}",
        ) from exc

    if input_tensor.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "encrypted_tokens must have shape "
                "(batch_size, sequence_length)."
            ),
        )

    batch_size, seq_len = input_tensor.shape

    if seq_len != 32:
        raise HTTPException(
            status_code=400,
            detail=f"Expected sequence length 32, received {seq_len}.",
        )

    if torch.any((input_tensor < 0) | (input_tensor >= 32)):
        raise HTTPException(
            status_code=400,
            detail="All token values must be between 0 and 31.",
        )

    if device.type == "cuda":
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)

        start_event.record()

        with torch.inference_mode():
            logits = model(input_tensor)

        end_event.record()

        # Required before reading CUDA event timing.
        cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)

    else:
        start_time = time.perf_counter()

        with torch.inference_mode():
            logits = model(input_tensor)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

    predictions = torch.argmax(
        logits,
        dim=-1,
    ).cpu().tolist()

    total_tokens = batch_size * seq_len

    throughput_tok_sec = (
        total_tokens / (latency_ms / 1000.0)
        if latency_ms > 0
        else 0.0
    )

    return {
        "model": model_name,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "predictions": predictions,
        "latency_ms": round(latency_ms, 4),
        "throughput_tokens_per_sec": round(
            throughput_tok_sec,
            2,
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.fast-api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
    )