import os
import time
import torch
import torch.cuda as cuda
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from modules.model import Model

app = FastAPI(title="Low-Rank Transformer Inference API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}

class InferenceRequest(BaseModel):
    encrypted_tokens: List[List[int]]  # Shape: (batch_size, seq_len)

def load_transformer_checkpoint(weights_path: str) -> Model:
    """Instantiates model architecture and loads weights state_dict."""
    model = Model(
        vocab_size=32,
        seq_len=32,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_layers=12
    ).to(device)
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
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

    # CUDA Warmup
    if models:
        dummy_input = torch.randint(0, 32, (32, 32), device=device)
        with torch.no_grad():
            for m in models.values():
                for _ in range(10):
                    _ = m(dummy_input)
        if device.type == "cuda":
            cuda.synchronize()
        print("[INIT] GPU Warmup Complete.")

@app.get("/health")
async def health():
    return {"status": "ok", "loaded_models": list(models.keys()), "device": str(device)}

@app.post("/predict/{model_name}")
async def predict(model_name: str, req: InferenceRequest):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded.")

    model = models[model_name]
    input_tensor = torch.tensor(req.encrypted_tokens, dtype=torch.long, device=device)
    batch_size, seq_len = input_tensor.shape

    if device.type == "cuda":
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            logits = model(input_tensor)
        end_event.record()
        cuda.synchronize()
        
        latency_ms = start_event.elapsed_time(end_event)
    else:
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = model(input_tensor)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

    predictions = torch.argmax(logits, dim=-1).cpu().tolist()
    total_tokens = batch_size * seq_len
    throughput_tok_sec = total_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

    return {
        "model": model_name,
        "predictions": predictions,
        "latency_ms": round(latency_ms, 2),
        "throughput_tokens_per_sec": round(throughput_tok_sec, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.fast-api:app", host="0.0.0.0", port=8000, workers=1)