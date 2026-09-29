import asyncio
import aiohttp
import time
import argparse
import random
import statistics

def generate_payload(batch_size: int = 128, seq_len: int = 32):
    return {
        "encrypted_tokens": [
            [random.randint(0, 31) for _ in range(seq_len)]
            for _ in range(batch_size)
        ]
    }

async def send_request(session, url, payload):
    start = time.perf_counter()
    async with session.post(url, json=payload) as response:
        await response.json()
        latency = (time.perf_counter() - start) * 1000.0
        return response.status, latency

async def run_benchmark(base_url: str, model_name: str, total_requests: int, concurrency: int, batch_size: int):
    target_url = f"{base_url.rstrip('/')}/predict/{model_name}"
    payload = generate_payload(batch_size=batch_size)
    
    print(f"\n--- Benchmarking Endpoint: {target_url} ---")
    print(f"Parameters: Total Requests={total_requests}, Concurrency={concurrency}, Batch Size={batch_size}")
    
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        await send_request(session, target_url, payload)
        
        start_time = time.perf_counter()
        tasks = [send_request(session, target_url, payload) for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)
        total_wall_time = time.perf_counter() - start_time

    latencies = [lat for status, lat in results if status == 200]
    failed = total_requests - len(latencies)
    
    total_tokens = len(latencies) * batch_size * 32
    rps = len(latencies) / total_wall_time
    tokens_per_sec = total_tokens / total_wall_time
    
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

    print(f"Wall Clock Time:     {total_wall_time:.2f} s")
    print(f"Successful Requests: {len(latencies)} (Failed: {failed})")
    print(f"Throughput (Req/s):  {rps:.2f} req/s")
    print(f"Throughput (Tok/s):  {tokens_per_sec:,.0f} tokens/s")
    print(f"Latency P50:         {p50:.2f} ms")
    print(f"Latency P95:         {p95:.2f} ms")
    print(f"Latency P99:         {p99:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Throughput Benchmark")
    parser.add_argument("--url", type=str, required=True, help="Base API URL (e.g. http://<ALB-DNS>)")
    parser.add_argument("--model", type=str, default="full", choices=["full", "compressed"])
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.model, args.requests, args.concurrency, args.batch_size))