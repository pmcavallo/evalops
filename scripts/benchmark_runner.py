import asyncio
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from evalops import EvalRunner, EvalDataset, EvalCase

async def mock_target(input: str) -> str:
    # Simulate I/O bound work
    await asyncio.sleep(0.1)
    return f"Processed: {input}"

async def run_benchmark():
    dataset = EvalDataset(
        name="benchmark",
        cases=[EvalCase(input=f"input_{i}") for i in range(20)]
    )

    runner = EvalRunner(enable_observability=False) # Disable logging/metrics to focus on execution time

    print(f"Running benchmark with {len(dataset)} cases...")
    start_time = time.perf_counter()
    result = await runner.evaluate(dataset, mock_target)
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Total time: {duration:.4f} seconds")
    print(f"Average time per case: {duration/len(dataset):.4f} seconds")
    return duration

if __name__ == "__main__":
    asyncio.run(run_benchmark())
