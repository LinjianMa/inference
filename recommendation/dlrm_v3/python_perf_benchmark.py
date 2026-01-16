# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple Python performance benchmark comparing different execution patterns.
Tests CPU-bound operations with threading and multiprocessing.
"""

import argparse
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Dict, List, Tuple


def get_python_info() -> Dict[str, str]:
    """Collect Python environment information."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_path": sys.executable,
    }


def cpu_bound_task(n: int) -> float:
    """
    CPU-bound task: compute sum of squares with some math operations.
    This is intentionally not using numpy to test pure Python performance.
    """
    total = 0.0
    for i in range(n):
        total += math.sqrt(i) * math.sin(i) + math.cos(i)
    return total


def memory_access_task(n: int) -> float:
    """
    Memory-intensive task: create and manipulate lists.
    """
    data = list(range(n))
    total = 0.0
    for i in range(len(data)):
        total += data[i] * 0.5
    return total


def simple_loop_task(n: int) -> int:
    """
    Simple loop with integer addition - minimal overhead.
    """
    total = 0
    for i in range(n):
        total += i
    return total


def run_single_threaded(
    task: Callable[[int], float],
    iterations: int,
    num_tasks: int,
) -> Tuple[float, List[float]]:
    """Run tasks sequentially in single thread."""
    results = []
    start = time.perf_counter()
    for _ in range(num_tasks):
        results.append(task(iterations))
    elapsed = time.perf_counter() - start
    return elapsed, results


def run_multithreaded(
    task: Callable[[int], float],
    iterations: int,
    num_tasks: int,
    num_threads: int,
) -> Tuple[float, List[float]]:
    """Run tasks using ThreadPoolExecutor."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(task, [iterations] * num_tasks))
    elapsed = time.perf_counter() - start
    return elapsed, results


def run_multiprocess(
    task: Callable[[int], float],
    iterations: int,
    num_tasks: int,
    num_processes: int,
) -> Tuple[float, List[float]]:
    """Run tasks using ProcessPoolExecutor."""
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(task, [iterations] * num_tasks))
    elapsed = time.perf_counter() - start
    return elapsed, results


def benchmark_task(
    task_name: str,
    task: Callable[[int], float],
    iterations: int,
    num_tasks: int,
    num_workers: int,
) -> Dict[str, float]:
    """Run all benchmark variants for a given task."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {task_name}")
    print(f"  Iterations per task: {iterations:,}")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Number of workers: {num_workers}")
    print(f"{'='*60}")

    # Warmup
    _ = task(1000)

    results = {}

    # Single-threaded
    elapsed, _ = run_single_threaded(task, iterations, num_tasks)
    results["single_thread"] = elapsed
    print(f"  Single-threaded:  {elapsed:.4f}s")

    # Multi-threaded
    elapsed, _ = run_multithreaded(task, iterations, num_tasks, num_workers)
    results["multi_thread"] = elapsed
    print(f"  Multi-threaded:   {elapsed:.4f}s ({num_workers} threads)")

    # Multi-process
    elapsed, _ = run_multiprocess(task, iterations, num_tasks, num_workers)
    results["multi_process"] = elapsed
    print(f"  Multi-process:    {elapsed:.4f}s ({num_workers} processes)")

    # Calculate speedups
    single = results["single_thread"]
    print("\n  Speedups vs single-threaded:")
    print(f"    Multi-threaded: {single/results['multi_thread']:.2f}x")
    print(f"    Multi-process:  {single/results['multi_process']:.2f}x")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python performance benchmark for buck2 vs conda comparison"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1_000_000,
        help="Number of iterations per task",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=8,
        help="Number of parallel tasks to run",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of threads/processes for parallel execution",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Python Performance Benchmark")
    print("=" * 60)

    # Print environment info
    info = get_python_info()
    print("\nEnvironment:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Define tasks to benchmark
    tasks = [
        ("simple_loop (integer addition)", simple_loop_task),
        ("cpu_bound (math operations)", cpu_bound_task),
        ("memory_access (list operations)", memory_access_task),
    ]

    all_results = {}
    for task_name, task_func in tasks:
        results = benchmark_task(
            task_name=task_name,
            task=task_func,
            iterations=args.iterations,
            num_tasks=args.num_tasks,
            num_workers=args.num_workers,
        )
        all_results[task_name] = results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Task':<40} {'Single(s)':<12} {'Thread(s)':<12} {'Process(s)':<12}"
    print(f"\n{header}")
    print("-" * 76)
    for task_name, results in all_results.items():
        short_name = task_name.split(" ")[0]
        print(
            f"{short_name:<40} "
            f"{results['single_thread']:<12.4f} "
            f"{results['multi_thread']:<12.4f} "
            f"{results['multi_process']:<12.4f}"
        )

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
