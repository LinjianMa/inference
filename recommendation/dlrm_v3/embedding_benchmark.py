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
PyTorch Embedding Lookup Benchmark.
Compares performance between buck2 and conda environments for embedding operations.
"""

import argparse
import platform
import sys
import time
from typing import Dict, Union

import torch


def get_environment_info() -> Dict[str, str]:
    """Collect environment information."""
    info = {
        "python_version": sys.version.split()[0],
        "python_path": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["cudnn_version"] = str(torch.backends.cudnn.version())
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = str(torch.cuda.device_count())

    # Check for fbgemm_gpu
    try:
        import fbgemm_gpu  # noqa: F401

        # pyre-ignore[16]: Module may not have __version__
        info["fbgemm_gpu_version"] = getattr(
            fbgemm_gpu, "__version__", "installed (version unknown)"
        )
    except ImportError:
        info["fbgemm_gpu_version"] = "Not installed"

    # Check for torchrec
    try:
        import torchrec

        # pyre-ignore[16]: Module may not have __version__
        info["torchrec_version"] = getattr(
            torchrec, "__version__", "installed (version unknown)"
        )
    except ImportError:
        info["torchrec_version"] = "Not installed"

    return info


def benchmark_embedding_bag(
    num_embeddings: int,
    embedding_dim: int,
    batch_size: int,
    bag_size: int,
    num_iterations: int,
    device: str,
    mode: str = "sum",
) -> Dict[str, float]:
    """Benchmark torch.nn.EmbeddingBag lookup."""
    embedding = torch.nn.EmbeddingBag(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        mode=mode,
        sparse=False,
    ).to(device)

    # Generate random indices
    indices = torch.randint(0, num_embeddings, (batch_size * bag_size,), device=device)
    offsets = torch.arange(0, batch_size * bag_size + 1, bag_size, device=device)[
        :-1
    ].contiguous()

    # Warmup
    for _ in range(10):
        _ = embedding(indices, offsets)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = embedding(indices, offsets)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    lookups_per_sec = (batch_size * num_iterations) / elapsed
    return {
        "elapsed_sec": elapsed,
        "avg_ms": (elapsed / num_iterations) * 1000,
        "lookups_per_sec": lookups_per_sec,
    }


def benchmark_embedding(
    num_embeddings: int,
    embedding_dim: int,
    batch_size: int,
    num_iterations: int,
    device: str,
) -> Dict[str, float]:
    """Benchmark torch.nn.Embedding lookup."""
    embedding = torch.nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    ).to(device)

    # Generate random indices
    indices = torch.randint(0, num_embeddings, (batch_size,), device=device)

    # Warmup
    for _ in range(10):
        _ = embedding(indices)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = embedding(indices)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    lookups_per_sec = (batch_size * num_iterations) / elapsed
    return {
        "elapsed_sec": elapsed,
        "avg_ms": (elapsed / num_iterations) * 1000,
        "lookups_per_sec": lookups_per_sec,
    }


def benchmark_fbgemm_jagged_embedding(
    num_embeddings: int,
    embedding_dim: int,
    batch_size: int,
    bag_size: int,
    num_iterations: int,
    device: str,
) -> Dict[str, Union[float, str]]:
    """Benchmark FBGEMM GPU jagged tensor embedding lookup if available."""
    try:
        import fbgemm_gpu  # noqa: F401
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            EmbeddingLocation,
            SplitTableBatchedEmbeddingBagsCodegen,
        )
    except ImportError:
        return {"error": "fbgemm_gpu not available"}

    if device != "cuda":
        return {"error": "FBGEMM requires CUDA"}

    try:
        # Create FBGEMM embedding table
        embedding_specs = [
            (
                num_embeddings,
                embedding_dim,
                EmbeddingLocation.DEVICE,
                ComputeDevice.CUDA,
            )
        ]
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=embedding_specs,
        )

        # Generate random indices with variable bag sizes
        indices = torch.randint(
            0, num_embeddings, (batch_size * bag_size,), device=device, dtype=torch.long
        )
        offsets = (
            torch.arange(0, batch_size + 1, device=device, dtype=torch.long) * bag_size
        )

        # Warmup
        for _ in range(10):
            _ = emb(indices, offsets)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = emb(indices, offsets)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        lookups_per_sec = (batch_size * num_iterations) / elapsed
        return {
            "elapsed_sec": elapsed,
            "avg_ms": (elapsed / num_iterations) * 1000,
            "lookups_per_sec": lookups_per_sec,
        }
    except Exception as e:
        return {"error": str(e)}


def benchmark_torchrec_embedding(
    num_embeddings: int,
    embedding_dim: int,
    batch_size: int,
    bag_size: int,
    num_iterations: int,
    device: str,
) -> Dict[str, Union[float, str]]:
    """Benchmark TorchRec EmbeddingBagCollection if available."""
    try:
        from torchrec import (
            EmbeddingBagCollection,
            EmbeddingBagConfig,
            KeyedJaggedTensor,
        )
    except ImportError:
        return {"error": "torchrec not available"}

    try:
        # Create TorchRec embedding
        ebc = EmbeddingBagCollection(
            device=torch.device(device),
            tables=[
                EmbeddingBagConfig(
                    name="table_0",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    feature_names=["feature_0"],
                )
            ],
        )

        # Create KeyedJaggedTensor input
        values = torch.randint(
            0, num_embeddings, (batch_size * bag_size,), device=device
        )
        lengths = torch.full((batch_size,), bag_size, device=device)
        kjt = KeyedJaggedTensor(
            keys=["feature_0"],
            values=values,
            lengths=lengths,
        )

        # Warmup
        for _ in range(10):
            _ = ebc(kjt)
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = ebc(kjt)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        lookups_per_sec = (batch_size * num_iterations) / elapsed
        return {
            "elapsed_sec": elapsed,
            "avg_ms": (elapsed / num_iterations) * 1000,
            "lookups_per_sec": lookups_per_sec,
        }
    except Exception as e:
        return {"error": str(e)}


def run_benchmarks(
    num_embeddings: int,
    embedding_dim: int,
    batch_size: int,
    bag_size: int,
    num_iterations: int,
    device: str,
) -> Dict[str, Dict[str, float]]:
    """Run all embedding benchmarks."""
    results = {}

    print(f"\n{'='*70}")
    print("Configuration:")
    print(f"  num_embeddings: {num_embeddings:,}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  batch_size: {batch_size}")
    print(f"  bag_size: {bag_size}")
    print(f"  num_iterations: {num_iterations}")
    print(f"  device: {device}")
    print(f"{'='*70}")

    # torch.nn.Embedding
    print("\n[1/4] Benchmarking torch.nn.Embedding...")
    results["nn.Embedding"] = benchmark_embedding(
        num_embeddings, embedding_dim, batch_size, num_iterations, device
    )
    if "error" not in results["nn.Embedding"]:
        print(f"      avg: {results['nn.Embedding']['avg_ms']:.4f} ms/iter")
        print(
            f"      throughput: {results['nn.Embedding']['lookups_per_sec']:,.0f} lookups/sec"
        )

    # torch.nn.EmbeddingBag
    print("\n[2/4] Benchmarking torch.nn.EmbeddingBag...")
    results["nn.EmbeddingBag"] = benchmark_embedding_bag(
        num_embeddings, embedding_dim, batch_size, bag_size, num_iterations, device
    )
    if "error" not in results["nn.EmbeddingBag"]:
        print(f"      avg: {results['nn.EmbeddingBag']['avg_ms']:.4f} ms/iter")
        print(
            f"      throughput: {results['nn.EmbeddingBag']['lookups_per_sec']:,.0f} lookups/sec"
        )

    # FBGEMM
    print("\n[3/4] Benchmarking FBGEMM SplitTableBatchedEmbeddingBags...")
    results["fbgemm"] = benchmark_fbgemm_jagged_embedding(
        num_embeddings, embedding_dim, batch_size, bag_size, num_iterations, device
    )
    if "error" in results["fbgemm"]:
        print(f"      Skipped: {results['fbgemm']['error']}")
    else:
        print(f"      avg: {results['fbgemm']['avg_ms']:.4f} ms/iter")
        print(
            f"      throughput: {results['fbgemm']['lookups_per_sec']:,.0f} lookups/sec"
        )

    # TorchRec
    print("\n[4/4] Benchmarking TorchRec EmbeddingBagCollection...")
    results["torchrec"] = benchmark_torchrec_embedding(
        num_embeddings, embedding_dim, batch_size, bag_size, num_iterations, device
    )
    if "error" in results["torchrec"]:
        print(f"      Skipped: {results['torchrec']['error']}")
    else:
        print(f"      avg: {results['torchrec']['avg_ms']:.4f} ms/iter")
        print(
            f"      throughput: {results['torchrec']['lookups_per_sec']:,.0f} lookups/sec"
        )

    return results


def print_summary(results: Dict[str, Dict[str, float]]) -> None:
    """Print summary table."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Operator':<35} {'Avg (ms)':<15} {'Throughput (lookups/s)':<20}")
    print("-" * 70)

    for name, result in results.items():
        if "error" in result:
            print(f"{name:<35} {'N/A':<15} {result['error']:<20}")
        else:
            print(
                f"{name:<35} {result['avg_ms']:<15.4f} {result['lookups_per_sec']:<20,.0f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch Embedding Lookup Benchmark")
    parser.add_argument(
        "--num-embeddings",
        type=int,
        default=1_000_000,
        help="Number of embeddings in the table",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for lookups",
    )
    parser.add_argument(
        "--bag-size",
        type=int,
        default=32,
        help="Number of indices per bag (for EmbeddingBag)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run benchmarks on",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PyTorch Embedding Lookup Benchmark")
    print("=" * 70)

    # Print environment info
    info = get_environment_info()
    print("\nEnvironment:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Run benchmarks
    results = run_benchmarks(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        bag_size=args.bag_size,
        num_iterations=args.num_iterations,
        device=args.device,
    )

    # Print summary
    print_summary(results)

    print(f"\n{'='*70}")
    print("Benchmark complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
