import dataclasses
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal
import torch
import yaml
from reference import check_implementation, generate_input
from utils import clear_l2_cache_large

try:
    from cutlass._mlir.ir import MLIRError
    from cutlass.cute.nvgpu.common import OpError
except ImportError:
    MLIRError = OpError = None


# https://github.com/gpu-mode/kernelbot/blob/7e37b098/src/runners/modal_runner.py
image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.13")
    .run_commands("ln -sf $(which python) /usr/local/bin/python3")
    .apt_install("git", "gcc-13", "g++-13", "clang-18")
    .uv_pip_install(
        "ninja~=1.11", "wheel~=0.45", "requests~=2.32.4", "packaging~=25.0", "numpy~=2.3", "pytest", "PyYAML"
    )
    .uv_pip_install("torch==2.9.1", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("tinygrad~=0.10")
    .uv_pip_install(
        "nvidia-cupynumeric~=25.3", "nvidia-cutlass-dsl==4.3.5", "cuda-core[cu13]", "cuda-python[all]==13.0"
    )
    .add_local_python_source("reference", "utils", "task", "submission")
)
app = modal.App("kernelbot", image=image)
GPU = "B200"


def _init_worker():
    os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"


NUM_ITERATIONS_PER_BENCHMARK = 15
UNSERIALIZABLE_EXCEPTIONS = (OpError, MLIRError)


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    variance = sum(map(lambda x: (x - avg) ** 2, durations))
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)

    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst))


def _clone_data(data):
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data


def _run_single_test(test: dict):
    from submission import custom_kernel

    data = generate_input(**test)
    torch.cuda.synchronize()
    try:
        submission_output = custom_kernel(_clone_data(data))
    except UNSERIALIZABLE_EXCEPTIONS as E:
        print(f"Encountered {E}", file=sys.stderr)
        return False, str(E)
    torch.cuda.synchronize()
    return check_implementation(data, submission_output)


@app.function(gpu=GPU)
def run_testing(tests: list[dict]):
    with mp.get_context("spawn").Pool(1, initializer=_init_worker) as pool:
        print("test-count", len(tests))
        for idx, test in enumerate(tests):
            good, message = pool.apply(_run_single_test, (test,))

            if not good:
                print(f"test.{idx}.status", "fail")
                print(f"test.{idx}.error", message)
            else:
                print(f"test.{idx}.status", "pass")
                if message:
                    print(f"test.{idx}.message", message)


def _run_single_benchmark(test: dict, recheck: bool, max_repeats: int, max_time_ns: float) -> Stats | Any:
    from submission import custom_kernel

    durations = []
    data_list = []
    # generate input data once

    for i in range(NUM_ITERATIONS_PER_BENCHMARK):
        if "seed" in test:
            test["seed"] += 42
        data = generate_input(**test)
        data_list.append(data)

    check_copy = _clone_data(data_list)

    #  first, one obligatory correctness check
    outputs = []
    try:
        for data in data_list:
            output = custom_kernel(_clone_data(data))
            outputs.append(output)
    except UNSERIALIZABLE_EXCEPTIONS as E:
        return f"Encountered {E}"
    for reference_output, custom_output in zip(check_copy, outputs):
        good, message = check_implementation(reference_output, custom_output)
        if not good:
            return message

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 200 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        torch.cuda.synchronize()

        outputs = []
        clear_l2_cache_large()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for data in data_list:
            output = custom_kernel(data)
            outputs.append(output)
        end_event.record()
        torch.cuda.synchronize()
        duration = (start_event.elapsed_time(end_event) / NUM_ITERATIONS_PER_BENCHMARK) * 1e6  # Convert ms to ns

        if recheck:
            for reference_output, custom_output in zip(check_copy, outputs):
                good, message = check_implementation(reference_output, custom_output)
            if not good:
                return message

        durations.append(duration)

        total_bm_duration = time.perf_counter_ns() - bm_start_time
        if i > 1 and total_bm_duration > 1e8:  # at least 2 runs, and at least 100 ms total time
            stats = calculate_stats(durations)
            # stop if either
            # a) relative error dips below 0.1%
            # b) we exceed the total time limit for benchmarking the kernel
            # c) we exceed 2 minutes of total wallclock time.
            if stats.err / stats.mean < 0.001 or stats.mean * stats.runs > max_time_ns or total_bm_duration > 120e9:
                break

    return calculate_stats(durations)


@app.function(gpu=GPU)
def run_benchmarking(tests: list[dict]):
    with mp.get_context("spawn").Pool(1, initializer=_init_worker) as pool:
        pool.apply(_run_single_benchmark, (tests[0], False, 100, 10e7))

        print("benchmark-count", len(tests))
        for idx, test in enumerate(tests):
            result = pool.apply(_run_single_benchmark, (test, False, 100, 10e9))

            print(f"benchmark.{idx}")
            if isinstance(result, Stats):
                print(f"  mean: {result.mean / 1e3:.4f} us, std: {result.std / 1e3:.4f} us")
                print(f"  fastest: {result.best / 1e3:.4f} us, slowest: {result.worst / 1e3:.4f} us")
            else:
                print("fail", result)


@app.function(gpu=GPU)
def run_leaderboard(tests: list[dict]):
    with mp.get_context("spawn").Pool(1, initializer=_init_worker) as pool:
        # Warmup all test shapes to ensure consistent benchmarking
        for test in tests:
            pool.apply(_run_single_benchmark, (test, False, 50, 5e8))

        print("benchmark-count", len(tests))
        for idx, test in enumerate(tests):
            result = pool.apply(_run_single_benchmark, (test, True, 100, 30e9))

            print(f"benchmark.{idx}")
            if isinstance(result, Stats):
                print(f"  mean: {result.mean / 1e3:.4f} us, std: {result.std / 1e3:.4f} us")
                print(f"  fastest: {result.best / 1e3:.4f} us, slowest: {result.worst / 1e3:.4f} us")
            else:
                print("fail", result)


@app.local_entrypoint()
def main(mode: str):
    task = yaml.safe_load(open(Path(__file__).parent / "task.yml"))

    if mode == "test":
        run_testing.remote(task["tests"])

    elif mode == "benchmark":
        run_benchmarking.remote(task["benchmarks"])

    elif mode == "leaderboard":
        run_leaderboard.remote(task["benchmarks"])

    else:
        raise ValueError
