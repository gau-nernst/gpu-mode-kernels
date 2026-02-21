import dataclasses
import math
import multiprocessing as mp
import sys
import time
from multiprocessing.pool import Pool
from typing import Any

import torch.cuda
from cutlass.cute.nvgpu.common import OpError
from reference import check_implementation, generate_input
from submission_v0 import custom_kernel
from utils import clear_l2_cache


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def calculate_stats(durations: list[int]):
    """
    Calculate statistical data from a list of durations.

    @param durations: A list of durations in nanoseconds.
    @return: A Stats object containing the number of runs, mean, standard deviation, error, best, and worst durations.
    """
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
    """
    Recursively goes through data and clones all tensors.
    """
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
    data = generate_input(**test)
    torch.cuda.synchronize()
    try:
        submission_output = custom_kernel(_clone_data(data))

    except OpError as E:
        print(f"Encountered {E}", file=sys.stderr)
        return False, str(E)
    torch.cuda.synchronize()
    return check_implementation(data, submission_output)


def run_single_test(pool: Pool, test: dict):
    return pool.apply(_run_single_test, (test,))


def run_testing(pool: Pool, tests: list[dict]):
    # Step 1: Compile kernel once before running tests
    # compile_success, compile_error = pool.apply(_compile_kernel_once)
    # if not compile_success:
    #     return 112

    # Step 2: Run all tests with compiled kernel
    print("test-count", len(tests))
    for idx, test in enumerate(tests):
        good, message = run_single_test(pool, test)

        if not good:
            print(f"test.{idx}.status", "fail")
            print(f"test.{idx}.error", message)

        else:
            print(f"test.{idx}.status", "pass")
            if message:
                print(f"test.{idx}.message", message)


def _run_single_benchmark(test: dict, recheck: bool, max_repeats: int, max_time_ns: float) -> Stats | Any:
    durations = []
    # generate input data once
    data = generate_input(**test)
    check_copy = _clone_data(data)

    #  first, one obligatory correctness check
    try:
        output = custom_kernel(_clone_data(data))
    except OpError as E:
        return f"Encountered {E}"
    good, message = check_implementation(check_copy, output)
    if not good:
        return message

    # now, do multiple timing runs without further correctness testing
    # there is an upper bound of 200 runs, and a lower bound of 3 runs;
    # otherwise, we repeat until we either measure at least 10 full seconds,
    # or the relative error of the mean is below 1%.

    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        if recheck:
            # ensure we use a different seed for every benchmark
            if "seed" in test:
                test["seed"] += 13

            data = generate_input(**test)
            check_copy = _clone_data(data)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        clear_l2_cache()

        start_event.record()
        output = custom_kernel(data)
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event) * 1e6  # Convert ms to ns

        if recheck:
            good, message = check_implementation(check_copy, output)
            if not good:
                return message

        del output
        durations.append(duration)

        if i > 1:
            total_bm_duration = time.perf_counter_ns() - bm_start_time
            stats = calculate_stats(durations)
            # stop if either
            # a) relative error dips below 0.1%
            # b) we exceed the total time limit for benchmarking the kernel
            # c) we exceed 2 minutes of total wallclock time.
            if stats.err / stats.mean < 0.001 or stats.mean * stats.runs > max_time_ns or total_bm_duration > 120e9:
                break

    return calculate_stats(durations)


def run_single_benchmark(
    pool: Pool,
    test: dict,
    recheck: bool,
    max_repeats: int,
    max_time_ns: float,
):
    return pool.apply(_run_single_benchmark, (test, recheck, max_repeats, max_time_ns))


def run_benchmarking(pool: Pool, tests: list[dict]):
    # Step 2: Warm up with compiled kernel
    run_single_benchmark(pool, tests[0], False, 200, 10e7)

    # Step 3: Run benchmarks (compilation time excluded)
    print("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        result = run_single_benchmark(pool, test, False, 200, 10e9)
        # result = run_single_benchmark(pool, test, True, 200, 30e9)  # leaderboard
        if isinstance(result, Stats):
            for field in dataclasses.fields(Stats):
                print(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
        else:
            print(f"benchmark.{idx}.status", "fail")
            print(f"benchmark.{idx}.error", result)


def main(args):
    import yaml

    task_config = yaml.safe_load(open("task.yml"))
    mp_context = mp.get_context("spawn")

    with mp_context.Pool(1) as pool:
        if args.action == "test":
            run_testing(pool, task_config["tests"])
        elif args.action == "benchmark":
            run_benchmarking(pool, task_config["benchmarks"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    args = parser.parse_args()

    main(args)
