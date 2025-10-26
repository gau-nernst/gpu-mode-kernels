import argparse
import dataclasses
import math
import multiprocessing as mp
import os
import time
from multiprocessing.pool import Pool
from typing import Any

import torch
import torch.distributed as dist
from reference import check_implementation, generate_input
from submission_v9 import custom_kernel
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


def _clone_data(data, rank: int):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(_clone_data(x, rank) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x, rank) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v, rank) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        device = f"cuda:{rank}"
        return data.clone().to(device)
    else:
        return data


def _run_distributed_test(test: dict, rank: int):
    """
    Runs a single test case. Do not call directly
    """
    world_size = test["world_size"]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group(
        "nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )
    try:
        data = generate_input(**test, rank=rank)
        torch.cuda.synchronize()
        submission_output = custom_kernel(_clone_data(data, rank))
        torch.cuda.synchronize()
        return check_implementation(data, submission_output)
    finally:
        dist.destroy_process_group()


def run_multi_gpu_test(pool: Pool, test: dict, world_size: int):
    """
    Runs a single test in another process.
    """
    rets = []
    # world_size is a mandatory argument for multi-gpu tests
    for i in range(world_size):
        rets.append(pool.apply_async(_run_distributed_test, args=(test, i)))
    # 60 seconds should be more than enough, we want tests to be fast
    # rets = [el.get(60) for el in rets]
    rets = [el.get(120) for el in rets]

    correct = all(ret[0] for ret in rets)
    error_messages = str.join("\n", [f"rank {rank} - {ret[1]}" for rank, ret in enumerate(rets) if not ret[0]])
    return correct, error_messages


def run_testing(pool: Pool, test_cases: list[dict]):
    print("test-count", len(test_cases))
    for idx, test in enumerate(test_cases):
        good, message = run_multi_gpu_test(pool, test, test["world_size"])
        if not good:
            print(f"test.{idx}.status", "fail")
            print(f"test.{idx}.error", message)
            break

        print(f"test.{idx}.status", "pass")
        if message:
            print(f"test.{idx}.message", message)


def _run_distributed_benchmark(
    test: dict, rank: int, recheck: bool, max_repeats: int, max_time_ns: float
) -> Stats | Any:
    """
    Runs one distributed benchmark. Do not call directly.
    """
    import torch.distributed as dist

    world_size = test["world_size"]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group(
        "nccl", init_method="env://", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}")
    )

    try:
        durations = []
        # generate input data once
        data = generate_input(**test, rank=rank)
        check_copy = _clone_data(data, rank)

        # first, one obligatory correctness check
        output = custom_kernel(_clone_data(data, rank))
        good, message = check_implementation(check_copy, output)
        if not good:
            return message

        # now, do multiple timing runs with proper distributed synchronization
        bm_start_time = time.perf_counter_ns()
        for i in range(max_repeats):
            error_message = None
            if recheck:
                # ensure we use a different seed for every benchmark
                if "seed" in test:
                    test["seed"] += 13

                data = generate_input(**test, rank=rank)
                check_copy = _clone_data(data, rank)

            # Synchronize all ranks before timing
            clear_l2_cache()
            torch.cuda.synchronize()
            dist.barrier()

            # Use distributed timing - only rank 0 records the overall time
            if rank == 0:
                start_time = time.perf_counter_ns()

            # All ranks execute the kernel
            output = custom_kernel(_clone_data(data, rank))

            # Synchronize all ranks after kernel execution
            torch.cuda.synchronize()
            dist.barrier()

            if rank == 0:
                end_time = time.perf_counter_ns()
                duration = end_time - start_time  # Already in nanoseconds
                durations.append(duration)

            if recheck:
                good, message = check_implementation(check_copy, output)
                if not good:
                    error_message = message

            del output

            has_error = torch.tensor(1 if error_message is not None else 0, dtype=torch.int32, device=f"cuda:{rank}")
            dist.reduce(has_error, 0)
            if has_error.item() > 0:
                return error_message

            # Only rank 0 checks convergence criteria
            if rank == 0 and i > 1:
                total_bm_duration = time.perf_counter_ns() - bm_start_time
                stats = calculate_stats(durations)
                # stop if either
                # a) relative error dips below 0.1%
                # b) we exceed the total time limit for benchmarking the kernel
                # c) we exceed 2 minutes of total wallclock time.
                should_stop = (
                    stats.err / stats.mean < 0.001 or stats.mean * stats.runs > max_time_ns or total_bm_duration > 120e9
                )
            else:
                should_stop = False

            # Broadcast stop decision to all ranks
            stop_tensor = torch.tensor(should_stop, dtype=torch.bool, device=f"cuda:{rank}")
            dist.broadcast(stop_tensor, 0)

            if stop_tensor.item():
                break

        # Only rank 0 returns meaningful stats
        if rank == 0:
            return calculate_stats(durations)
        else:
            # Non-zero ranks return a dummy stats object
            return Stats(runs=len(durations), mean=0.0, std=0.0, err=0.0, best=0.0, worst=0.0)

    finally:
        dist.destroy_process_group()


def run_multi_gpu_benchmark(
    pool: Pool, test: dict, recheck: bool, max_repeats: int, max_time_ns: float, world_size: int
):
    """
    Runs a multi-GPU benchmark across all ranks.
    """
    rets = []
    for i in range(world_size):
        rets.append(
            pool.apply_async(
                _run_distributed_benchmark,
                args=(test, i, recheck, max_repeats, max_time_ns),
            )
        )

    # 120 seconds for benchmarking + we run a pre-benchmark test and want to leave some slack
    rets = [el.get(timeout=180) for el in rets]

    # For multi-GPU benchmarking, only rank 0 has meaningful stats
    failed_ranks = []
    rank_0_result = None

    for rank, ret in enumerate(rets):
        if isinstance(ret, Stats):
            if rank == 0:
                rank_0_result = ret
        else:
            # ret is an error message
            failed_ranks.append((rank, ret))

    if failed_ranks:
        error_messages = str.join("\n", [f"rank {rank} - {msg}" for rank, msg in failed_ranks])
        return error_messages
    else:
        return rank_0_result if rank_0_result else "No stats returned from rank 0"


def run_benchmarking(pool: Pool, tests: list[dict]):
    """
    Executes benchmarking code for a CUDA Kernel and logs runtimes.

    @param logger: A PopcornOutput object used for logging benchmark results.
    @param pool: Process on which the benchmarks will be launched.
    @param tests: A list of TestCase objects representing the test cases to be benchmarked.
    @return: An integer representing the exit status: 0 if all benchmarks pass, otherwise 112.
    """
    # warm up
    run_multi_gpu_benchmark(pool, tests[0], False, 100, 10e7, tests[0]["world_size"])

    print("benchmark-count", len(tests))
    for idx, test in enumerate(tests):
        result = run_multi_gpu_benchmark(pool, test, False, 100, 10e9, test["world_size"])
        if isinstance(result, Stats):
            for field in dataclasses.fields(Stats):
                print(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
        else:
            print(f"benchmark.{idx}.status", "fail")
            print(f"benchmark.{idx}.error", result)


def main(args: argparse.Namespace):
    import yaml

    task_config = yaml.safe_load(open("task.yml"))
    n_gpus = 8

    mp_context = mp.get_context("spawn")

    with mp_context.Pool(n_gpus) as pool:
        if args.action == "test":
            run_testing(pool, task_config["tests"])
        elif args.action == "benchmark":
            run_benchmarking(pool, task_config["benchmarks"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    args = parser.parse_args()

    main(args)
