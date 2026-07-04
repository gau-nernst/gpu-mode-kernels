from pathlib import Path

from modal import App, Image

CURRENT_DIR = Path(__file__).parent


# https://github.com/gpu-mode/kernelbot/blob/e97bee8ee5d257ccf6ea19419daa80723ef49408/src/runners/modal_runner.py
cuda_version = "12.9.1"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
    .run_commands("ln -sf $(which python) /usr/local/bin/python3")
    .apt_install("git", "curl", "gcc-13", "g++-13", "clang-18")
    .uv_pip_install(
        "ninja~=1.11", "wheel~=0.45", "requests~=2.32.4", "packaging~=25.0", "numpy~=2.3", "pytest", "PyYAML"
    )
    # nvidia cuda packages
    .uv_pip_install(
        "nvidia-cupynumeric~=25.3", "nvidia-cutlass-dsl==4.5.2", "cuda-core[cu13]", "cuda-python[all]==13.0"
    )
    # Install torch last so its CUDA/NCCL dependency set wins over broader CUDA Python packages.
    .uv_pip_install("torch==2.12.0")
    .workdir("/workspace")
    .add_local_dir(CURRENT_DIR, remote_path="/workspace", ignore=["*.venv"])
    .add_local_file(CURRENT_DIR / "../../pmpp_v2/utils.py", remote_path="/workspace/utils.py")
)
app = App("qr_v2", image=image)


@app.function(gpu="B200")
def run(mode: str, file: str, case: int | None):
    import dataclasses
    import importlib
    import multiprocessing

    import yaml
    from eval import PopcornOutput, Stats, TestCase, run_benchmarking, run_single_benchmark, run_testing, set_seed

    with open("submission.py", "w") as f:
        f.write(r"""
import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
""")
        f.write(open(file).read())

    if mode == "resource_usage":
        import glob
        import subprocess

        importlib.import_module("submission")
        libraries = glob.glob("/workspace/.build/qr32_codex_ext*.so")
        if not libraries:
            raise RuntimeError("extension library not found")
        subprocess.run(
            ["cuobjdump", "--dump-resource-usage", libraries[0]],
            check=True,
        )
        return 0

    if mode in ("cluster_panel_check", "wide_panel_check"):
        import torch

        implementation = importlib.import_module("submission")
        rows = case or 2048
        width = 512 if mode == "wide_panel_check" else 128
        data = torch.randn((1, rows, width), device="cuda", dtype=torch.float32)
        panel_op = getattr(torch.ops.qr32_codex, f"flag_panel4096x{width}")
        panel, tau, v, v_half = panel_op(data)
        expected_v, _ = implementation._build_v_triton(
            panel,
            tau,
            include_fp16=False,
        )
        q = torch.linalg.householder_product(panel, tau)
        r = torch.triu(panel[:1, :width, :])
        reconstruction = q @ r
        reference_panel, reference_tau = torch.geqrf(data)
        print(f"rows={rows},width={width}")
        print(f"panel_finite={torch.isfinite(panel).all().item()}")
        print(f"tau_finite={torch.isfinite(tau).all().item()}")
        print(f"v_max_diff={(v - expected_v).abs().max().item():.9g}")
        print(f"v_half_max_diff={(v_half.float() - expected_v).abs().max().item():.9g}")
        print(f"reconstruction_max_diff={(reconstruction - data).abs().max().item():.9g}")
        print(f"reconstruction_norm={(reconstruction - data).norm().item():.9g}")
        print(f"tau_max_diff={(tau - reference_tau).abs().max().item():.9g}")
        print(f"panel_max_diff={(panel - reference_panel).abs().max().item():.9g}")
        return 0

    if mode == "layout":
        import torch

        def benchmark_ms(fn, warmup=3, repeats=10):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / repeats

        def column_major_view(value):
            return value.transpose(-2, -1).contiguous().transpose(-2, -1)

        shapes = [
            (2, 4096, 16),
            (2, 4096, 32),
            (2, 4096, 64),
            (2, 4096, 128),
            (2, 4096, 256),
            (2, 4096, 512),
            (2, 512, 512),
            (2, 1024, 1024),
            (2, 2048, 2048),
            (2, 4096, 4096),
        ]
        for batch, rows, cols in shapes:
            row_major = torch.randn(
                (batch, rows, cols),
                device="cuda",
                dtype=torch.float32,
            )
            column_major = column_major_view(row_major)
            assert torch.equal(row_major, column_major)
            repeats = 3 if rows == 4096 and cols == 4096 else 10
            row_ms = benchmark_ms(lambda: torch.geqrf(row_major), repeats=repeats)
            column_ms = benchmark_ms(lambda: torch.geqrf(column_major), repeats=repeats)
            transpose_ms = benchmark_ms(lambda: column_major_view(row_major), repeats=repeats)
            end_to_end_ms = benchmark_ms(
                lambda: torch.geqrf(column_major_view(row_major)),
                repeats=repeats,
            )
            print(
                "layout,"
                f"batch={batch},rows={rows},cols={cols},"
                f"row_stride={row_major.stride()},"
                f"column_stride={column_major.stride()},"
                f"row_ms={row_ms:.6f},column_ms={column_ms:.6f},"
                f"transpose_ms={transpose_ms:.6f},"
                f"end_to_end_ms={end_to_end_ms:.6f}"
            )
            del row_major, column_major
            torch.cuda.empty_cache()
        return 0

    if mode == "trsm_compare":
        import torch

        importlib.import_module("submission")

        def benchmark_ms(fn, warmup=3, repeats=12):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / repeats

        shapes = [
            (640, 96, 416),
            (640, 64, 256),
            (640, 32, 224),
            (60, 48, 976),
            (60, 48, 928),
            (60, 48, 880),
            (60, 56, 824),
            (60, 56, 768),
            (60, 64, 704),
            (60, 72, 632),
            (60, 80, 552),
            (60, 88, 464),
            (60, 104, 360),
            (60, 136, 224),
            (60, 32, 192),
        ]
        for batch, width, rhs_cols in shapes:
            triangular = torch.triu(
                torch.randn(
                    (batch, width, width),
                    device="cuda",
                    dtype=torch.float32,
                )
                * 0.01
            )
            triangular.diagonal(dim1=-2, dim2=-1).fill_(1.0)
            rhs = torch.randn(
                (batch, width, rhs_cols),
                device="cuda",
                dtype=torch.float32,
            )
            identity = torch.eye(width, device="cuda", dtype=torch.float32).expand(batch, -1, -1)
            lower = triangular.transpose(-2, -1)

            reference = torch.linalg.solve_triangular(lower, rhs, upper=False)
            custom_result = rhs.clone()
            torch.ops.qr32_codex.batched_trsm_(triangular, custom_result)
            custom_error = (custom_result - reference).abs().max().item()

            trsm_rhs = rhs.clone()
            trsm_ms = benchmark_ms(lambda: torch.ops.qr32_codex.batched_trsm_(triangular, trsm_rhs))

            def solve_identity_gemm():
                t_transpose = torch.linalg.solve_triangular(lower, identity, upper=False)
                return torch.bmm(t_transpose, rhs)

            def inverse_gemm():
                t_transpose = torch.linalg.inv(triangular).transpose(-2, -1)
                return torch.bmm(t_transpose, rhs)

            solve_gemm_ms = benchmark_ms(solve_identity_gemm)
            inverse_gemm_ms = benchmark_ms(inverse_gemm)
            solve_error = (solve_identity_gemm() - reference).abs().max().item()
            inverse_error = (inverse_gemm() - reference).abs().max().item()
            print(
                "trsm_compare,"
                f"batch={batch},width={width},rhs={rhs_cols},"
                f"trsm_ms={trsm_ms:.6f},"
                f"solve_gemm_ms={solve_gemm_ms:.6f},"
                f"inverse_gemm_ms={inverse_gemm_ms:.6f},"
                f"custom_error={custom_error:.3e},"
                f"solve_error={solve_error:.3e},"
                f"inverse_error={inverse_error:.3e}"
            )
        return 0

    if mode == "qr176_batch_sweep":
        import torch

        implementation = importlib.import_module("submission")

        def benchmark_us(fn, warmup=4, repeats=20):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 1000.0 / repeats

        for batch in (1, 2, 4, 8, 16, 40, 80, 160, 320, 640):
            data = torch.randn(
                (batch, 176, 176),
                device="cuda",
                dtype=torch.float32,
            )
            elapsed_us = benchmark_us(
                lambda: implementation.custom_kernel(data),
                repeats=40 if batch <= 40 else 12,
            )
            print(f"qr176_batch_sweep,batch={batch},us={elapsed_us:.6f}")
        return 0

    if mode == "torch_profile":
        import torch

        implementation = importlib.import_module("submission")
        n = case or 2048
        batches = {32: 20, 176: 40, 352: 40, 512: 640, 1024: 60, 2048: 8, 4096: 2}
        if n not in batches:
            raise ValueError(f"unsupported torch_profile size: {n}")
        data = torch.randn(
            (batches[n], n, n),
            device="cuda",
            dtype=torch.float32,
        )
        for _ in range(3):
            implementation.custom_kernel(data)
        with torch.profiler.profile() as prof:
            implementation.custom_kernel(data)
        prof.export_chrome_trace("/tmp/trace.json.gz")
        return open("/tmp/trace.json.gz", "rb").read()
        return 0

    if mode == "cost_model_512":
        import torch

        implementation = importlib.import_module("submission")
        n = case or 512
        if n not in (512, 1024):
            raise ValueError("cost_model_512 supports QR512 or QR1024")
        batch = {512: 640, 1024: 60}[n]
        if n == 1024:
            torch.set_float32_matmul_precision("high")
        candidate_widths = {
            512: (32, 48, 64, 80, 96, 112, 128),
            1024: (96, 128),
        }[n]
        row_step = 16 if n == 512 else 8
        terminal_sizes = (176, 192, 208) if n == 512 else (128,)
        max_smem_bytes = 227 * 1024

        def benchmark_ms(fn, warmup=2, repeats=3):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            samples.sort()
            return samples[len(samples) // 2]

        def panel_fits(rows, width):
            storage_rows = (rows + 31) // 32 * 32
            if n == 1024:
                return (storage_rows * (width // 2) + width) * 4 + width * 8 <= max_smem_bytes
            return storage_rows * (width + 3) * 4 + 8 <= max_smem_bytes

        source = torch.randn(
            (batch, n, n),
            device="cuda",
            dtype=torch.float32,
        )
        destination = torch.empty_like(source)
        stage_costs = {}
        for rows in range(
            min(terminal_sizes) + row_step,
            n + 1,
            row_step,
        ):
            offset = n - rows
            for width in candidate_widths:
                if rows - width < min(terminal_sizes):
                    continue
                if not panel_fits(rows, width):
                    continue
                panel_input = source[:, offset:, offset : offset + width]
                panel_output = destination[:, offset:, offset : offset + width]
                panel_tau_full = torch.empty(
                    (batch, n),
                    device="cuda",
                    dtype=torch.float32,
                )
                panel_tau = panel_tau_full[:, offset : offset + width]
                v_f32 = source.new_empty(batch, width, rows).transpose(1, 2)
                v_f16 = source.new_empty(batch, width, rows, dtype=torch.float16).transpose(1, 2)

                def factor_panel():
                    torch.ops.codex.panelMN(
                        panel_input,
                        panel_output,
                        panel_tau,
                        v_f32,
                        v_f16,
                    )

                panel_ms = benchmark_ms(factor_panel)
                factor_panel()
                trailing = destination[:, offset:, offset + width :]
                compact_ms = benchmark_ms(
                    lambda panel_output=panel_output, panel_tau=panel_tau, trailing=trailing: (
                        implementation.compact_wy_apply_transpose(
                            panel_output,
                            panel_tau,
                            trailing,
                            trailing,
                            problem_size=n,
                            v_f32=v_f32,
                            v_f16=v_f16,
                        )
                    )
                )
                stage_costs[(rows, width)] = panel_ms + compact_ms
                print(
                    "cost_smem_stage,"
                    f"n={n},rows={rows},width={width},"
                    f"panel_ms={panel_ms:.6f},"
                    f"compact_ms={compact_ms:.6f},"
                    f"total_ms={stage_costs[(rows, width)]:.6f}"
                )

        terminal_costs = {}
        for terminal in terminal_sizes:
            offset = n - terminal
            terminal_input = source[:, offset:, offset:]
            terminal_output = destination[:, offset:, offset:]
            terminal_tau_full = torch.empty(
                (batch, n),
                device="cuda",
                dtype=torch.float32,
            )
            terminal_tau = terminal_tau_full[:, offset : offset + terminal]
            terminal_costs[terminal] = benchmark_ms(
                lambda terminal_input=terminal_input, terminal_output=terminal_output, terminal_tau=terminal_tau: (
                    torch.ops.codex.panelMN(
                        terminal_input,
                        terminal_output,
                        terminal_tau,
                    )
                )
            )
            print(f"cost_smem_terminal,n={n},rows={terminal},total_ms={terminal_costs[terminal]:.6f}")

        top_paths = {terminal: [(cost, (terminal,))] for terminal, cost in terminal_costs.items()}
        for rows in range(
            min(terminal_sizes) + row_step,
            n + 1,
            row_step,
        ):
            candidates = []
            for width in candidate_widths:
                edge = (rows, width)
                next_rows = rows - width
                if edge not in stage_costs or next_rows not in top_paths:
                    continue
                for suffix_cost, suffix in top_paths[next_rows]:
                    candidates.append((stage_costs[edge] + suffix_cost, (width,) + suffix))
            candidates.sort(key=lambda candidate: candidate[0])
            unique = []
            seen = set()
            for candidate in candidates:
                if candidate[1] not in seen:
                    unique.append(candidate)
                    seen.add(candidate[1])
                if len(unique) == 3:
                    break
            if unique:
                top_paths[rows] = unique

        predictions = top_paths[n]
        for rank, (predicted_ms, schedule) in enumerate(predictions):
            print(f"cost_smem_prediction,n={n},rank={rank},predicted_ms={predicted_ms:.6f},schedule={schedule}")

        baseline_schedule = implementation.SHORT_BLOCK_WIDTHS[n]
        baseline_rows = n
        baseline_predicted = 0.0
        for width in baseline_schedule[:-1]:
            baseline_predicted += stage_costs[(baseline_rows, width)]
            baseline_rows -= width
        baseline_predicted += terminal_costs[baseline_schedule[-1]]
        finalists = list(predictions[:2])
        if all(schedule != baseline_schedule for _, schedule in finalists):
            finalists.append((baseline_predicted, baseline_schedule))

        results = []
        for predicted_ms, schedule in finalists:
            measured_ms = benchmark_ms(
                lambda schedule=schedule: implementation.smem_blocked_qr(source, schedule, problem_size=n),
                repeats=7,
            )
            results.append((measured_ms, predicted_ms, schedule))
            print(
                "cost_smem_candidate,"
                f"n={n},predicted_ms={predicted_ms:.6f},"
                f"measured_ms={measured_ms:.6f},"
                f"schedule={schedule}"
            )

        results.sort()
        print(f"cost_smem_summary,n={n},predicted_best={predictions[0][1]},measured_best={results[0][2]}")
        return 0

    if mode == "cost_model":
        import torch

        implementation = importlib.import_module("submission")
        n = case or 4096
        if n not in (2048, 4096):
            raise ValueError("cost_model supports QR2048 or QR4096")
        batch = {2048: 8, 4096: 2}[n]
        widths = (64, 128)
        tail_sizes = (0, 256, 384, 512, 640, 768, 896, 1024)

        def benchmark_ms(fn, warmup=2, repeats=3):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            samples.sort()
            return samples[len(samples) // 2]

        data = torch.randn(
            (batch, n, n),
            device="cuda",
            dtype=torch.float32,
        )

        check_input = data[:, :, :64]
        check_panel, check_tau = implementation.factor_global_panel(
            check_input,
            64,
        )
        check_q = torch.linalg.householder_product(check_panel, check_tau)
        check_r = torch.triu(check_panel[:, :64, :])
        check_residual = (
            (check_q.double() @ check_r.double() - check_input.double()).norm() / check_input.double().norm()
        ).item()
        print(f"cost_panel_check,width=64,residual={check_residual:.6e}")
        if check_residual > 1e-5:
            raise RuntimeError(f"width-64 panel residual is too large: {check_residual}")

        stage_costs = {}
        for rows in range(n, 63, -64):
            offset = n - rows
            for width in widths:
                if width > rows:
                    continue
                panel_input = data[:, offset:, offset : offset + width]
                panel, panel_tau = implementation.factor_global_panel(
                    panel_input,
                    width,
                )
                trailing = data[:, offset:, offset + width :]
                panel_ms = benchmark_ms(
                    lambda panel_input=panel_input, width=width: implementation.factor_global_panel(panel_input, width)
                )
                if trailing.shape[-1]:
                    compact_ms = benchmark_ms(
                        lambda panel=panel, panel_tau=panel_tau, trailing=trailing: (
                            implementation.compact_wy_apply_transpose(
                                panel,
                                panel_tau,
                                trailing,
                                trailing,
                                problem_size=4096,
                            )
                        )
                    )
                else:
                    compact_ms = 0.0
                stage_costs[(rows, width)] = panel_ms + compact_ms
                print(
                    "cost_stage,"
                    f"n={n},rows={rows},width={width},"
                    f"panel_ms={panel_ms:.6f},"
                    f"compact_ms={compact_ms:.6f},"
                    f"total_ms={stage_costs[(rows, width)]:.6f}"
                )

        tail_costs = {0: 0.0}
        for tail_size in tail_sizes[1:]:
            tail_widths = implementation.tall_block_widths(tail_size)
            tail_input = data[:, n - tail_size :, n - tail_size :]
            tail_costs[tail_size] = benchmark_ms(
                lambda tail_input=tail_input, tail_widths=tail_widths: implementation.smem_blocked_qr(
                    tail_input,
                    tail_widths,
                    matmul_precision="high",
                )
            )
            print(f"cost_tail,n={n},tail={tail_size},widths={tail_widths},total_ms={tail_costs[tail_size]:.6f}")

        top_paths = {}
        for rows in range(0, n + 1, 64):
            candidates = []
            if rows in tail_costs:
                candidates.append((tail_costs[rows], (), rows))
            for width in widths:
                next_rows = rows - width
                if next_rows < 0 or next_rows not in top_paths:
                    continue
                edge_cost = stage_costs[(rows, width)]
                for suffix_cost, suffix_widths, tail_size in top_paths[next_rows]:
                    candidates.append(
                        (
                            edge_cost + suffix_cost,
                            (width,) + suffix_widths,
                            tail_size,
                        )
                    )
            candidates.sort(key=lambda candidate: candidate[0])
            unique = []
            seen = set()
            for candidate in candidates:
                key = (candidate[1], candidate[2])
                if key not in seen:
                    unique.append(candidate)
                    seen.add(key)
                if len(unique) == 3:
                    break
            if unique:
                top_paths[rows] = unique

        predicted_paths = top_paths[n]
        for rank, (cost, path_widths, tail_size) in enumerate(predicted_paths):
            print(f"cost_prediction,rank={rank},predicted_ms={cost:.6f},tail={tail_size},widths={path_widths}")

        baseline_tail = {2048: 0, 4096: 512}[n]
        baseline = (
            sum(stage_costs[(rows, 128)] for rows in range(n, baseline_tail, -128)) + tail_costs[baseline_tail],
            (128,) * ((n - baseline_tail) // 128),
            baseline_tail,
        )
        finalists = list(predicted_paths[:2])
        if all((path, tail) != (baseline[1], baseline[2]) for _, path, tail in finalists):
            finalists.append(baseline)

        results = []
        for predicted_ms, path_widths, tail_size in finalists:
            measured_ms = benchmark_ms(
                lambda path_widths=path_widths, tail_size=tail_size: implementation.scheduled_global_qr(
                    data,
                    path_widths,
                    implementation.tall_block_widths(tail_size) if tail_size else (),
                ),
                repeats=7,
            )
            results.append((measured_ms, predicted_ms, path_widths, tail_size))
            print(
                "cost_candidate,"
                f"predicted_ms={predicted_ms:.6f},"
                f"measured_ms={measured_ms:.6f},"
                f"tail={tail_size},widths={path_widths}"
            )

        results.sort()
        print(
            "cost_summary,"
            f"predicted_best={(predicted_paths[0][1], predicted_paths[0][2])},"
            f"measured_best={(results[0][2], results[0][3])}"
        )
        return 0

    if mode == "profile":
        import torch

        implementation = importlib.import_module("submission")
        sizes = [case] if case is not None else [2048, 4096]

        def benchmark_ms(fn, warmup=2, repeats=10):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / repeats

        for n in sizes:
            if n not in (2048, 4096):
                raise ValueError("profile case must be 2048 or 4096")
            batch = {2048: 8, 4096: 2}[n]
            if n == 2048:
                offsets = list(range(0, n - 128, 128))
                panel_widths = [128] * len(offsets)
            else:
                offsets = list(range(0, 2048, 128))
                panel_widths = [128] * len(offsets)
                tail_offset = 2048
                tail_widths = implementation.tall_block_widths(2048)[:-1]
                for width in tail_widths:
                    offsets.append(tail_offset)
                    panel_widths.append(width)
                    tail_offset += width
            sample_count = min(10, len(panel_widths))
            sample_indices = sorted(
                {round(i * (len(panel_widths) - 1) / (sample_count - 1)) for i in range(sample_count)}
            )

            source = torch.randn(
                (batch, n, n),
                device="cuda",
                dtype=torch.float32,
            )
            destination = torch.randn_like(source)
            print(f"profile,n={n},batch={batch},panels={len(panel_widths)}")

            for stage in sample_indices:
                offset = offsets[stage]
                width = panel_widths[stage]
                rows = n - offset
                trailing_cols = rows - width
                panel_input = source[:, offset:, offset : offset + width]
                panel_output = destination[:, offset:, offset : offset + width]
                panel_tau = torch.empty(
                    (batch, width),
                    device="cuda",
                    dtype=torch.float32,
                )

                panel_ms = benchmark_ms(
                    lambda: torch.ops.qr32_codex.panelMN(
                        panel_input,
                        panel_output,
                        panel_tau,
                    )
                )
                torch.ops.qr32_codex.panelMN(
                    panel_input,
                    panel_output,
                    panel_tau,
                )
                trailing = destination[:, offset:, offset + width :]
                compact_ms = benchmark_ms(
                    lambda: implementation.compact_wy_apply_transpose(
                        panel_output,
                        panel_tau,
                        trailing,
                        trailing,
                        problem_size=n,
                    )
                )

                def build_v():
                    v_result = torch.tril(panel_output, diagonal=-1)
                    v_result.diagonal(dim1=-2, dim2=-1).fill_(1.0)
                    zero_result = panel_tau == 0
                    v_result.mul_((~zero_result).unsqueeze(-2))
                    return v_result

                v_ms = benchmark_ms(build_v)
                zero_tau = panel_tau == 0
                v = torch.tril(panel_output, diagonal=-1)
                v.diagonal(dim1=-2, dim2=-1).fill_(1.0)
                v.mul_((~zero_tau).unsqueeze(-2))
                vt = v.transpose(-2, -1)

                def build_t_inv():
                    t_result = torch.triu(vt @ v, diagonal=1)
                    safe_tau = torch.where(
                        zero_tau,
                        torch.ones_like(panel_tau),
                        panel_tau,
                    )
                    t_result.diagonal(dim1=-2, dim2=-1).copy_(safe_tau.reciprocal())
                    return t_result

                gram_ms = benchmark_ms(build_t_inv)
                t_inv = build_t_inv()
                project_ms = benchmark_ms(lambda: vt @ trailing)
                projected = vt @ trailing
                solve_ms = benchmark_ms(
                    lambda: torch.linalg.solve_triangular(
                        t_inv.transpose(-2, -1),
                        projected,
                        upper=False,
                    )
                )
                transformed = torch.linalg.solve_triangular(
                    t_inv.transpose(-2, -1),
                    projected,
                    upper=False,
                )
                lower = t_inv.transpose(-2, -1).contiguous()
                lower_pack_ms = benchmark_ms(lambda: t_inv.transpose(-2, -1).contiguous())
                lower_solve_ms = benchmark_ms(
                    lambda: torch.linalg.solve_triangular(
                        lower,
                        projected,
                        upper=False,
                    )
                )
                projected_col = projected.transpose(-2, -1).contiguous().transpose(-2, -1)
                projected_col_pack_ms = benchmark_ms(lambda: projected.transpose(-2, -1).contiguous().transpose(-2, -1))
                projected_col_solve_ms = benchmark_ms(
                    lambda: torch.linalg.solve_triangular(
                        lower,
                        projected_col,
                        upper=False,
                    )
                )
                identity = torch.eye(
                    width,
                    device="cuda",
                    dtype=torch.float32,
                ).expand(batch, -1, -1)

                def inverse_then_gemm():
                    inverse = torch.linalg.solve_triangular(
                        lower,
                        identity,
                        upper=False,
                    )
                    return inverse @ projected

                inverse_gemm_ms = benchmark_ms(inverse_then_gemm)
                apply_ms = benchmark_ms(
                    lambda: torch.baddbmm(
                        trailing,
                        v,
                        transformed,
                        beta=1.0,
                        alpha=-1.0,
                        out=trailing,
                    )
                )
                wy_parts_ms = v_ms + gram_ms + project_ms + solve_ms + apply_ms
                print(
                    "component,"
                    f"n={n},stage={stage},rows={rows},width={width},"
                    f"trailing={trailing_cols},panel_ms={panel_ms:.6f},"
                    f"compact_ms={compact_ms:.6f},v_ms={v_ms:.6f},"
                    f"gram_ms={gram_ms:.6f},project_ms={project_ms:.6f},"
                    f"solve_ms={solve_ms:.6f},apply_ms={apply_ms:.6f},"
                    f"lower_pack_ms={lower_pack_ms:.6f},"
                    f"lower_solve_ms={lower_solve_ms:.6f},"
                    f"projected_col_pack_ms={projected_col_pack_ms:.6f},"
                    f"projected_col_solve_ms={projected_col_solve_ms:.6f},"
                    f"inverse_gemm_ms={inverse_gemm_ms:.6f},"
                    f"parts_ms={wy_parts_ms:.6f}"
                )

            del source, destination
            torch.cuda.empty_cache()
        return 0

    seed = None
    set_seed(seed or 42)

    task_config = yaml.safe_load(open(CURRENT_DIR / "task.yml"))
    cases = task_config["tests" if mode == "test" else "benchmarks"]
    if case is not None:
        cases = [cases[case]]
    tests = [TestCase(x, spec="; ".join(f"{k}:{v}" for k, v in x.items())) for x in cases]

    with PopcornOutput(1) as logger:
        mp_context = multiprocessing.get_context("spawn")
        with mp_context.Pool(1) as pool:
            if mode == "test":
                return run_testing(logger, pool, tests)
            if mode == "benchmark":
                return run_benchmarking(logger, pool, tests)
            if mode == "leaderboard":
                for test in tests:
                    run_single_benchmark(pool, test, False, 1000, 5e8)
                logger.log("benchmark-count", len(tests))
                passed = True
                for idx, test in enumerate(tests):
                    logger.log(f"benchmark.{idx}.spec", test.spec)
                    result = run_single_benchmark(pool, test, True, 1000, 30e9)
                    if isinstance(result, Stats):
                        for field in dataclasses.fields(Stats):
                            logger.log(f"benchmark.{idx}.{field.name}", getattr(result, field.name))
                    else:
                        logger.log(f"benchmark.{idx}.status", "fail")
                        logger.log(f"benchmark.{idx}.error", str(result))
                        passed = False
                        break
                logger.log("check", "pass" if passed else "fail")
                return 0 if passed else 112
            return 2


@app.local_entrypoint()
def main(mode: str, file: str, case: int | None = None):
    data = run.remote(mode, file, case)
    if not isinstance(data, int):
        open(f"trace_{case}.json.gz", "wb").write(data)
