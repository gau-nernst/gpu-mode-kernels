import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["CUTE_DSL_KEEP_PTX"] = "1"
# os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ["CUTE_DSL_LINEINFO"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = str(Path(__file__).parent / "dump")
os.environ["CUTE_DSL_NO_CACHE"] = "1"

import yaml

TASK_DIR = Path(__file__).resolve().parent
UTILS_FILE = TASK_DIR.parents[1] / "pmpp_v2" / "utils.py"
EVAL_FILES = ("eval.py", "reference.py", "task.py")
GREEN = "\033[1;32m"
RED = "\033[1;31m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("test", "benchmark", "leaderboard"),
        default="test",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=int,
        metavar="INDEX",
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_cases(mode: str, selected: list[int] | None) -> list[dict]:
    with (TASK_DIR / "task.yml").open() as task_file:
        task = yaml.safe_load(task_file)

    section = "tests" if mode == "test" else "benchmarks"
    cases = task[section]
    if selected is None:
        return cases

    invalid = [index for index in selected if index < 0 or index >= len(cases)]
    if invalid:
        valid_range = f"0..{len(cases) - 1}"
        raise ValueError(f"case index out of range: {invalid}; valid range is {valid_range}")
    return [cases[index] for index in selected]


def write_test_specs(path: Path, cases: list[dict]) -> None:
    lines = ["; ".join(f"{key}: {value}" for key, value in case.items()) for case in cases]
    path.write_text("\n".join(lines) + "\n")


def write_submission_shim(path: Path, submission: Path) -> None:
    source = f"""\
import importlib.util
import os
import sys
from pathlib import Path

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

submission_path = Path({str(submission)!r})
sys.path.insert(0, str(submission_path.parent))
spec = importlib.util.spec_from_file_location("_local_submission", submission_path)
if spec is None or spec.loader is None:
    raise ImportError(f"could not load submission from {{submission_path}}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
custom_kernel = module.custom_kernel
"""
    path.write_text(source)


def print_output(line: str) -> None:
    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    text = line.rstrip("\n")
    if color and text.endswith(": pass"):
        text = f"{GREEN}{text}{RESET}"
    elif color and (text.endswith(": fail") or ".error:" in text or text.startswith("error:")):
        text = f"{RED}{text}{RESET}"
    print(text, flush=True)


def run_evaluator(submission: Path, mode: str, cases: list[dict], seed: int | None) -> int:
    if not UTILS_FILE.is_file():
        raise FileNotFoundError(f"evaluator utility not found: {UTILS_FILE}")

    with tempfile.TemporaryDirectory(prefix="qr-eval-") as temp_dir:
        work_dir = Path(temp_dir)
        for file_name in EVAL_FILES:
            shutil.copy2(TASK_DIR / file_name, work_dir / file_name)
        shutil.copy2(UTILS_FILE, work_dir / "utils.py")
        write_submission_shim(work_dir / "submission.py", submission)

        specs_file = work_dir / "cases.txt"
        write_test_specs(specs_file, cases)

        env = os.environ.copy()
        env["POPCORN_FD"] = "1"
        if seed is not None:
            env["POPCORN_SEED"] = str(seed)

        command = [sys.executable, "eval.py", mode, str(specs_file)]
        process = subprocess.Popen(
            command,
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print_output(line)
        return process.wait()


def main() -> int:
    args = parse_args()
    submission = args.submission.expanduser().resolve()
    if not submission.is_file():
        print(f"submission not found: {submission}", file=sys.stderr)
        return 2

    try:
        cases = load_cases(args.mode, args.cases)
        return run_evaluator(submission, args.mode, cases, args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
