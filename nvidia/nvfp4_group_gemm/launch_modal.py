import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args, rest = parser.parse_known_args()

    current_dir = Path(__file__).parent
    with open(current_dir / args.file) as fin, open(current_dir / "submission.py", "w") as fout:
        fout.write(fin.read())

    try:
        subprocess.run(["modal", "run", str(current_dir / "eval.py"), *rest], check=True)
    except Exception as e:
        print(e)
