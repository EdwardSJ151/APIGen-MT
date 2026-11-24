import os
import sys
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]
    src = Path(__file__).parent / "src"
    if args[:1] == ["apigen-mt"]:
        script = src / "apigen_mt_cli.py"
        args = args[1:]
    elif args[:2] == ["--mode", "apigen-mt"]:
        script = src / "apigen_mt_cli.py"
        args = args[2:]
    else:
        script = src / "generate_step_by_step.py"
    os.execv(sys.executable, [sys.executable, str(script), *args])


if __name__ == "__main__":
    main()
