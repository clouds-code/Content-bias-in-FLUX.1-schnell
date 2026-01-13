from __future__ import annotations

import subprocess
import sys


def _run(module: str) -> int:
    return subprocess.call([sys.executable, module])


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python main.py [prompt_gen|image_engine|auditor|analyzer|all]")
        raise SystemExit(2)
    cmd = sys.argv[1].strip().lower()
    mapping = {
        "prompt_gen": "prompt_gen.py",
        "image_engine": "image_engine.py",
        "auditor": "auditor.py",
        "analyzer": "analyzer.py",
    }
    if cmd == "all":
        for key in ["prompt_gen", "image_engine", "auditor", "analyzer"]:
            code = _run(mapping[key])
            if code != 0:
                raise SystemExit(code)
        return
    if cmd not in mapping:
        print("Usage: python main.py [prompt_gen|image_engine|auditor|analyzer|all]")
        raise SystemExit(2)
    raise SystemExit(_run(mapping[cmd]))


if __name__ == "__main__":
    main()

