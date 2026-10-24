from __future__ import annotations

from pathlib import Path

from diagnostics_stage1 import run_scenario


def run_demo(output_dir: str = "diagnostics") -> Path:
    return run_scenario("normal", Path(output_dir), seed=0)


if __name__ == "__main__":
    output = run_demo()
    print(f"Normal diagnostic written to: {output}")
