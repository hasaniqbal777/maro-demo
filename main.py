"""MARO — entry point. Use one of the scripts in scripts/ or run the Streamlit demo."""

USAGE = """\
MARO: Multi-Agent Framework with Automated Decision Rule Optimization.

Common commands (all via uv):

  uv run python scripts/download_pheme.py
  uv run python scripts/run_inference.py   --news "<text>" --model gpt-5.4-mini
  uv run python scripts/run_optimization.py --event charlie_hebdo --iters 30
  uv run python scripts/run_evaluation.py   --n 100
  uv run streamlit run demo/app.py
"""


def main() -> None:
    print(USAGE)


if __name__ == "__main__":
    main()
