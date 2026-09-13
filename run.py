"""
Top-level entry point for Affordra ⚡ Autonomous AI Financial Decision Agent.
Delegates to code/main.py.
"""
import sys
from pathlib import Path

# Add code/ directory to path
code_dir = Path(__file__).parent / "code"
sys.path.insert(0, str(code_dir))

from main import main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Affordra ⚡ — Autonomous AI Financial Decision Agent")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM calls (default: template explanations)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation step")
    args = parser.parse_args()

    main(use_llm=args.use_llm, run_eval=not args.no_eval)
