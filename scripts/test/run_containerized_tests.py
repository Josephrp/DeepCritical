#!/usr/bin/env python3
"""
Containerized test runner for DeepCritical.

This script runs tests in containerized environments for enhanced isolation
and security validation.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_docker_tests():
    """Run Docker-specific tests."""
    print("🐳 Running Docker sandbox tests...")

    env = os.environ.copy()
    env["DOCKER_TESTS"] = "true"

    cmd = ["python", "-m", "pytest", "tests/test_docker_sandbox/", "-v", "--tb=short"]

    try:
        result = subprocess.run(cmd, check=False, env=env, cwd=Path.cwd())
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running Docker tests: {e}")
        return False


def run_bioinformatics_tests():
    """Run bioinformatics tools tests."""
    print("🧬 Running bioinformatics tools tests...")

    env = os.environ.copy()
    env["DOCKER_TESTS"] = "true"

    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/test_bioinformatics_tools/",
        "-v",
        "--tb=short",
    ]

    try:
        result = subprocess.run(cmd, check=False, env=env, cwd=Path.cwd())
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running bioinformatics tests: {e}")
        return False


def run_llm_tests():
    """Run LLM framework tests."""
    print("🤖 Running LLM framework tests...")

    cmd = ["python", "-m", "pytest", "tests/test_llm_framework/", "-v", "--tb=short"]

    try:
        result = subprocess.run(cmd, check=False, cwd=Path.cwd())
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running LLM tests: {e}")
        return False


def run_performance_tests():
    """Run performance tests."""
    print("📊 Running performance tests...")

    env = os.environ.copy()
    env["PERFORMANCE_TESTS"] = "true"

    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/",
        "-m",
        "performance",
        "--benchmark-only",
        "--benchmark-json=benchmark.json",
    ]

    try:
        result = subprocess.run(cmd, check=False, env=env, cwd=Path.cwd())
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running performance tests: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run containerized tests for DeepCritical"
    )
    parser.add_argument(
        "--docker", action="store_true", help="Run Docker sandbox tests"
    )
    parser.add_argument(
        "--bioinformatics", action="store_true", help="Run bioinformatics tools tests"
    )
    parser.add_argument("--llm", action="store_true", help="Run LLM framework tests")
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all containerized tests"
    )

    args = parser.parse_args()

    # If no specific tests requested, run all
    if not any(
        [args.docker, args.bioinformatics, args.llm, args.performance, args.all]
    ):
        args.all = True

    success = True

    if args.all or args.docker:
        success &= run_docker_tests()

    if args.all or args.bioinformatics:
        success &= run_bioinformatics_tests()

    if args.all or args.llm:
        success &= run_llm_tests()

    if args.all or args.performance:
        success &= run_performance_tests()

    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
