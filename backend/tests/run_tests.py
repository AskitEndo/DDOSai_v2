#!/usr/bin/env python3
"""
Test runner for DDoS.AI platform
"""
import argparse
import subprocess
import os
import sys
import time
import json
from datetime import datetime

# Test categories
TEST_CATEGORIES = {
    "unit": ["tests/test_metrics.py"],
    "integration": ["tests/integration/test_ai_pipeline.py"],
    "e2e": ["tests/e2e/test_api_endpoints.py"],
    "load": ["tests/load/test_load.py"],
    "security": ["tests/security/test_security.py"],
    "all": []  # Will be populated with all tests
}

# Populate "all" category
for category, tests in TEST_CATEGORIES.items():
    if category != "all":
        TEST_CATEGORIES["all"].extend(tests)

def run_tests(categories, verbose=False, html_report=False, json_report=False):
    """Run tests for the specified categories"""
    # Collect all test files to run
    test_files = []
    for category in categories:
        if category in TEST_CATEGORIES:
            test_files.extend(TEST_CATEGORIES[category])
        else:
            print(f"Warning: Unknown test category '{category}'")
    
    # Remove duplicates
    test_files = list(set(test_files))
    
    if not test_files:
        print("No test files to run")
        return 1
    
    # Build pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add HTML report
    if html_report:
        report_dir = "test_reports"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        cmd.extend(["--html", report_file])
    
    # Add JSON report
    if json_report:
        report_dir = "test_reports"
        os.makedirs(report_dir, exist_ok=True)
        json_file = os.path.join(report_dir, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        cmd.extend(["--json", json_file])
    
    # Add test files
    cmd.extend(test_files)
    
    # Run tests
    print(f"Running tests: {' '.join(cmd)}")
    start_time = time.time()
    result = subprocess.run(cmd)
    end_time = time.time()
    
    # Print summary
    print(f"\nTest run completed in {end_time - start_time:.2f} seconds")
    print(f"Exit code: {result.returncode}")
    
    if html_report:
        print(f"HTML report saved to {report_file}")
    
    if json_report:
        print(f"JSON report saved to {json_file}")
    
    return result.returncode

def generate_test_data(output_dir="test_data", attack=True):
    """Generate test data for tests"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate traffic dataset
    traffic_file = os.path.join(output_dir, "traffic_dataset.json")
    cmd = [
        "python", "tests/data/generate_test_data.py",
        "--type", "traffic",
        "--output", traffic_file,
        "--duration", "5",
        "--rate", "10"
    ]
    
    if attack:
        cmd.extend([
            "--attack",
            "--attack-type", "syn_flood",
            "--attack-duration", "2",
            "--attack-start", "2",
            "--attack-rate", "50"
        ])
    
    print(f"Generating traffic dataset: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    # Generate feature dataset
    feature_file = os.path.join(output_dir, "feature_dataset.csv")
    cmd = [
        "python", "tests/data/generate_test_data.py",
        "--type", "features",
        "--output", feature_file,
        "--samples", "1000",
        "--malicious-ratio", "0.2"
    ]
    
    print(f"Generating feature dataset: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    print(f"Test data generated in {output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run tests for DDoS.AI platform")
    parser.add_argument("--category", nargs="+", default=["all"],
                        help=f"Test categories to run: {', '.join(TEST_CATEGORIES.keys())}")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML report")
    parser.add_argument("--json", action="store_true",
                        help="Generate JSON report")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate test data before running tests")
    parser.add_argument("--data-dir", default="test_data",
                        help="Directory for generated test data")
    parser.add_argument("--no-attack", action="store_true",
                        help="Don't include attack traffic in generated data")
    
    args = parser.parse_args()
    
    # Generate test data if requested
    if args.generate_data:
        generate_test_data(args.data_dir, not args.no_attack)
    
    # Run tests
    return run_tests(args.category, args.verbose, args.html, args.json)

if __name__ == "__main__":
    sys.exit(main())