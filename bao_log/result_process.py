#!/usr/bin/env python3

import os
import sys
import re


def parse_log_file(file_path, delimiter=',', filter_column=None, filter_value=None, target_column=-2):
    """
    Parse a log file and extract values from a specified column.

    Args:
        file_path: Path to the log file
        delimiter: Column delimiter (None = whitespace, ',' = comma, etc.)
        filter_column: Column index to filter (0-based), None means no filter
        filter_value: Value to match in the filter column
        target_column: Column index to extract (negative for from-end indexing, default: -2)

    Returns:
        List of float values from the target column
    """
    values = []
    filtered_count = 0
    total_count = 0

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return values

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comment lines
            if not line or line.startswith('#'):
                continue

            # Split the line by delimiter
            columns = [col.strip() for col in line.split(delimiter)]

            # Check if we have at least 2 columns
            if len(columns) < 2:
                continue

            total_count += 1

            # Apply filter if specified
            if filter_column is not None and filter_value is not None:
                if filter_column >= len(columns):
                    continue

                filter_col_value = columns[filter_column].strip()
                if filter_col_value != str(filter_value):
                    continue

            filtered_count += 1

            try:
                # Get the target column
                value_str = columns[target_column].strip()

                # Remove any trailing commas or non-numeric characters except decimal point and minus
                value_str = re.sub(r'[^\d.-]', '', value_str)

                if not value_str:
                    continue

                value = float(value_str)
                values.append(value)
            except (ValueError, IndexError) as e:
                # Only print warning for lines that look like they should have data
                if len(columns) >= 2 and abs(target_column) <= len(columns):
                    print(f"Warning: Could not parse line {line_num}")
                    print(f"  Column value: '{columns[target_column] if abs(target_column) <= len(columns) else 'N/A'}'")
                    print(f"  Error: {e}")
                continue

    if filter_column is not None:
        print(f"  Total lines: {total_count}, Filtered (column {filter_column} == {filter_value}): {filtered_count}")

    return values


def calculate_ratio(bao_log_path, pg_log_path, delimiter=','):
    """
    Calculate the ratio of sums from two log files.

    Args:
        bao_log_path: Path to BAO test log file
        pg_log_path: Path to PG test log file
        delimiter: Column delimiter

    Returns:
        Ratio of BAO sum to PG sum
    """
    print("="*80)
    print("Processing log files (filtering column 1 == 2)...")
    print("="*80)

    # Parse BAO log file (only rows where column 1 equals 2, use third-to-last column)
    print(f"\nReading: {bao_log_path} (using column -3)")
    bao_values = parse_log_file(bao_log_path, delimiter, filter_column=1, filter_value=2, target_column=-3)

    if not bao_values:
        print(f"Error: No valid data found in {bao_log_path}")
        return None

    bao_sum = sum(bao_values)
    print(f"  Found {len(bao_values)} values")
    print(f"  Sum of third-to-last column: {bao_sum:.2f}")

    # Parse PG log file (only rows where column 1 equals 2, use second-to-last column)
    print(f"\nReading: {pg_log_path} (using column -2)")
    pg_values = parse_log_file(pg_log_path, delimiter, filter_column=1, filter_value=2, target_column=-2)

    if not pg_values:
        print(f"Error: No valid data found in {pg_log_path}")
        return None

    pg_sum = sum(pg_values)
    print(f"  Found {len(pg_values)} values")
    print(f"  Sum of second-to-last column: {pg_sum:.2f}")

    # Calculate ratio
    if pg_sum == 0:
        print("\nError: PG sum is zero, cannot calculate ratio")
        return None

    ratio = bao_sum / pg_sum

    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"BAO Sum:  {bao_sum:.2f}")
    print(f"PG Sum:   {pg_sum:.2f}")
    print(f"Ratio (1-BAO/PG): {1-ratio:.4f} ({(1-ratio)*100:.2f}%)")
    print("="*80)

    return ratio


def main():
    """Main function to run the script."""
    # Default file paths (script is in bao_log directory)
    bao_log_file = "bao_test_under_non_drift.log"
    pg_log_file = "pg_test.log"

    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        bao_log_file = sys.argv[1]
    if len(sys.argv) > 2:
        pg_log_file = sys.argv[2]

    # Default delimiter is comma for CSV files
    delimiter = ','
    if len(sys.argv) > 3:
        delimiter = sys.argv[3]
        if delimiter.lower() == 'tab':
            delimiter = '\t'
        elif delimiter.lower() == 'space':
            delimiter = None

    # Calculate ratio
    ratio = calculate_ratio(bao_log_file, pg_log_file, delimiter)

    if ratio is not None:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
