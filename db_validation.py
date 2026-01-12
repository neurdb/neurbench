"""
Database validation module for generated data.

This module:
1. Imports generated CSV tables into PostgreSQL
2. Runs JOB queries using existing Bao/PG test infrastructure
3. Compares query results to validate data quality per table

Uses psycopg2 for all database operations (no psql dependency).
"""

import os
import subprocess
import json
import io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import psycopg2
from psycopg2 import sql


@dataclass
class TableValidationResult:
    """Result of validating a single generated table."""
    table_name: str
    queries_run: int
    gen_total_time: float
    real_total_time: float
    time_ratio: float
    passed: bool
    threshold: float
    import_failed: bool = False  # True if data import failed (don't retry)


class DatabaseValidator:
    """Validates generated data by comparing query execution."""

    SCHEMA_FILE = "benchmarks/lqos/balsa/scripts/load-postgres/schema.sql"
    FK_INDEX_FILE = "benchmarks/lqos/balsa/scripts/load-postgres/fkindexes.sql"
    JOB_QUERY_DIR = "datasets/workloads/bao/join-order-benchmark"
    BAO_DIR = "benchmarks/lqos/bao"

    def __init__(
        self,
        gen_db: str = "imdb_2017_gen",
        real_db: str = "imdb_2017",
        host: str = "172.17.0.1",
        port: int = 5430,
        user: str = "postgres",
        password: str = "postgres",
    ):
        self.gen_db = gen_db
        self.real_db = real_db
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    def _connection_string(self, dbname: str) -> str:
        return f"dbname={dbname} user={self.user} password={self.password} host={self.host} port={self.port}"

    def _get_connection(self, dbname: str):
        return psycopg2.connect(self._connection_string(dbname))

    def _execute_sql(self, dbname: str, sql_str: str, timeout: int = 300) -> Tuple[bool, str]:
        """Execute SQL command using psycopg2."""
        try:
            conn = self._get_connection(dbname)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(f"SET statement_timeout TO {timeout * 1000}")
            cur.execute(sql_str)
            result = ""
            try:
                result = str(cur.fetchall())
            except:
                pass
            cur.close()
            conn.close()
            return True, result
        except Exception as e:
            return False, str(e)

    def _execute_sql_file(self, dbname: str, filepath: str) -> Tuple[bool, str]:
        """Execute SQL file using psycopg2."""
        try:
            with open(filepath, 'r') as f:
                sql_content = f.read()
            conn = self._get_connection(dbname)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(sql_content)
            cur.close()
            conn.close()
            return True, ""
        except Exception as e:
            return False, str(e)

    def get_generated_tables(self, dataset_name: str) -> List[str]:
        info_path = f"datasets/{dataset_name}/dataset_info.json"
        if not os.path.exists(info_path):
            return []
        with open(info_path, "r") as f:
            config = json.load(f)
        return [table for table, info in config.items() if info is not None]

    def database_exists(self, dbname: str) -> bool:
        """Check if a database exists using psycopg2."""
        try:
            conn = self._get_connection("postgres")
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            result = cur.fetchone()
            cur.close()
            conn.close()
            return result is not None
        except Exception as e:
            print(f"Error checking database existence: {e}")
            return False

    def create_database(self, dbname: str, force: bool = False) -> bool:
        """Check if database exists. Actual creation is done in copy_all_tables."""
        if self.database_exists(dbname):
            if not force:
                print(f"Database {dbname} already exists, skipping creation.")
                return True
            # Will be dropped and recreated in copy_all_tables
        return True

    def _copy_table_data(self, src_db: str, dst_db: str, table_name: str) -> bool:
        """Copy table data from one database to another using psycopg2 copy_expert."""
        import tempfile

        try:
            # Use temp file for large tables
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                tmp_path = tmp.name

            # Export from source using psycopg2 copy_expert
            src_conn = self._get_connection(src_db)
            src_cur = src_conn.cursor()
            with open(tmp_path, 'w', encoding='utf-8') as f:
                src_cur.copy_expert(f"COPY {table_name} TO STDOUT WITH CSV HEADER", f)
            src_cur.close()
            src_conn.close()

            # Truncate destination table
            dst_conn = self._get_connection(dst_db)
            dst_conn.autocommit = True
            dst_cur = dst_conn.cursor()
            # Disable FK checks, delete data, then re-enable
            dst_cur.execute("SET session_replication_role = 'replica'")
            dst_cur.execute(f"TRUNCATE TABLE {table_name}")
            dst_cur.execute("SET session_replication_role = 'origin'")

            # Import to destination using psycopg2 copy_expert
            with open(tmp_path, 'r', encoding='utf-8') as f:
                dst_cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV HEADER", f)
            dst_cur.close()
            dst_conn.close()

            # Clean up temp file
            os.remove(tmp_path)
            return True
        except Exception as e:
            print(f"FAILED: {e}")
            return False

    def _get_table_row_count(self, dbname: str, table_name: str) -> int:
        """Get row count for a table."""
        try:
            conn = self._get_connection(dbname)
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except:
            return -1

    def copy_all_tables(self, src_db: str, dst_db: str) -> bool:
        """Drop, create and copy all tables using dropdb && createdb && pg_dump | psql via docker exec."""
        print(f"\nSetting up {dst_db} from {src_db} via dropdb && createdb && pg_dump | psql...")

        # Combine all operations into one atomic command
        # This ensures database is created and fully populated before returning
        cmd = f"docker exec -i pg_bao bash -c \"su - postgres -c 'dropdb -p 5432 --if-exists {dst_db} && createdb -p 5432 {dst_db} && pg_dump -p 5432 {src_db} | psql -p 5432 {dst_db}'\""
        print(f"  Running: {cmd}")

        # Use capture_output=False to show real-time output
        result = subprocess.run(cmd, shell=True)

        if result.returncode != 0:
            print(f"FAILED")
            return False

        print("OK - database created and all tables copied")
        return True

    def _get_integer_columns(self, dbname: str, table_name: str) -> List[str]:
        """Get list of integer columns for a table."""
        try:
            conn = self._get_connection(dbname)
            cur = conn.cursor()
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                AND data_type IN ('integer', 'bigint', 'smallint')
            """, (table_name,))
            columns = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            return columns
        except Exception as e:
            print(f"Warning: Could not get integer columns: {e}")
            return []

    def _get_not_null_string_columns(self, dbname: str, table_name: str) -> List[str]:
        """Get list of NOT NULL string columns for a table."""
        try:
            conn = self._get_connection(dbname)
            cur = conn.cursor()
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                AND is_nullable = 'NO'
                AND data_type IN ('character varying', 'varchar', 'text', 'character', 'char')
            """, (table_name,))
            columns = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            return columns
        except Exception as e:
            print(f"Warning: Could not get NOT NULL string columns: {e}")
            return []

    def _get_not_null_int_columns(self, dbname: str, table_name: str) -> List[str]:
        """Get list of NOT NULL integer columns for a table."""
        try:
            conn = self._get_connection(dbname)
            cur = conn.cursor()
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                AND is_nullable = 'NO'
                AND data_type IN ('integer', 'bigint', 'smallint')
            """, (table_name,))
            columns = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            return columns
        except Exception as e:
            print(f"Warning: Could not get NOT NULL int columns: {e}")
            return []

    def _clean_csv_for_import(self, csv_path: str, int_columns: List[str],
                               not_null_string_columns: List[str] = None) -> str:
        """Clean CSV file for PostgreSQL import.

        Handles:
        1. Float columns that should be integers (3.0 → 3)
        2. Year columns where 0 should be NULL (0 is not a valid year)
        3. Year columns with float artifacts (1966.499... → 1966)
        4. String columns with backslash-escaped quotes (\" → "")
        5. Proper UTF-8 encoding for strings
        6. NOT NULL string columns with empty values (fill with placeholder)
        """
        if not_null_string_columns is None:
            not_null_string_columns = []
        import pandas as pd
        import numpy as np

        # Year columns where 0 should be treated as NULL
        year_columns = {'production_year', 'start_year', 'end_year', 'year'}

        # First pass: fix backslash-escaped quotes in raw file
        with open(csv_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace backslash-escaped quotes with doubled quotes for CSV
        # This handles cases like: Benjamin Britten's \"Paul Bunyan\"
        if '\\"' in content:
            content = content.replace('\\"', '""')
            # Write to temp file for pandas to read
            temp_path = csv_path + ".temp.csv"
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            csv_path = temp_path

        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='warn', low_memory=False)

        # Clean up temp file if created
        if csv_path.endswith('.temp.csv') and os.path.exists(csv_path):
            pass  # Will be cleaned up later

        # Convert integer columns (handle float, string, or mixed types)
        for col in int_columns:
            if col not in df.columns:
                continue

            # For year columns, handle specially
            if col in year_columns:
                def clean_year(x):
                    if pd.isna(x) or x == '' or x == 'None':
                        return None
                    try:
                        val = round(float(x))  # Convert string/float to int
                        if val <= 0 or val > 2100:  # Invalid year range
                            return None
                        return val
                    except (ValueError, TypeError):
                        return None
                df[col] = df[col].apply(clean_year)
            else:
                # Convert to nullable int, handling NaN/string/float values
                # Also handle INT64_MIN/MAX which indicate NaN conversion errors
                INT32_MIN = -2147483648
                INT32_MAX = 2147483647
                INT64_MIN = -9223372036854775808
                INT64_MAX = 9223372036854775807
                overflow_count = [0]  # Use list for closure

                def to_int(x):
                    if pd.isna(x) or x == '' or x == 'None':
                        return None
                    try:
                        val = int(round(float(x)))
                        # INT64 extremes are artifacts from float->int conversion, convert to 0
                        if val == INT64_MIN or val == INT64_MAX:
                            overflow_count[0] += 1
                            return 0
                        # Values outside INT32 range also converted to 0
                        if val < INT32_MIN or val > INT32_MAX:
                            overflow_count[0] += 1
                            return 0
                        return val
                    except (ValueError, TypeError, OverflowError):
                        return None
                df[col] = df[col].apply(to_int)
                if overflow_count[0] > 0:
                    print(f"  Column '{col}': {overflow_count[0]} overflow values converted to 0")

        # Convert integer columns to proper format for CSV (avoid 2003.0 issue)
        # When a column has None values, pandas converts it to float64
        # We need to format these as integers in the CSV output
        for col in int_columns:
            if col in df.columns:
                # Convert to string, replacing NaN with empty string
                df[col] = df[col].apply(lambda x: '' if pd.isna(x) else str(int(x)))

        # Fix NOT NULL string columns - fill empty/null values with placeholder
        for col in not_null_string_columns:
            if col in df.columns:
                # Count how many we're fixing
                null_count = df[col].isna().sum()
                empty_count = (df[col] == '').sum() if df[col].dtype == object else 0
                if null_count > 0 or empty_count > 0:
                    print(f"  Fixing NOT NULL column '{col}': {null_count} null, {empty_count} empty -> '(unknown)'")
                    df[col] = df[col].fillna('(unknown)')
                    df[col] = df[col].replace('', '(unknown)')

        # Save to temp file with UTF-8 encoding
        clean_path = csv_path.replace('.temp.csv', '') + ".clean.csv"
        df.to_csv(clean_path, index=False, encoding='utf-8')

        # Clean up temp file
        if csv_path.endswith('.temp.csv') and os.path.exists(csv_path):
            os.remove(csv_path)

        return clean_path

    def import_generated_table(self, dbname: str, dataset_name: str, table_name: str,
                               reference_dataset: str = None, variant_id: int = -1) -> bool:
        """Import generated CSV into table using psycopg2."""
        # Build path matching dbproc.py logic
        dataset_dir = dataset_name
        if reference_dataset and reference_dataset != dataset_name:
            dataset_dir = f"{dataset_name}_ref_{reference_dataset}"
        if variant_id > 0:
            dataset_dir += f"-{variant_id}"
        csv_path = os.path.join("expdir", dataset_dir, table_name, f"{table_name}.drifted.csv")
        if not os.path.exists(csv_path):
            print(f"Generated file not found: {csv_path}")
            return False
        print(f"Importing generated {table_name}...", end=" ", flush=True)
        try:
            conn = self._get_connection(dbname)
            conn.autocommit = True
            cur = conn.cursor()

            # Get integer columns and NOT NULL string columns, then clean CSV
            int_columns = self._get_integer_columns(dbname, table_name)
            not_null_str_columns = self._get_not_null_string_columns(dbname, table_name)
            print(f"(int_columns={int_columns})...", end=" ", flush=True)
            import_path = self._clean_csv_for_import(csv_path, int_columns, not_null_str_columns)
            print(f"(cleaned: {import_path})...", end=" ", flush=True)

            # Debug: print first 3 lines of cleaned file
            with open(import_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 3:
                        print(f"\n  DEBUG line {i}: {line.strip()[:200]}")
                    else:
                        break

            # Disable FK checks, delete data, then re-enable
            cur.execute("SET session_replication_role = 'replica'")  # Disable FK triggers
            cur.execute(f"TRUNCATE TABLE {table_name}")
            cur.execute("SET session_replication_role = 'origin'")   # Re-enable FK triggers
            # Import CSV with explicit UTF-8 encoding
            with open(import_path, 'r', encoding='utf-8') as f:
                copy_sql = f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8')"
                cur.copy_expert(copy_sql, f)
            # Analyze
            cur.execute(f"ANALYZE {table_name}")
            cur.close()
            conn.close()

            # Clean up temp file
            if import_path != csv_path and os.path.exists(import_path):
                os.remove(import_path)

            print("OK")
            return True
        except Exception as e:
            print(f"FAILED: {e}")
            return False

    def restore_table(self, table_name: str) -> bool:
        """Restore table from real database using psycopg2."""
        print(f"Restoring {table_name} from {self.real_db}...", end=" ", flush=True)
        if self._copy_table_data(self.real_db, self.gen_db, table_name):
            print("OK")
            return True
        else:
            print("FAILED")
            return False

    def analyze_database(self, dbname: str) -> bool:
        """Run ANALYZE on the database to update statistics for query planner."""
        print(f"Running ANALYZE VERBOSE on {dbname}...", flush=True)
        try:
            conn = self._get_connection(dbname)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("ANALYZE VERBOSE")
            cur.close()
            conn.close()
            print("ANALYZE complete.", flush=True)
            return True
        except Exception as e:
            print(f"ANALYZE FAILED: {e}", flush=True)
            return False

    def run_job_queries(self, dbname: str, output_file: str) -> bool:
        print(f"\nRunning JOB queries on {dbname}...", flush=True)
        # Always ANALYZE before running queries to ensure accurate statistics
        self.analyze_database(dbname)
        if os.path.exists(output_file):
            os.remove(output_file)
        cmd = f"cd {self.BAO_DIR} && python3 run_test_queries.py " \
              f"--use_postgres --database_name {dbname} " \
              f"--query_dir ../../../{self.JOB_QUERY_DIR} " \
              f"--output_file ../../../{output_file} --db-port {self.port}"
        print(f"Command: {cmd}", flush=True)
        # Use Popen for real-time output that works with TeeLogger
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
        return process.returncode == 0

    def parse_query_results(self, output_file: str) -> Dict[str, float]:
        """Parse query results, only use 3rd execution (i=2) for each query."""
        results = {}
        if not os.path.exists(output_file):
            return results
        with open(output_file, 'r') as f:
            for line in f:
                parts = line.strip().split(', ')
                if len(parts) >= 6:
                    exec_idx = int(parts[1])  # 0, 1, 2
                    # Only use 3rd execution (index 2) - after warmup
                    if exec_idx != 2:
                        continue
                    query_path = parts[3]
                    exec_time = float(parts[5])
                    query_name = os.path.basename(query_path).replace('.sql', '')
                    results[query_name] = exec_time
        return results

    def compare_results(self, gen_results: Dict[str, float], real_results: Dict[str, float], threshold: float = 1.2) -> Tuple[int, int, float]:
        """Compare query results with bidirectional threshold check.

        For threshold=1.2 (20%), accepts ratio in range [1/1.2, 1.2] = [0.833, 1.2]
        """
        total = passed = 0
        ratios = []
        lower_bound = 1.0 / threshold  # e.g., 0.833 for threshold=1.2
        upper_bound = threshold         # e.g., 1.2

        for query_name in gen_results:
            if query_name not in real_results:
                continue
            gen_time = gen_results[query_name]
            real_time = real_results[query_name]
            ratio = gen_time / real_time if real_time > 0 else (1.0 if gen_time == 0 else float('inf'))
            total += 1
            # Bidirectional check: pass if within [lower_bound, upper_bound]
            is_passed = lower_bound <= ratio <= upper_bound
            if is_passed:
                passed += 1
            ratios.append(ratio)
            status = "PASS" if is_passed else "FAIL"
            print(f"  {query_name}: {status} (gen={gen_time:.1f}ms, real={real_time:.1f}ms, ratio={ratio:.2f}x)")
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        return total, passed, avg_ratio

    def validate_single_table(self, dataset_name: str, table_name: str, threshold: float = 1.2,
                              skip_real_queries: bool = False, dry_run: bool = False,
                              reference_dataset: str = None, variant_id: int = -1) -> TableValidationResult:
        print(f"\n{'='*60}")
        print(f"Validating table: {table_name}" + (" [DRY RUN - no table replacement]" if dry_run else ""))
        print(f"{'='*60}")

        # In dry_run mode, skip table import - just compare current db state
        if not dry_run:
            if not self.import_generated_table(self.gen_db, dataset_name, table_name,
                                               reference_dataset=reference_dataset, variant_id=variant_id):
                # Import failed - restore table and mark as import_failed
                print(f"Import failed, restoring {table_name} from real database...")
                self.restore_table(table_name)
                return TableValidationResult(
                    table_name=table_name, queries_run=0, gen_total_time=0, real_total_time=0,
                    time_ratio=float('inf'), passed=False, threshold=threshold, import_failed=True
                )

        # Build log directory path (same as gd_logs structure)
        log_dir = dataset_name
        if reference_dataset and reference_dataset != dataset_name:
            log_dir = f"{dataset_name}_ref_{reference_dataset}"
        if variant_id > 0:
            log_dir = f"{log_dir}-{variant_id}"
        log_path = f"validation_logs/{log_dir}"
        os.makedirs(log_path, exist_ok=True)

        gen_output = f"{log_path}/{table_name}_gen.log"
        self.run_job_queries(self.gen_db, gen_output)
        real_output = f"{log_path}/real.log"
        # Use cached real query results if available, unless explicitly told to re-run
        if not skip_real_queries:
            if os.path.exists(real_output):
                print(f"\nUsing cached real database results: {real_output}")
            else:
                print(f"\nRunning queries on real database {self.real_db}...")
                self.run_job_queries(self.real_db, real_output)
        gen_results = self.parse_query_results(gen_output)
        real_results = self.parse_query_results(real_output)
        print(f"\nComparing results for {table_name}:")
        total, passed_count, _ = self.compare_results(gen_results, real_results, threshold)
        gen_total = sum(gen_results.values())
        real_total = sum(real_results.values())
        # Use total time ratio (sum of all query times) for final judgment
        total_ratio = gen_total / real_total if real_total > 0 else float('inf')
        # Bidirectional check: pass if total_ratio within [1/threshold, threshold]
        lower_bound = 1.0 / threshold
        upper_bound = threshold
        is_passed = lower_bound <= total_ratio <= upper_bound
        result = TableValidationResult(table_name=table_name, queries_run=total, gen_total_time=gen_total, real_total_time=real_total, time_ratio=total_ratio, passed=is_passed, threshold=threshold)
        range_str = f"[{lower_bound:.2f}, {upper_bound:.2f}]"
        print(f"\nTable {table_name} result: gen_total={gen_total:.1f}ms, real_total={real_total:.1f}ms, ratio={total_ratio:.2f}x, range={range_str}, {'PASSED' if result.passed else 'FAILED'}")

        # In dry_run mode, no table changes were made, so no restore needed
        if not dry_run:
            # Restore table only if validation failed (keep generated data if passed)
            if not is_passed:
                print(f"Validation failed, restoring {table_name} from real database...")
                self.restore_table(table_name)
            else:
                print(f"Validation passed, keeping generated data for {table_name}")
        return result

    def validate_all_tables(self, dataset_name: str, threshold: float = 1.2,
                            force_reload: bool = False, dry_run: bool = False,
                            reference_dataset: str = None, variant_id: int = -1) -> Dict[str, TableValidationResult]:
        print(f"\n{'#'*60}")
        print(f"Validating all generated tables for {dataset_name}" + (" [DRY RUN]" if dry_run else ""))
        print(f"{'#'*60}")

        # Check if gen_db already exists
        if self.database_exists(self.gen_db):
            if force_reload:
                print(f"Database {self.gen_db} already exists, force reloading...")
                if not self.create_database(self.gen_db, force=True):
                    return {}
                if not self.copy_all_tables(self.real_db, self.gen_db):
                    return {}
            else:
                print(f"Database {self.gen_db} already exists, skipping setup.")
        else:
            # Create database and copy tables
            if not self.create_database(self.gen_db):
                return {}
            if not self.copy_all_tables(self.real_db, self.gen_db):
                return {}
        generated_tables = self.get_generated_tables(dataset_name)
        print(f"\nGenerated tables: {generated_tables}")
        real_output = f"validation_logs/{dataset_name}_real.log"
        os.makedirs("validation_logs", exist_ok=True)
        # Use cached real query results if available
        if os.path.exists(real_output):
            print(f"\nUsing cached real database results: {real_output}")
        else:
            print(f"\nRunning queries on real database {self.real_db}...")
            self.run_job_queries(self.real_db, real_output)
        results = {}
        failed_tables = []
        for table_name in generated_tables:
            # skip_real_queries=True since we already ran them above
            result = self.validate_single_table(dataset_name, table_name, threshold,
                                                skip_real_queries=True, dry_run=dry_run,
                                                reference_dataset=reference_dataset, variant_id=variant_id)
            results[table_name] = result
            if not result.passed:
                failed_tables.append(table_name)
        print(f"\n{'#'*60}")
        print(f"SUMMARY: {len(generated_tables) - len(failed_tables)}/{len(generated_tables)} passed")
        if failed_tables:
            print(f"Failed: {failed_tables}")
        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate generated database tables")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--table-name", type=str, default=None)
    parser.add_argument("--gen-db", type=str, default="imdb_2017_gen")
    parser.add_argument("--real-db", type=str, default="imdb_2017")
    parser.add_argument("--host", type=str, default="172.17.0.1")
    parser.add_argument("--port", type=int, default=5430)
    parser.add_argument("--threshold", type=float, default=1.2)
    args = parser.parse_args()
    validator = DatabaseValidator(gen_db=args.gen_db, real_db=args.real_db, host=args.host, port=args.port)
    if args.table_name:
        validator.create_database(validator.gen_db)
        validator.copy_all_tables(validator.real_db, validator.gen_db)
        result = validator.validate_single_table(args.dataset_name, args.table_name, args.threshold)
        exit(0 if result.passed else 1)
    else:
        results = validator.validate_all_tables(args.dataset_name, args.threshold)
        failed = [t for t, r in results.items() if not r.passed]
        exit(0 if len(failed) == 0 else 1)
