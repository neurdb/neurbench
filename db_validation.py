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
        gen_db: str = "imdb_17v2_gen",
        real_db: str = "imdb_17v2",
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
        """Create database using psycopg2. If exists and force=False, skip creation."""
        if self.database_exists(dbname):
            if not force:
                print(f"Database {dbname} already exists, skipping creation.")
                return True
            print(f"Database {dbname} already exists, dropping...")
            try:
                conn = self._get_connection("postgres")
                conn.autocommit = True
                cur = conn.cursor()
                # Terminate connections
                cur.execute("""
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                """, (dbname,))
                # Drop database
                cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(dbname)))
                cur.close()
                conn.close()
            except Exception as e:
                print(f"Failed to drop database: {e}")
                return False

        print(f"Creating database {dbname}...")
        try:
            # Create database
            conn = self._get_connection("postgres")
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
            cur.close()
            conn.close()

            # Apply schema
            print("Applying schema...")
            success, err = self._execute_sql_file(dbname, self.SCHEMA_FILE)
            if not success:
                print(f"Failed to apply schema: {err}")
                return False

            # Apply indexes
            if os.path.exists(self.FK_INDEX_FILE):
                print("Applying indexes...")
                self._execute_sql_file(dbname, self.FK_INDEX_FILE)

            print(f"Database {dbname} created successfully")
            return True
        except Exception as e:
            print(f"Failed to create database: {e}")
            return False

    def _copy_table_data(self, src_db: str, dst_db: str, table_name: str) -> bool:
        """Copy table data from one database to another using psycopg2 copy_expert."""
        try:
            # Export from source
            src_conn = self._get_connection(src_db)
            src_cur = src_conn.cursor()
            buffer = io.StringIO()
            copy_sql = f"COPY {table_name} TO STDOUT WITH CSV HEADER"
            src_cur.copy_expert(copy_sql, buffer)
            src_cur.close()
            src_conn.close()

            # Import to destination
            buffer.seek(0)
            dst_conn = self._get_connection(dst_db)
            dst_conn.autocommit = True
            dst_cur = dst_conn.cursor()
            dst_cur.execute(f"TRUNCATE TABLE {table_name} CASCADE")
            buffer.seek(0)
            copy_sql = f"COPY {table_name} FROM STDIN WITH CSV HEADER"
            dst_cur.copy_expert(copy_sql, buffer)
            dst_cur.close()
            dst_conn.close()
            return True
        except Exception as e:
            print(f"FAILED: {e}")
            return False

    def copy_all_tables(self, src_db: str, dst_db: str) -> bool:
        """Copy all tables from source to destination database using psycopg2."""
        print(f"\nCopying all tables from {src_db} to {dst_db}...")
        all_tables = [
            "aka_name", "aka_title", "cast_info", "char_name", "comp_cast_type",
            "company_name", "company_type", "complete_cast", "info_type", "keyword",
            "kind_type", "link_type", "movie_companies", "movie_info", "movie_info_idx",
            "movie_keyword", "movie_link", "name", "person_info", "role_type", "title"
        ]
        for table in all_tables:
            print(f"  Copying {table}...", end=" ", flush=True)
            if self._copy_table_data(src_db, dst_db, table):
                print("OK")
            else:
                print("FAILED")
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

    def _clean_csv_for_import(self, csv_path: str, int_columns: List[str]) -> str:
        """Clean CSV file for PostgreSQL import.

        Handles:
        1. Float columns that should be integers (3.0 → 3)
        2. Year columns where 0 should be NULL (0 is not a valid year)
        3. Year columns with float artifacts (1966.499... → 1966)
        4. String columns with backslash-escaped quotes (\" → "")
        5. Proper UTF-8 encoding for strings
        """
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

        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='warn')

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
                def to_int(x):
                    if pd.isna(x) or x == '' or x == 'None':
                        return None
                    try:
                        return int(round(float(x)))
                    except (ValueError, TypeError):
                        return None
                df[col] = df[col].apply(to_int)

        # Convert integer columns to proper format for CSV (avoid 2003.0 issue)
        # When a column has None values, pandas converts it to float64
        # We need to format these as integers in the CSV output
        for col in int_columns:
            if col in df.columns:
                # Convert to string, replacing NaN with empty string
                df[col] = df[col].apply(lambda x: '' if pd.isna(x) else str(int(x)))

        # Save to temp file with UTF-8 encoding
        clean_path = csv_path.replace('.temp.csv', '') + ".clean.csv"
        df.to_csv(clean_path, index=False, encoding='utf-8')

        # Clean up temp file
        if csv_path.endswith('.temp.csv') and os.path.exists(csv_path):
            os.remove(csv_path)

        return clean_path

    def import_generated_table(self, dbname: str, dataset_name: str, table_name: str) -> bool:
        """Import generated CSV into table using psycopg2."""
        csv_path = os.path.join("expdir", dataset_name, table_name, f"{table_name}.drifted.csv")
        if not os.path.exists(csv_path):
            print(f"Generated file not found: {csv_path}")
            return False
        print(f"Importing generated {table_name}...", end=" ", flush=True)
        try:
            conn = self._get_connection(dbname)
            conn.autocommit = True
            cur = conn.cursor()

            # Get integer columns and clean CSV (always clean for string escape issues)
            int_columns = self._get_integer_columns(dbname, table_name)
            print(f"(int_columns={int_columns})...", end=" ", flush=True)
            import_path = self._clean_csv_for_import(csv_path, int_columns)
            print(f"(cleaned: {import_path})...", end=" ", flush=True)

            # Debug: print first 3 lines of cleaned file
            with open(import_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 3:
                        print(f"\n  DEBUG line {i}: {line.strip()[:200]}")
                    else:
                        break

            # Truncate table
            cur.execute(f"TRUNCATE TABLE {table_name} CASCADE")
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

    def run_job_queries(self, dbname: str, output_file: str) -> bool:
        print(f"\nRunning JOB queries on {dbname}...")
        if os.path.exists(output_file):
            os.remove(output_file)
        cmd = f"cd {self.BAO_DIR} && python3 run_test_queries.py " \
              f"--use_postgres --database_name {dbname} " \
              f"--query_dir ../../../{self.JOB_QUERY_DIR} " \
              f"--output_file ../../../{output_file} --db-port {self.port}"
        print(f"Command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=False)
        return result.returncode == 0

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

    def validate_single_table(self, dataset_name: str, table_name: str, threshold: float = 1.2) -> TableValidationResult:
        print(f"\n{'='*60}")
        print(f"Validating table: {table_name}")
        print(f"{'='*60}")
        if not self.import_generated_table(self.gen_db, dataset_name, table_name):
            # Import failed - mark as import_failed so we don't retry auto-tune
            return TableValidationResult(
                table_name=table_name, queries_run=0, gen_total_time=0, real_total_time=0,
                time_ratio=float('inf'), passed=False, threshold=threshold, import_failed=True
            )
        gen_output = f"validation_logs/{dataset_name}_{table_name}_gen.log"
        os.makedirs("validation_logs", exist_ok=True)
        self.run_job_queries(self.gen_db, gen_output)
        real_output = f"validation_logs/{dataset_name}_real.log"
        if not os.path.exists(real_output):
            print(f"\nRunning queries on real database {self.real_db}...")
            self.run_job_queries(self.real_db, real_output)
        gen_results = self.parse_query_results(gen_output)
        real_results = self.parse_query_results(real_output)
        print(f"\nComparing results for {table_name}:")
        total, passed_count, _ = self.compare_results(gen_results, real_results, threshold)
        self.restore_table(table_name)
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
        return result

    def validate_all_tables(self, dataset_name: str, threshold: float = 1.2) -> Dict[str, TableValidationResult]:
        print(f"\n{'#'*60}")
        print(f"Validating all generated tables for {dataset_name}")
        print(f"{'#'*60}")

        # Check if gen_db already exists
        if self.database_exists(self.gen_db):
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
        if not os.path.exists(real_output):
            print(f"\nCaching real database query results...")
            self.run_job_queries(self.real_db, real_output)
        results = {}
        failed_tables = []
        for table_name in generated_tables:
            result = self.validate_single_table(dataset_name, table_name, threshold)
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
    parser.add_argument("--gen-db", type=str, default="imdb_17v2_gen")
    parser.add_argument("--real-db", type=str, default="imdb_17v2")
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
