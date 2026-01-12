#!/usr/bin/env python3
"""
Load IMDB CSV files into PostgreSQL using psycopg2.
"""

import os
import sys
import csv
import argparse
import io
import psycopg2
from psycopg2 import sql
from pathlib import Path

# Tables in load order (respect foreign key dependencies)
TABLES = [
    # No dependencies
    "comp_cast_type",
    "company_type",
    "info_type",
    "kind_type",
    "link_type",
    "role_type",
    "keyword",
    # Depends on kind_type
    "title",
    "company_name",
    # Depends on title
    "aka_title",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "movie_companies",
    "complete_cast",
    # Depends on nothing special
    "name",
    "char_name",
    # Depends on name
    "aka_name",
    "person_info",
    # Depends on title, name, etc
    "cast_info",
]

# PostgreSQL connection
PG_HOST = "172.28.176.110"
PG_PORT = 5430
PG_USER = "postgres"
PG_PASSWORD = "postgres"

SCHEMA_FILE = "benchmarks/lqos/balsa/scripts/load-postgres/schema.sql"
FK_INDEX_FILE = "benchmarks/lqos/balsa/scripts/load-postgres/fkindexes.sql"
ADD_FK_FILE = "benchmarks/lqos/balsa/scripts/load-postgres/add_fks.sql"


def get_connection(dbname: str = "postgres"):
    """Get PostgreSQL connection."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=dbname
    )


def database_exists(dbname: str) -> bool:
    """Check if database exists."""
    conn = get_connection("postgres")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result is not None


def drop_database(dbname: str) -> bool:
    """Drop database if exists."""
    print(f"Dropping database {dbname}...")
    try:
        conn = get_connection("postgres")
        conn.autocommit = True
        cur = conn.cursor()
        # Terminate existing connections
        cur.execute("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
        """, (dbname,))
        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(dbname)))
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error dropping database: {e}")
        return False


def create_database(dbname: str) -> bool:
    """Drop and recreate database."""
    # Drop if exists
    if database_exists(dbname):
        if not drop_database(dbname):
            return False

    # Create new
    print(f"Creating database {dbname}...")
    try:
        conn = get_connection("postgres")
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


def run_sql_file(dbname: str, filepath: str) -> bool:
    """Execute SQL file."""
    try:
        with open(filepath, 'r') as f:
            sql_content = f.read()

        conn = get_connection(dbname)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(sql_content)
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error executing {filepath}: {e}")
        return False


class StreamingCSVReader:
    """
    Streaming CSV cleaner that reads and cleans on-the-fly.
    Implements file-like interface for copy_expert.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.file = None
        self.reader = None
        self.buffer = ""
        self.header = None
        self.row_count = 0
        self._init_reader()

    def _init_reader(self):
        self.file = open(self.csv_path, 'r', newline='', encoding='utf-8', errors='replace')
        # Check for backslash escaping
        sample = self.file.read(8192)
        self.file.seek(0)
        has_backslash_escape = '\\"' in sample

        if has_backslash_escape:
            self.reader = csv.reader(self.file, escapechar='\\', doublequote=True)
        else:
            self.reader = csv.reader(self.file)

        # Read header
        self.header = next(self.reader)

    def read(self, size=8192):
        """Read cleaned CSV data in chunks."""
        while len(self.buffer) < size:
            try:
                row = next(self.reader)
                self.row_count += 1
                # Write row to buffer as CSV
                output = io.StringIO()
                writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(row)
                self.buffer += output.getvalue()
            except StopIteration:
                break

        # Return requested chunk
        result = self.buffer[:size]
        self.buffer = self.buffer[size:]
        return result

    def close(self):
        if self.file:
            self.file.close()


def clean_and_load_csv(conn, table_name: str, csv_path: str) -> int:
    """
    Stream-clean CSV and load into table using COPY.
    Returns number of rows loaded.
    """
    stream = StreamingCSVReader(csv_path)

    try:
        cur = conn.cursor()
        columns = ', '.join([f'"{col}"' for col in stream.header])
        copy_sql = f'COPY {table_name} ({columns}) FROM STDIN WITH CSV'

        cur.copy_expert(copy_sql, stream)
        conn.commit()
        cur.close()

        return stream.row_count
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        stream.close()


def load_table(dbname: str, data_dir: str, table_name: str) -> bool:
    """Load a single table from CSV."""
    csv_path = os.path.join(data_dir, f"{table_name}.csv")

    if not os.path.exists(csv_path):
        print(f"  Skipping {table_name}: CSV not found")
        return True

    print(f"Loading {table_name}...", end=" ", flush=True)

    try:
        conn = get_connection(dbname)
        row_count = clean_and_load_csv(conn, table_name, csv_path)
        conn.close()
        print(f"OK ({row_count} rows)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Load IMDB CSV files into PostgreSQL")
    parser.add_argument("data_dir", help="Directory containing CSV files")
    parser.add_argument("dbname", help="Target database name")
    parser.add_argument("--tables", nargs="+", help="Specific tables to load (default: all)")
    parser.add_argument("--skip-schema", action="store_true", help="Skip schema creation")
    parser.add_argument("--skip-fk", action="store_true", help="Skip foreign key creation")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    # Create database
    if not create_database(args.dbname):
        sys.exit(1)

    # Create schema
    if not args.skip_schema:
        print("Creating schema...")
        if not run_sql_file(args.dbname, SCHEMA_FILE):
            sys.exit(1)
        print("Creating indexes...")
        if not run_sql_file(args.dbname, FK_INDEX_FILE):
            sys.exit(1)

    # Determine tables to load
    tables_to_load = args.tables if args.tables else TABLES

    # Filter to only existing CSV files
    available_tables = []
    for t in tables_to_load:
        csv_path = os.path.join(data_dir, f"{t}.csv")
        if os.path.exists(csv_path):
            available_tables.append(t)
        else:
            print(f"Warning: {t}.csv not found, skipping")

    print(f"\nLoading {len(available_tables)} tables from {data_dir}")
    print("=" * 60)

    # Load tables
    failed = []
    for table in available_tables:
        if not load_table(args.dbname, data_dir, table):
            failed.append(table)

    # Add foreign keys
    if not args.skip_fk and not failed:
        print("\nAdding foreign key constraints...")
        run_sql_file(args.dbname, ADD_FK_FILE)

    # Analyze
    print("\nRunning ANALYZE...")
    try:
        conn = get_connection(args.dbname)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("ANALYZE VERBOSE")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Warning: ANALYZE failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    if failed:
        print(f"Failed tables: {failed}")
        sys.exit(1)
    else:
        print(f"Successfully loaded {len(available_tables)} tables into {args.dbname}")


if __name__ == "__main__":
    main()
