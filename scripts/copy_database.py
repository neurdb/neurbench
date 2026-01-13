#!/usr/bin/env python3
"""
Copy database between PostgreSQL instances.
Supports reusing existing dump files.

Usage:
    python copy_database.py imdb 5430 imdb_copy 5500
    python copy_database.py imdb 5430 imdb_copy 5500 --dump-dir ./dumps
"""

import os
import sys
import argparse
import subprocess

# PostgreSQL connection defaults
PG_HOST = "172.28.176.110"
PG_USER = "postgres"
PG_PASSWORD = "postgres"


def run_cmd(cmd, env=None):
    """Run command and return success status."""
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0


def database_exists(dbname: str, port: int, host: str = PG_HOST) -> bool:
    """Check if database exists."""
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=PG_USER, password=PG_PASSWORD, dbname="postgres"
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result is not None
    except Exception as e:
        print(f"Error checking database: {e}")
        return False


def create_database(dbname: str, port: int, host: str = PG_HOST) -> bool:
    """Create database (drop if exists)."""
    import psycopg2
    from psycopg2 import sql

    try:
        conn = psycopg2.connect(
            host=host, port=port, user=PG_USER, password=PG_PASSWORD, dbname="postgres"
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Terminate existing connections and drop
        if database_exists(dbname, port, host):
            print(f"Dropping existing database {dbname}...")
            cur.execute("""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
            """, (dbname,))
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(dbname)))

        print(f"Creating database {dbname}...")
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


def dump_database(dbname: str, port: int, dump_file: str, force: bool = False, host: str = PG_HOST) -> bool:
    """Dump database to file. Skip if file exists and not force."""

    if os.path.exists(dump_file) and not force:
        print(f"Using existing dump file: {dump_file}")
        return True

    print(f"Dumping {dbname}@{port} to {dump_file}...")

    env = os.environ.copy()
    env["PGPASSWORD"] = PG_PASSWORD

    cmd = [
        "pg_dump",
        "-h", host,
        "-p", str(port),
        "-U", PG_USER,
        "-d", dbname,
        "-Fc",  # Custom format (compressed)
        "-f", dump_file,
    ]

    if run_cmd(cmd, env):
        size_mb = os.path.getsize(dump_file) / (1024 * 1024)
        print(f"Dump complete: {size_mb:.1f} MB")
        return True
    return False


def restore_database(dbname: str, port: int, dump_file: str, host: str = PG_HOST) -> bool:
    """Restore database from dump file."""

    if not os.path.exists(dump_file):
        print(f"Error: Dump file not found: {dump_file}")
        return False

    # Create empty database
    if not create_database(dbname, port, host):
        return False

    print(f"Restoring {dump_file} to {dbname}@{port}...")

    env = os.environ.copy()
    env["PGPASSWORD"] = PG_PASSWORD

    cmd = [
        "pg_restore",
        "-h", host,
        "-p", str(port),
        "-U", PG_USER,
        "-d", dbname,
        "--no-owner",
        "--no-privileges",
        "-j", "4",  # Parallel jobs
        dump_file,
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # pg_restore may return non-zero for warnings, check stderr for real errors
    # Ignore harmless "unrecognized configuration parameter" errors (version mismatch)
    if result.returncode != 0:
        stderr_lower = result.stderr.lower()
        # Check for real errors, not just config param warnings
        is_config_param_error = "unrecognized configuration parameter" in stderr_lower
        has_fatal_error = "fatal" in stderr_lower or "could not connect" in stderr_lower

        if has_fatal_error or (not is_config_param_error and "error" in stderr_lower):
            print(f"Error: {result.stderr}")
            return False
        elif result.stderr:
            print(f"Warning (ignored): {result.stderr[:200]}...")

    print("Restore complete!")
    return True


def analyze_database(dbname: str, port: int, host: str = PG_HOST):
    """Run ANALYZE on database."""
    import psycopg2

    print(f"Running ANALYZE on {dbname}...")
    try:
        conn = psycopg2.connect(
            host=host, port=port, user=PG_USER, password=PG_PASSWORD, dbname=dbname
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("ANALYZE")
        cur.close()
        conn.close()
        print("ANALYZE complete!")
    except Exception as e:
        print(f"Warning: ANALYZE failed: {e}")


def copy_database(
    src_db: str,
    src_port: int,
    dst_db: str,
    dst_port: int,
    dump_dir: str = "./db_dumps",
    force_dump: bool = False,
    host: str = PG_HOST,
):
    """
    Copy database from source to destination.

    Args:
        src_db: Source database name
        src_port: Source PostgreSQL port
        dst_db: Destination database name
        dst_port: Destination PostgreSQL port
        dump_dir: Directory to store dump files
        force_dump: Force re-dump even if file exists
        host: PostgreSQL host
    """
    print("=" * 60)
    print(f"Copy Database")
    print(f"  Source:      {src_db}@{src_port}")
    print(f"  Destination: {dst_db}@{dst_port}")
    print("=" * 60)

    # Check source exists
    if not database_exists(src_db, src_port, host):
        print(f"Error: Source database {src_db} does not exist on port {src_port}")
        return False

    # Prepare dump directory
    os.makedirs(dump_dir, exist_ok=True)
    dump_file = os.path.join(dump_dir, f"{src_db}_{src_port}.dump")

    # Dump source
    if not dump_database(src_db, src_port, dump_file, force_dump, host):
        return False

    # Restore to destination
    if not restore_database(dst_db, dst_port, dump_file, host):
        return False

    # Analyze
    analyze_database(dst_db, dst_port, host)

    print("\n" + "=" * 60)
    print(f"✓ Successfully copied {src_db}@{src_port} to {dst_db}@{dst_port}")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Copy database between PostgreSQL instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy imdb from port 5430 to port 5500
  python copy_database.py imdb 5430 imdb_copy 5500

  # Copy with custom dump directory
  python copy_database.py imdb 5430 imdb_copy 5500 --dump-dir ./my_dumps

  # Force re-dump (ignore existing dump file)
  python copy_database.py imdb 5430 imdb_copy 5500 --force-dump

  # Copy to same port with different name
  python copy_database.py imdb 5430 imdb_backup 5430
        """
    )

    parser.add_argument("src_db", help="Source database name")
    parser.add_argument("src_port", type=int, help="Source PostgreSQL port")
    parser.add_argument("dst_db", help="Destination database name")
    parser.add_argument("dst_port", type=int, help="Destination PostgreSQL port")
    parser.add_argument("--dump-dir", default="./db_dumps", help="Directory for dump files (default: ./db_dumps)")
    parser.add_argument("--force-dump", action="store_true", help="Force re-dump even if dump file exists")
    parser.add_argument("--host", default=PG_HOST, help=f"PostgreSQL host (default: {PG_HOST})")

    args = parser.parse_args()

    success = copy_database(
        src_db=args.src_db,
        src_port=args.src_port,
        dst_db=args.dst_db,
        dst_port=args.dst_port,
        dump_dir=args.dump_dir,
        force_dump=args.force_dump,
        host=args.host,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
