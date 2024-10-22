import psycopg2
import argparse
import re

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run EXPLAIN ANALYZE on SQL queries.")
parser.add_argument("--dbname", type=str, required=True, help="Name of the database to connect to")
args = parser.parse_args()

# Database configuration
PORT = 5432
HOST = "localhost"
USER = "postgres"
PASSWORD = "postgres"
DB = args.dbname
CONNECTION_STR = f"dbname={DB} user={USER} password={PASSWORD} host={HOST} port={PORT}"
TIMEOUT = 30000000

# File with SQL queries
SQL_FILE_PATH = "imdb_q_test_0.txt"  # Replace with the path to your SQL file

# Dictionary to store queries and execution times
queries = {}
execution_times = {}


# Function to split and sort the query IDs
def sort_sql_ids(sql_id):
    match = re.match(r"q(\d+)([a-z]+)", sql_id)
    if match:
        return int(match.group(1)), match.group(2)
    return sql_id

try:
    # Connect to PostgreSQL database
    connection = psycopg2.connect(CONNECTION_STR)
    connection.autocommit = True
    cursor = connection.cursor()

    # Read the SQL queries from the file and store in a dictionary
    with open(SQL_FILE_PATH, 'r') as file:
        lines = file.readlines()
    for line in lines:
        query_id, query = line.split("#####")
        query_id = query_id.strip()
        if query_id.startswith("q"):
            queries[query_id] = query.strip()

    # Sort queries by their SQL IDs
    sorted_queries = sorted(queries.items(), key=lambda x: sort_sql_ids(x[0]))
    print([ele[0] for ele in sorted_queries])
    # Execute the queries in sorted order
    NUM_EXECUTIONS = 1
    for query_id, query in sorted_queries:
        try:
            for i in range(NUM_EXECUTIONS):
                explain_query = f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {query}"
                cursor.execute(explain_query)
                explain_result = cursor.fetchall()[0][0]
                execution_time = explain_result[0].get("Execution Time", "N/A")
                execution_times[query_id] = execution_time
        except Exception as e:
            execution_times[query_id] = "failed"
            print(f"Error with {query_id}: {e}")

        # Print the execution time for each query
        print(f"Execution Time for {query_id}: {execution_times[query_id]} ms")

except psycopg2.Error as e:
    print("Error: Unable to connect or execute the SQL query")
    print(e)

finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()
