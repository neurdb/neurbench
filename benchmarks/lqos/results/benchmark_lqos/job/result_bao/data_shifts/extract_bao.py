import re

# Define input and output files
input_file = 'data_v2_07'  # Update with your actual file
output_file = 'execution_times_output.txt'

# Dictionary to store execution times for each SQL file
execution_times = {}

# Updated regex to match the required format
log_pattern = re.compile(
    r".*, (\d+), .*?, (/data/datasets/imdb_v2/query/q_test_sample_20%/(\d+[a-z]*)\.sql), (\d+\.\d+), (\d+\.\d+), Bao")

# Read the input log file
with open(input_file, 'r') as file:
    for line in file:
        match = log_pattern.match(line.strip())
        if match:
            run_id = int(match.group(1))  # Run ID (0, 1, 2)
            sql_file = match.group(2)  # Full SQL file path
            sql_number = match.group(3)  # Extract SQL file number
            float_value = float(match.group(5))  # Extract the additional float value

            # Store only the last occurrence (run_id == 2)
            if run_id == 2:
                execution_times[sql_number] = (sql_file, float_value)

# Sort by SQL file number and write results
with open(output_file, 'w') as output:
    for sql_number in sorted(execution_times.keys(), key=lambda x: int(re.findall(r'\d+', x)[0])):
        sql_file, float_value = execution_times[sql_number]
        output.write(f"{sql_file}: {float_value:.3f} ms\n")

print(f"Extracted float values written to {output_file}.")
