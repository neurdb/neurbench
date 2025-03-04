import re
import json

def extract_id_and_execution_time(input_file, output_file):
    results = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                # Extract the query ID (supports variations)
                id_match = re.match(r'q(\d+)([a-zA-Z]*)#####', line)  # Fix regex
                if not id_match:
                    print(f"Skipping line (ID not found): {line.strip()}")
                    continue
                query_id = int(id_match.group(1))  # Extract main query number
                suffix = id_match.group(2)  # Extract suffix (like 'e', 'f')

                # Extract the execution time
                execution_time_match = re.search(r'"Execution Time"\s*:\s*([\d.]+)', line)
                if execution_time_match:
                    execution_time_s = float(execution_time_match.group(1))
                else:
                    # Fallback: Extract from JSON structure if available
                    json_match = re.search(r'(\[.*\])', line)
                    if json_match:
                        try:
                            json_data = json.loads(json_match.group(1))
                            execution_time_s = None
                            if isinstance(json_data, list):
                                for obj in json_data:
                                    if isinstance(obj, dict) and "Execution Time" in obj:
                                        execution_time_s = float(obj["Execution Time"])
                                        break
                            if execution_time_s is None:
                                print(f"Skipping line (Execution Time not found in JSON): {line.strip()}")
                                continue
                        except json.JSONDecodeError:
                            print(f"Skipping line (JSON parsing failed): {line.strip()}")
                            continue
                    else:
                        print(f"Skipping line (Execution Time not found): {line.strip()}")
                        continue

                # Store the result, now including suffix
                results.append((query_id, suffix, execution_time_s))

            except Exception as e:
                print(f"Error processing line: {line.strip()}")
                print(f"Error: {e}")

    # Sort results first by query number, then by suffix
    results.sort(key=lambda x: (x[0], x[1]))  # Sorting considers numeric ID first, then suffix

    # Write sorted results to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for id1 in range(1, 34):  # Adjust range dynamically if needed
            found = False
            for id2, suffix, execution_time_s in results:
                if id2 == id1:
                    suffix_str = suffix if suffix else ""  # Handle cases where there is no suffix
                    outfile.write(f'q{id2}{suffix_str}##### {execution_time_s:.3f}\n')
                    found = True
            if not found:
                outfile.write(f'q{id1}##### failed\n')

if __name__ == '__main__':
    input_file = 'query_shifts/query_shift_01/table/q_test_imdb_query_shift_table_01.test'  # Change this to your actual input file
    output_file = 'output.txt'  # Change this to your desired output file
    extract_id_and_execution_time(input_file, output_file)
    print(f"Results have been written to {output_file}")
