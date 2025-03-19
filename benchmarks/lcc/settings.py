import re

access_dist_exp_settings = [
    {
        "opts": "r,r,r,w,r,r,r,w,r,r",
        "access": "0,0,0,1,0,0,0,0,0,0",
        "policy": "./policies/pattern1.txt"
    },
    {
        "opts": "r,r,r,w,r,r,r,w,r,r",
        "access": "1,0,0,1,1,0,0,0,0,0",
        "policy": "./policies/pattern2.txt"
    },
    {
        "opts": "r,r,r,w,r,r,r,w,r,r",
        "access": "0,0,0,1,0,1,0,1,1,0",
        "policy": "./policies/pattern3.txt"
    },
    {
        "opts": "r,r,r,w,r,r,r,w,r,r",
        "access": "1,1,0,0,1,0,0,1,1,0",
        "policy": "./policies/pattern4.txt"
    },
]

skewness_exp_settings = [
    {
        "opts": "r,r,r,r,r,w,w,w,w,w",
        "access": "0,0,0,0,0,0,0,0,0,0"
    },
    {
        "opts": "r,r,r,r,r,w,w,w,w,w",
        "access": "0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
    },
    {
        "opts": "r,r,r,r,r,w,w,w,w,w",
        "access": "0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4"
    },
    {
        "opts": "r,r,r,r,r,w,w,w,w,w",
        "access": "0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6"
    },
    {
        "opts": "r,r,r,r,r,w,w,w,w,w",
        "access": "0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8"
    },
    {
        "opts": "r,r,r,r,r,w,w,w,w,w",
        "access": "1,1,1,1,1,1,1,1,1,1"
    },
]

txn_length_exp = [
    {
        "opts": "w",
        "access": "1"
    },
    {
        "opts": "r,w",
        "access": "1,1"
    },
    {
        "opts": "r,r,w,w",
        "access": "1,1,1,1"
    },
    {
        "opts": "r,r,r,r,w,w,w,w",
        "access": "1,1,1,1,1,1,1,1"
    },
    {
        "opts": "r,r,r,r,r,r,r,r,w,w,w,w,w,w,w,w",
        "access": "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    },
]

rw_rate_exp = [
    {
        "opts": "r,r,r,r,r,r,r,r,r,r",
        "access": "1,1,1,1,1,1,1,1,1,1"
    },
    {
        "opts": "r,r,r,r,r,r,r,r,w,w",
        "access": "1,1,1,1,1,1,1,1,1,1"
    },
    {
        "opts": "r,r,r,r,r,r,w,w,w,w",
        "access": "1,1,1,1,1,1,1,1,1,1"
    },
    {
        "opts": "r,r,r,r,w,w,w,w,w,w",
        "access": "1,1,1,1,1,1,1,1,1,1"
    },
    {
        "opts": "r,r,w,w,w,w,w,w,w,w",
        "access": "1,1,1,1,1,1,1,1,1,1"
    },
    {
        "opts": "w,w,w,w,w,w,w,w,w,w",
        "access": "1,1,1,1,1,1,1,1,1,1"
    },
]


def update_setting(current_value: str):
    file_path = "./training/chop_helper.py"

    # The line we are looking to replace (or a unique part of it)
    assert len(current_value) % 2 == 1
    pattern = r'dist_opts_str\s*=\s*"[rw,]*"'
    # The new line we want to replace it with
    new_line = f'dist_opts_str = "{current_value}"'

    # Read the entire file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace the old line with the new one
    modified_content = re.sub(pattern, new_line, content)

    # Write back the modified content to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print("Loading setting complete.")
