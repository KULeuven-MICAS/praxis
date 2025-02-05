import json
import pandas as pd
import re
import glob


def aggregate_results(filename: str):
    FILENAME_PATTERN = filename + r"_(\d+)_(\d+)_(\d+)_traces\.json"
    json_files = glob.glob(f"{filename}_*_traces.json")  # Adjust the path
    data_list = []
    for file in json_files:
        match = re.search(FILENAME_PATTERN, file)
        if match:
            n, m, k = map(int, match.groups())  # Extract n, m, k as integers
            with open(file, "r") as f:
                json_data = json.load(f)
                extracted_cycles = json_data[0][0][2]["cycles"]
            data_list.append({"n": n, "m": m, "k": k, "cycles": extracted_cycles})
    df = pd.DataFrame(data_list)
    df.to_csv(f"{filename}_results.csv", index=False)
    return df


if __name__ == "__main__":
    df = aggregate_results("quantized_matmul")
    print(df)
