import json
import pandas as pd
import pickle
import os
import plotly.express as px


def aggregate_results(filename, input_jsons: list, input_cmes: list):
    data_list = []
    # Iterate over input files and extract relevant values
    for json_file, pickle_file in zip(input_jsons, input_cmes):
        with open(json_file, "r") as f:
            json_data = json.load(f)
            extracted_cycles = json_data[0][0][2]["cycles"]

        with open(pickle_file, "rb") as f:
            cmes = pickle.load(f)
            zigzag_cycles = int(cmes[0][0].latency_total2)
        bare_extension = os.path.basename(json_file.replace(filename, ""))
        # Extract m, n, k from the filename
        _, m, n, k, _ = bare_extension.split("_")  # Split filename
        data_list.append(
            {
                "n": int(n),
                "m": int(m),
                "k": int(k),
                "rtl cycles": extracted_cycles,
                "zz cycles": zigzag_cycles,
            }
        )

    df = pd.DataFrame(data_list)
    df.to_hdf(f"{filename}_results.hd5", key="results")


def plot_results(filename: str):
    df = pd.read_hdf(f"{filename}_results.hd5")
    # Create an interactive scatter plot
    fig = px.scatter(
        df,
        x="rtl cycles",
        y="zz cycles",
        hover_data=["n", "m", "k"],
        title="RTL vs Estimated Cycles",
        labels={"rtl cycles": "RTL Cycles", "zz cycles": "Estimated Cycles (ZZ)"},
        template="plotly_white",
    )
    # Add y = x reference line
    fig.add_shape(
        type="line",
        x0=min(df["rtl cycles"]),
        y0=min(df["rtl cycles"]),  # pyright: ignore
        x1=max(df["rtl cycles"]),
        y1=max(df["rtl cycles"]),  # pyright: ignore
        line=dict(color="gray", dash="dash"),
    )
    # Export the figure
    fig.write_html(f"{filename}_plot.html")  # Saves an interactive HTML file
