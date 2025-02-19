import json
import pandas as pd
import pickle
import os
import plotly.express as px


def relevant_pairs_iterator(filename, input_jsons: list[str]):
    for json_file in input_jsons:
        # Extract file name from full path
        file_base = os.path.basename(json_file.replace(filename, ""))
        # Extract m, n, k from the filename
        _, m, n, k, hw_config, _ = file_base.split("_")
        cme = f"{filename}_{m}_{n}_{k}_cmes.pickle"
        f"{filename}_{m}_{n}_{k}_{hw_config}_traces.json"

        yield ((m, n, k, hw_config), json_file, cme)


def aggregate_results(filename, input_jsons: list[str], input_cmes: list[str]):
    data_list = []
    # Iterate over input files and extract relevant values
    for params, json_file, pickle_file in relevant_pairs_iterator(
        filename, input_jsons
    ):
        with open(json_file, "r") as f:
            json_data = json.load(f)
            extracted_cycles = json_data[0][0][2]["cycles"]

        with open(pickle_file, "rb") as f:
            cmes = pickle.load(f)
            zigzag_cycles = int(cmes[0][0].latency_total2)
        # Extract m, n, k from the filename
        m, n, k, hw_config = params
        data_list.append(
            {
                "n": int(n),
                "m": int(m),
                "k": int(k),
                "rtl cycles": extracted_cycles,
                "zz cycles": zigzag_cycles,
                "hw config": hw_config,
            }
        )

    df = pd.DataFrame(data_list)
    print(df)
    df.to_hdf(f"{filename}_results.hd5", key="results")


def plot_results(filename: str):
    df = pd.read_hdf(f"{filename}_results.hd5")
    # Create an interactive scatter plot
    fig = px.scatter(
        df,
        x="rtl cycles",
        y="zz cycles",
        hover_data=["n", "m", "k"],
        facet_col="hw config",
        title="RTL vs Estimated Cycles",
        labels={"rtl cycles": "RTL Cycles", "zz cycles": "Estimated Cycles (ZZ)"},
        template="plotly_white",
    )
    # Add y = x reference line on all subplots
    for facet in fig.data:
        subplot_idx = facet.xaxis.replace("x", "")  # Get the subplot index

        # Get min and max values for this specific subplot's data
        x_data = facet.x
        min_val = min(x_data) if len(x_data) > 0 else min(df["rtl cycles"])
        max_val = max(x_data) if len(x_data) > 0 else max(df["rtl cycles"])

        fig.add_shape(
            type="line",
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
            line=dict(color="gray", dash="dash"),
            xref=f"x{subplot_idx}" if subplot_idx else "x",
            yref=f"y{subplot_idx}" if subplot_idx else "y",
        )
    # export figure
    fig.write_html(f"{filename}_plot.html")  # Saves an interactive HTML file
