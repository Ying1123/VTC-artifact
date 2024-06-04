import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def bar_plot(data, xlabel, ylabel, figname):
    unique, counts = np.unique(data, return_counts=True)
    frequency = dict(zip(unique, counts))

    plt.bar(frequency.keys(), frequency.values())
    plt.xlabel(xlabel, fontsize=26)
    plt.ylabel(ylabel, fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)

    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {figname}")


if __name__ == "__main__":
    path = f"../VTC/all_results_real.jsonl"

    with open(path, "r") as f:
        log = json.loads(f.readline())
    responses = log["result"]["responses"]
    input_lens = [x["prompt_len"] for x in responses]
    output_lens = [x["output_len"] for x in responses]
    # print(sorted(input_lens))
    # print(sorted(output_lens))
    print(np.mean(input_lens), np.min(input_lens), np.max(input_lens))
    print(np.mean(output_lens), np.min(output_lens), np.max(output_lens))

    bar_plot(input_lens, xlabel="Input length", ylabel="Frequency", figname="input_len_distribution")
    bar_plot(output_lens, xlabel="Output length", ylabel="Frequency", figname="output_len_distribution")

