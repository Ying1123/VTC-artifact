import argparse
import json
import os
from matplotlib import pyplot as plt
FONTSIZE = 24

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    x_axis = ["5", "10", "15", "20", "30"]
    y_axis = []
    dirs = ["../LShare/rpm5", "../LShare/rpm10", "../LShare/rpm15", "../LShare/rpm20", "../LShare/rpm30", "../VTC"]
    for dir in dirs:
        with open(os.path.join(dir, "all_results_real.jsonl"), "r") as f:
            json_file = json.load(f)
        
        total_time = json_file["result"]["total_time"]
        total_work  = 0
        for r in json_file["result"]["responses"]:
            if r["first_token_latency"] != -1:
                total_work += r["prompt_len"] + r["output_len"]
        y_axis.append(total_work/total_time)
        print(total_work/total_time)
    
    # a threshold line
    plt.axhline(y=y_axis[-1], color='r', linestyle='--', label="VTC Throughput", marker="x", markersize=10)

    plt.plot(x_axis, y_axis[:-1], marker=".", label="RPM Throughput", markersize=10)
    plt.xlabel("Number of RPM threshold", fontsize=FONTSIZE)
    plt.ylabel("Throughput (Token/s)", fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE - 3)
    # set the font size of the x-axis and y-axis
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    
    figname = "../LShare/LShare_overall_throughput.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, required=True)
#     args = parser.parse_args()
# 
#     with open(args.input, "r") as f:
#         json_file = json.load(f)
#     
#     total_time = json_file["result"]["total_time"]
#     total_work  = 0
#     for r in json_file["result"]["responses"]:
#         if r["first_token_latency"] != -1:
#             total_work += r["output_len"]
#     print(total_work/total_time)
