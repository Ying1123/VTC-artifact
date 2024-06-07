import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# from plot.plot_utils import plot
from visualize import (get_req_rate_over_time, get_throughput_over_time, get_service_over_time,
                       get_response_time_over_time, to_client_name,
                       FONTSIZE, MARKERSIZE, legend_x, legend_y, ylabel_x, ylabel_y)


distinct_colors = ["rosybrown", "sienna", "sandybrown", "gold", "darkgreen", "turquoise", "deepskyblue", "navy", "blue", "blueviolet", "purple", "magenta", "hotpink", "pink", "black", "gray", "olivedrab", "chartreuse", "plum", "violet", "indigo", "khaki", "honeydew", "lavender", "teal", "salmon", "linen"]
def plot(names, x, ys, x_label, y_label, figname):
    FONTSIZE = 26
    MARKERSIZE = 5
    legend_x = 0.42
    legend_y = 1.1
    ylabel_x = -0.1
    ylabel_y = 0.5

    legends = []
    curves = []
    fig, ax = plt.subplots()
    for i, (name, y) in enumerate(zip(names, ys)):
        y = np.array([np.nan if v is None else v for v in y])
        # if max([kk for kk in y if kk is not None]) > 100 and "response_time" in figname:
        #    print(y)
        if len(names) > 4:
            curves.append(ax.plot(x, y, color=distinct_colors[i], markersize=MARKERSIZE)[0])
        else:
            curves.append(ax.plot(x, y, color=f"C{i}", markersize=MARKERSIZE)[0])
        legends.append(to_client_name(name))

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.0f}")

    #ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(0)
    ax.set_ylim(0)
    if "rpm" in figname:
        ax.set_ylim(0, 60)
    if "rpm5" in figname:
        ax.set_ylim(0, 1)
    if "rpm10" in figname or "rpm15" in figname:
        ax.set_ylim(0, 15)
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=2, width=1)
    ax.yaxis.set_major_formatter(y_format)
    #fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
    #           ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=FONTSIZE)
    fig.text(ylabel_x, ylabel_y, y_label, va='center', rotation='vertical', fontsize=FONTSIZE)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {figname}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--method", type=str, default="VTC")
    # args = parser.parse_args()

    methods = ["VTC", "FCFS", "LShare/rpm5", "LShare/rpm10", "LShare/rpm15", "LShare/rpm20", "LShare/rpm30"]
    for method in methods:
        json_file = f"../{method}/all_results_real.jsonl"
        if not os.path.exists(json_file): continue
        exps = []
        with open(json_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                exps.append({})
                exps[-1]["config"] = json.loads(line)["config"]
                exps[-1]["result"] = json.loads(line)["result"]

        # get data points
        for exp in exps:
            config = exp["config"]
            result = exp["result"]

            responses = result["responses"]
            T = max([response["req_time"] for response in responses])
            T = int(T) / 10 * 10
            num_x = 100
            window = 60
            x_ticks = [T / num_x * i for i in range(num_x)]

            users = sorted(list(set([response["adapter_dir"] for response in responses])))
            cnt = {}
            for user_name in users:
                cnt[user_name] = 0
            for response in responses:
                cnt[response["adapter_dir"]] += 1
            # print(cnt)
            sorted_cnt = [key for key, value in sorted(cnt.items(), key=lambda x: x[1])]

            req_rate = get_req_rate_over_time(responses, T, window, x_ticks, users)
            plot(users, x_ticks, req_rate, "Time (s)", "Request Rate (token/s)", f"../{method}/sec6.3_real_req_rate")
            total_req_rate = np.array(req_rate[0])
            for i in range(1, len(req_rate)):
                total_req_rate += np.array(req_rate[i])
            plot([None], x_ticks, [total_req_rate], "Time (s)", "Request Rate (token/s)",
                 f"../{method}/sec6.3_real_total_req_rate")

            if "rpm5" not in method:
                users = [sorted_cnt[i] for i in [12, 13, 25, 26]]
            # print(users)

            throughput = get_throughput_over_time(responses, T, window, x_ticks, users)
            service = get_service_over_time(responses, T, window, x_ticks, users)
            response_time = get_response_time_over_time(responses, T, window, x_ticks, users)

        # plot
        # plot(users, x_ticks, throughput, "Time (s)", "Throughput (token/s)", f"../{args.method}/sec6.3_real_throughput")
        # plot(users, x_ticks, service, "Time (s)", "Service (Token/s)", f"../{args.method}/sec6.3_real_service")
        plot(users, x_ticks, response_time, "Time (s)", "Response Time (s)", f"../{method}/sec6.3_real_response_time")


