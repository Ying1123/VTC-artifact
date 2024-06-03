import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# from plot.plot_utils import plot
from visualize import (get_req_rate_over_time, get_throughput_over_time, get_service_over_time,
                       get_response_time_over_time, to_client_name,
                       get_acc_service_diff_over_time, get_service_diff_over_time,
                       get_acc_service_diff_with_first_client,
                       gen_quant_fairness_table,
                       get_overall_throughput,
                       FONTSIZE, MARKERSIZE, legend_x, legend_y, ylabel_x, ylabel_y)


distinct_colors = ["rosybrown", "sienna", "sandybrown", "gold", "darkgreen", "turquoise", "deepskyblue", "navy", "blue", "blueviolet", "purple", "magenta", "hotpink", "pink", "black", "gray", "olivedrab", "chartreuse", "plum", "violet", "indigo", "khaki", "honeydew", "lavender", "teal", "salmon", "linen"]


def plot(baslines, x, ys, x_label, y_label, figname):
    FONTSIZE = 20
    MARKERSIZE = 6
    legend_x = 0.5
    legend_y = 1.1
    ylabel_x = 0
    ylabel_y = 0.5
    markers = ['v','s','o','+','s','D', 'P','X']


    legends = []
    curves = []
    fig, ax = plt.subplots()
    max_y = 0
    for i, (name, y) in enumerate(zip(baslines, ys)):
        # if y[-3] is None and y[-2] is not None:
        #     print(i, y)
        curves.append(ax.plot(x, y, color=distinct_colors[i], markersize=MARKERSIZE)[0])
        legends.append(to_client_name(name))

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")

    ax.set_xlim(1)
    ax.set_ylim(0)
    ax.set_xlabel(x_label, fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=2, width=1)
    # ax.yaxis.set_major_formatter(y_format)
    # fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
    #            ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=18)
    fig.text(ylabel_x, ylabel_y, y_label, va='center', rotation='vertical', fontsize=21)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    plt.savefig(figname, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {figname}")


if __name__ == "__main__":
    workloads = ["real", "overload"]
    
    for workload in workloads:
        service_diffs = []
        if workload in ["real"]:
            baselines = ["VTC_profile", "VTC_oracle_profile", "VTC_predict_profile", "FCFS", "LCF",
                         "LShare/rpm30", "LShare/rpm20", "LShare/rpm5"]
        elif workload in ["overload"]:
            baselines = ["FCFS", "VTC_profile", "VTC_oracle_profile"]

        throughputs = []
        for baseline in baselines:
            path = f"../{baseline}/all_results_{workload}.jsonl"
            with open(path, "r") as f:
                line = f.readline()
                config = json.loads(line)["config"]
                result = json.loads(line)["result"]
                result["responses"] = [x for x in result["responses"] if x["first_token_latency"] != -1]
            # get data points
            responses = result["responses"]
            # T = max([response["req_time"] for response in responses])
            T = 600
            num_x = 20
            window = 60
            warmup = 30
            x_ticks = [(T - warmup) / num_x * i for i in range(num_x)]
            x_ticks = x_ticks[1:]

            users = sorted(list(set([response["adapter_dir"] for response in responses])))

            req_rate = get_req_rate_over_time(responses, T, window, x_ticks, users, warmup=warmup)

            print("\n\nbaseline", baseline)
            service_diff = get_service_diff_over_time(responses, T, window, x_ticks, users, req_rate,
                                                      warmup=warmup, func_type="profile")
            service_diffs.append(service_diff)

            throughputs.append(get_overall_throughput(result))

            baseline_name = baseline if "/" not in baseline else baseline.replace("/", "_")
            service = get_service_over_time(responses, T, window, x_ticks, users, func_type="profile")
            plot(users, x_ticks, service, "Time (s)", "Service",
                 f"revision_profile_h_{workload}_{baseline_name}_service")

            response_time = get_response_time_over_time(responses, T, window, x_ticks, users)
            plot(users, x_ticks, response_time, "Time (s)", "Response Time (s)",
                 f"revision_profile_h_{workload}_{baseline_name}_response_time")

        if workload in ["real", "overload"]:
            print(baselines)
            gen_quant_fairness_table(baselines, service_diffs, throughputs, f"revision_profile_h_{workload}_quant_fairness")

