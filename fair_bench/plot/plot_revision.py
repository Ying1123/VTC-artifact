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


def cost_func(input_len, output_len):
    return input_len + 2*output_len


def plot(baslines, x, ys, x_label, y_label, figname):
    FONTSIZE = 20
    MARKERSIZE = 8
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
        max_y = max(max_y, max(y))
        curves.append(ax.plot(x, y, color=f"C{i}", marker=markers[i], markersize=MARKERSIZE)[0])
        legends.append(to_client_name(name))

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")

    ax.set_xlim(1)
    ax.set_ylim(0)
    ax.set_xlabel(x_label, fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=2, width=1)
    # ax.yaxis.set_major_formatter(y_format)
    fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
               ncol=len(legends) // min(2, len(legends) // 4 + 1), fontsize=18)
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
    # baselines = ['VTC','LCF','FCFS']
    workloads = ["overload", "overload-multi", "real", "overload-weighted"]
    
    for workload in workloads:
        acc_services_diffs = []
        service_diffs = []
        if workload in ["overload", "overload-multi"]:
            baselines = ["VTC", "VTC_pred_50", "VTC_oracle"]
        elif workload in ["real"]:
            baselines = ["VTC", "VTC_oracle", "VTC_predict", "FCFS", "LCF",
                         "LShare/rpm30", "LShare/rpm20", "LShare/rpm5"]
        elif workload in ["overload-weighted"]:
            baselines = ["VTC", "WVTC"]

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

            acc_service_delta = get_acc_service_diff_with_first_client(responses, T, window, x_ticks, users, warmup=warmup)
            if workload != "real":
                plot(users, x_ticks, acc_service_delta, "Time (s)", "Accumulated service diff with client 0",
                     f"{workload}_{baseline}_acc_service_diff")

            req_rate = get_req_rate_over_time(responses, T, window, x_ticks, users, warmup=warmup)
            if workload != "real":
                plot(users, x_ticks, req_rate, "time progression (s)", "req_rate (token/s)",
                     f"{workload}_req_rate")
 
            acc_service_diff = get_acc_service_diff_over_time(responses, T, window, x_ticks, users, warmup=warmup)
            acc_services_diffs.append(acc_service_diff)
            print("\n\nbaseline", baseline)
            service_diff = get_service_diff_over_time(responses, T, window, x_ticks, users, req_rate, warmup=warmup)
            service_diffs.append(service_diff)

            throughputs.append(get_overall_throughput(result))
            if workload in ["overload-weighted"]:
                service = get_service_over_time(responses, T, window, x_ticks, users)
                plot(users, x_ticks, service, "Time (s)", "Service",
                     f"revision_{workload}_{baseline}_service")

        # plot
        if workload in ["overload", "overload-multi"]:
            plot(baselines, x_ticks, acc_services_diffs, "Time (s)",
                 "Absolute Difference in Service", f"revision_{workload}_acc_service_diff")
        if workload in ["overload", "overload-multi", "real"]:
            gen_quant_fairness_table(baselines, service_diffs, throughputs, f"{workload}_quant_fairness")

