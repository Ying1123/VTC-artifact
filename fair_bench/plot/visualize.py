import argparse
import json

from plot_utils import plot


FONTSIZE = 25
MARKERSIZE = 8
legend_x = 0.44
legend_y = 1.15
ylabel_x = -0.13
ylabel_y = 0.5


def to_client_name(adapter_name):
    if adapter_name == "dummy-lora-7b-rank-8-0":
        return "Client 1"
    if adapter_name == "dummy-lora-7b-rank-8-1":
        return "Client 2"
    if adapter_name == "dummy-lora-7b-rank-8-2":
        return "Client 3"


def get_req_rate_over_time(responses, T, window, x_ticks, users):
    y = []
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2
            r = x + window / 2
            for response in responses:
                if response["adapter_dir"] == user_name:
                    req_time = response["req_time"]
                    num_token = response["output_len"]
                    if l <= req_time and req_time <= r:
                        y[-1][i] += num_token
            y[-1][i] /= window
    return y


def get_throughput_over_time(responses, T, window, x_ticks, users):
    y = []
    sum = 0
    for r in responses:
        if r["first_token_latency"] != -1:
            sum += 1
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2
            r = x + window / 2
            for response in responses:
                if response["adapter_dir"] == user_name:
                    start_time = response["req_time"] + response["first_token_latency"]
                    end_time = response["req_time"] + response["request_latency"]
                    num_token = response["output_len"]
                    overlap = max(min(r, end_time) - max(l, start_time), 0)
                    if end_time == start_time: # for aborted requests
                        assert response["first_token_latency"] == -1, "first_token_latency should be -1 for aborted requests"
                        y[-1][i] += 0
                    else:
                        y[-1][i] += num_token * overlap / (end_time - start_time)
            y[-1][i] /= window
    return y

def cost_func(input_len, output_len):
    return input_len + 2*output_len

def get_service_over_time(responses, T, window, x_ticks, users):
    y = []
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2
            r = x + window / 2
            for response in responses:
                if response["adapter_dir"] == user_name:
                    start_time = response["req_time"] + response["first_token_latency"]
                    end_time = response["req_time"] + response["request_latency"]
                    service = cost_func(response['prompt_len'],response["output_len"])
                    overlap = max(min(r, end_time) - max(l, start_time), 0)
                    y[-1][i] += service * overlap / (end_time - start_time)
            y[-1][i] /= window
    return y


def get_response_time_over_time(responses, T, window, x_ticks, users):
    y = []
    for i, user_name in enumerate(users):
        y.append([0] * len(x_ticks))
        for i, x in enumerate(x_ticks):
            l = x - window / 2
            r = x + window / 2
            cnt = 0
            for response in responses:
                if response["adapter_dir"] == user_name:
                    req_time = response["req_time"] + response["first_token_latency"]
                    response_time = response["first_token_latency"]
                    if l <= req_time and req_time <= r:
                        if not response_time == -1: # for aborted requests
                            y[-1][i] += response_time
                            cnt += 1
            if cnt == 0:
                y[-1][i] = None
            else:
                y[-1][i] /= cnt
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    args = parser.parse_args()

    exps = []
    with open(args.input, "r") as f:
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

        users = list(set([response["adapter_dir"] for response in responses]))

        req_rate = get_req_rate_over_time(responses, T, window, x_ticks, users)
        service = get_service_over_time(responses, T, window, x_ticks, users)
        throughput = get_throughput_over_time(responses, T, window, x_ticks, users)
        response_time = get_response_time_over_time(responses, T, window, x_ticks, users)

    # plot
    plot(users, x_ticks, req_rate, "time progression (s)", "req_rate (token/s)", "req_rate")
    plot(users, x_ticks, throughput, "time progression (s)", "throughput (token/s)", "throughput")
    plot(users, x_ticks, service, "time progression (s)", "service (token/s)", "service")
    plot(users, x_ticks, response_time, "time progression (s)", "response_time (s)", "response_time")

    cnt = {}
    for user_name in users:
        cnt[user_name] = 0

    for response in responses:
        cnt[response["adapter_dir"]] += 1
    print(cnt)