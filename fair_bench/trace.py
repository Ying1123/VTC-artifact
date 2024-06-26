from collections import Counter
import json
import logging
from itertools import groupby
import numpy as np
from typing import List, Tuple, Any
from tqdm import tqdm
import random
from transformers import AutoTokenizer

class Request:
    def __init__(self, req_id, model_dir, adapter_dir, prompt, prompt_len, output_len, req_time):
        self.req_id = req_id
        self.model_dir = model_dir 
        self.adapter_dir = adapter_dir
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.req_time = req_time

    
    def __lt__(self, other):
        return self.req_time < other.req_time
    
    def __repr__(self):
        return f"req_id={self.req_id}, " \
               f"model_dir={self.model_dir}, adapter_dir={self.adapter_dir}, " \
               f"prompt_len={self.prompt_len}, output_len={self.output_len}, " \
               f"req_time={self.req_time}"


def dummy_prompt(prompt_len):
    return "Hello " * prompt_len


def generate_requests_increase(num_adapters, alpha, req_rate, cv, duration,
                               input_range, output_range, on_off, mode,
                               adapter_dirs, # (base_dir, adapter_dir))
                               seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    np.random.seed(seed)

    requests = []
    # generate for adapter 0
    tot_req = int(req_rate[0] * duration)
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
    tic = 0
    for i in range(tot_req):
        tic += 1 / req_rate[0]
        if on_off != -1 and int(tic // on_off) & 1 == 1:
            continue
        requests.append(Request(len(requests), adapter_dirs[0][0], adapter_dirs[0][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))
    # generate for adapter 1
    start_rate = 0.2
    end_rate = req_rate[1]
    tot_req = int(0.5 * (start_rate + end_rate) * duration)
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
    tic = 0
    for i in range(tot_req):
        current_rate = start_rate + (end_rate - start_rate) * tic / duration
        tic += 1 / current_rate
        requests.append(Request(len(requests), adapter_dirs[1][0], adapter_dirs[1][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))

    requests = sorted(requests)
    return requests


def generate_requests_uniform(num_adapters, alpha, req_rate, cv, duration,
                              input_range, output_range, on_off, mode,
                              adapter_dirs, # (base_dir, adapter_dir))
                              seed=42):
    assert num_adapters == len(req_rate)
    np.random.seed(seed)

    requests = []
    for i in range(num_adapters):
        tot_req = int(req_rate[i] * duration)
        input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
        output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
        tic = np.random.rand() * 1 / req_rate[i]
        for j in range(tot_req):
            tic += 1 / req_rate[i]
            if on_off != -1 and i == 0 and int(tic // on_off) & 1 == 1:
                continue
            requests.append(Request(len(requests), adapter_dirs[i][0], adapter_dirs[i][1],
                                    dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
                                    tic))
    requests = sorted(requests)
    return requests


def generate_requests_poisson_short_long(num_adapters, alpha, req_rate, cv, duration,
                                         input_range, output_range, on_off, mode,
                                         adapter_dirs, # (base_dir, adapter_dir))
                                         seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    np.random.seed(seed)

    tot_req = int(sum(req_rate) * duration)

    # generate adapter id
    probs = np.random.rand(tot_req)
    ind = (probs > (req_rate[0] / (req_rate[0] + req_rate[1]))).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / sum(req_rate)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        if ind[i] == 0:
            if on_off != -1 and int(tic // on_off) & 1 == 1:
                continue
            input_len = input_lens[i] // 4
            output_len = output_lens[i] // 4
        else:
            input_len = input_lens[i]
            output_len = output_lens[i]
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_len), int(input_len), int(output_len),
                                tic))
    return requests


def generate_requests_poisson_short_long_2(num_adapters, alpha, req_rate, cv, duration,
                                           input_range, output_range, on_off, mode,
                                           adapter_dirs, # (base_dir, adapter_dir))
                                           seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    np.random.seed(seed)

    tot_req = int(sum(req_rate) * duration)

    # generate adapter id
    probs = np.random.rand(tot_req)
    ind = (probs > (req_rate[0] / (req_rate[0] + req_rate[1]))).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / sum(req_rate)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        if ind[i] == 0:
            if on_off != -1 and int(tic // on_off) & 1 == 1:
                continue
            input_len = output_lens[i]
            output_len = input_lens[i]
        else:
            input_len = input_lens[i]
            output_len = output_lens[i]
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_len), int(input_len), int(output_len),
                                tic))
    return requests


def generate_requests_poisson(num_adapters, alpha, req_rate, cv, duration,
                              input_range, output_range, on_off, mode,
                              adapter_dirs, # (base_dir, adapter_dir))
                              seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    np.random.seed(seed)

    tot_req = int(sum(req_rate) * duration)

    # generate adapter id
    probs = np.random.rand(tot_req)
    ind = (probs > (req_rate[0] / (req_rate[0] + req_rate[1]))).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / sum(req_rate)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        if ind[i] == 0 and on_off != -1 and int(tic // on_off) & 1 == 1:
            continue
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))
    return requests


def generate_requests_dist_shift(num_adapters, alpha, req_rate, cv, duration,
                                 input_range, output_range, on_off, mode,
                                 adapter_dirs, # (base_dir, adapter_dir))
                                 seed=42):
    assert num_adapters == 2 and len(req_rate) == 2
    assert req_rate == [-1, -1]
    np.random.seed(seed)

    requests = []

    # on_off phase
    req_rate = [0.5, 2]
    on_off = 60
    for i in range(num_adapters):
        tot_req = int(req_rate[i] * duration / 3)
        input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
        output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
        tic = np.random.rand() * 1 / req_rate[i]
        for j in range(tot_req):
            tic += 1 / req_rate[i]
            if on_off != -1 and i == 0 and int(tic // on_off) & 1 == 1:
                continue
            requests.append(Request(len(requests), adapter_dirs[i][0], adapter_dirs[i][1],
                                    dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
                                    tic))

    # overload phase
    req_rate = [1, 1]
    on_off = -1
    for i in range(num_adapters):
        tot_req = int(req_rate[i] * duration / 3)
        input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
        output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
        tic = duration / 3 + np.random.rand() * 1 / req_rate[i]
        for j in range(tot_req):
            tic += 1 / req_rate[i]
            if on_off != -1 and i == 0 and int(tic // on_off) & 1 == 1:
                continue
            requests.append(Request(len(requests), adapter_dirs[i][0], adapter_dirs[i][1],
                                    dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
                                    tic))

    # proportional phase
    req_rate = [0.5, 1.5]
    on_off = -1
    for i in range(num_adapters):
        tot_req = int(req_rate[i] * duration / 3)
        input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
        output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
        tic = duration / 3 * 2 + np.random.rand() * 1 / req_rate[i]
        for j in range(tot_req):
            tic += 1 / req_rate[i]
            if on_off != -1 and i == 0 and int(tic // on_off) & 1 == 1:
                continue
            requests.append(Request(len(requests), adapter_dirs[i][0], adapter_dirs[i][1],
                                    dummy_prompt(input_lens[j]), int(input_lens[j]), int(output_lens[j]),
                                    tic))

    requests = sorted(requests)
    return requests


def generate_requests(num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range, on_off, mode,
                      adapter_dirs, # (base_dir, adapter_dir)
                      seed=42):
    if mode == "increase":
        return generate_requests_increase(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs, seed)
    elif mode == "uniform":
        return generate_requests_uniform(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs, seed)
    elif mode == "poisson-short-long":
        return generate_requests_poisson_short_long(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs, seed)
    elif mode == "poisson-short-long-2":
        return generate_requests_poisson_short_long_2(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs, seed)
    elif mode == "poisson":
        return generate_requests_poisson(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs, seed)
    elif mode == "dist_shift":
        return generate_requests_dist_shift(
                num_adapters, alpha, req_rate, cv, duration,
                input_range, output_range, on_off, mode, adapter_dirs, seed)

    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_adapters).astype(int)

    # generate input output len
    input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))
    return requests


def get_real_requests(trace_file, req_rate, duration, base_model, adapter_dirs, input_range, output_range, seed=42):
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    conversations = downsample(trace_file, req_rate, duration, tokenizer, input_range, output_range)
    model_mapping = generate_model_mapping(conversations, adapter_dirs)
    conversations = sort_and_rescale_by_req_time(conversations, duration)
    reqs = parse_into_req(base_model, conversations, model_mapping, tokenizer)
    return list(model_mapping.values()), reqs


# functions below are used to generate real requests
def downsample(json_file, req_rate, duration, tokenizer, input_range, output_range):
    with open(json_file, "r") as file:
       all_conversations = json.load(file)
    
    more_ratio = 2
    need_num = int(req_rate * duration)
    # sample a bit more than needed
    selected_indicies = np.random.choice(len(all_conversations), more_ratio * need_num, replace=False)
    downsampled_conversations = [all_conversations[idx] for idx in selected_indicies]
    for idx in reversed(range(len(downsampled_conversations))):
        conv = downsampled_conversations[idx]
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        if prompt_len >= input_range[1] or output_len >= output_range[1]:
            downsampled_conversations.pop(idx)
        
    downsampled_conversations = downsampled_conversations[:need_num]
    print(f"Downsampled {len(downsampled_conversations)}")
    return downsampled_conversations 

def generate_model_mapping(conversations, adapter_dirs):
    model_mapping = {}
    num_ranks = [0] * len(adapter_dirs)
    for conv in conversations:
        model = conv["model"]
        if model not in model_mapping.keys():
            adapter_dir = random.choice(adapter_dirs)
            name = f"{adapter_dir}-{num_ranks[adapter_dirs.index(adapter_dir)]}"
            num_ranks[adapter_dirs.index(adapter_dir)] += 1
            model_mapping[model] = name
    print(model_mapping)
    return model_mapping


def sort_and_rescale_by_req_time(conversations, duration):
    # sort first
    sorted_conversations = sorted(conversations, key=lambda d: d['tstamp']) 
    interval_start = sorted_conversations[0]["tstamp"]
    interval_end = sorted_conversations[-1]["tstamp"]
    # print(f"sorted time step: {[s['tstamp'] for s in sorted_conversations]}")

    for conv in conversations:
        tstamp = conv["tstamp"]
        assert interval_start <= tstamp and tstamp <= interval_end
        rescaled_tstamp = (tstamp - interval_start) / (interval_end - interval_start) * duration
        conv["tstamp"] = rescaled_tstamp
    return sorted_conversations 


def parse_into_req(base_model, conversations, model_mapping, tokenizer):
    reqs = []
    for idx, conv in enumerate(conversations):
        model = conv["model"]
        name = model_mapping[model]
        # print(conv["conversation"][0]["content"])
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        
        req = Request(req_id=idx, model_dir=base_model, adapter_dir=name, 
              prompt=dummy_prompt(prompt_len), prompt_len=prompt_len,
              output_len=output_len, req_time=conv["tstamp"])
        reqs.append(req)
    # print(reqs)
    return reqs

