import numpy as np
import os
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Queue
from tqdm import tqdm

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def run(engine, batch_size, input_len, output_len, warmup=True):
    # warm up
    test_data = np.vstack([np.arange(1, input_len+1) for _ in range(batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data).cuda()
    
    if warmup:
        warmup_token = min(8, output_len)
        b_loc = torch.zeros(batch_size, input_len + warmup_token, dtype=torch.long, device="cuda")
        b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            b_loc[i, 0:input_len] = i * input_len + torch.arange(0, input_len, dtype=torch.int32, device="cuda")
            b_start_loc[i] = i * input_len
            b_seq_len[i] = input_len

        total_token_num = input_len * batch_size
        logics = engine.forward(batch_size, 
                                total_token_num, 
                                input_len, 
                                test_data,
                                b_loc,
                                b_start_loc,
                                b_seq_len,
                                is_prefill=True)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

        for i in range(warmup_token):
            b_loc[:, input_len + i] = total_token_num + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            total_token_num += batch_size
            b_seq_len += 1
            logics = engine.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
                predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False)
            prob_out = torch.softmax(logics, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            predict_ids = predict_ids.detach().cpu().numpy()
        
        max_len_in_batch = input_len + warmup_token
        for i in range(batch_size):
            engine.mem_manager.free(b_loc[i, max_len_in_batch - b_seq_len[i]:max_len_in_batch])
        print("can use mem size:", engine.mem_manager.can_use_mem_size)
            
        b_loc = None
        b_start_loc = None
        b_seq_len = None

    # start profiling
    import torch.distributed as dist
    dist.barrier()
    torch.cuda.synchronize()

    # profile prefill
    prefill_start_time = time.time()

    b_loc = torch.zeros(batch_size, input_len + output_len, dtype=torch.long, device="cuda")
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = batch_size * input_len
    logics = engine.forward(batch_size, total_token_num, input_len, test_data,
                            b_loc, b_start_loc, b_seq_len, is_prefill=True)
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    torch.cuda.synchronize()

    prefill_time = time.time() - prefill_start_time
    decode_start_time = time.time()

    # profile decode
    for i in range(output_len):
        torch.cuda.synchronize()
        step_start = time.time()
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1

        logics = engine.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
            predict_ids).cuda().reshape(-1), b_loc, b_start_loc, b_seq_len, is_prefill=False)
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        torch.cuda.synchronize()
        if i % 100 == 0:
            print(i, "step decode time:", time.time() - decode_start_time)

    torch.cuda.synchronize()
    end_time = time.time()

    total_decode_time = time.time() - decode_start_time
    
    max_len_in_batch = input_len + output_len
    for i in range(batch_size):
        engine.mem_manager.free(b_loc[i, max_len_in_batch - b_seq_len[i]:max_len_in_batch])
    print("can use mem size:", engine.mem_manager.can_use_mem_size)
    
    return prefill_time, total_decode_time
 

def profile(model):
    # input_lens = [8] + list(range(50, 512, 50))
    input_lens = [8, 16, 32, 64, 128, 256, 512]
    focus_input = [8, 64, 256, 512]
    output_lens = [1, 64, 128, 196, 256]
    ret = defaultdict(dict)

    torch.cuda.synchronize()
    for input_len in tqdm(input_lens, desc="input_len"):
        for output_len in tqdm(output_lens, desc="output_len"):
            bs = model.model.mem_manager.can_use_mem_size // (input_len + output_len)
            # bs = 12
            repeat = 5 if output_len == 1 else 1
            prefill_times = []
            for k in range(repeat):
                prefill_time, decode_time = run(model.model, bs, input_len, output_len)
                prefill_times.append(prefill_time)
            prefill_time = np.mean(prefill_times) / bs
            decode_time /= bs
            ret[input_len][output_len] = (bs, prefill_time, decode_time)
            print(f"input_len={input_len}, output_len={output_len}, bs={bs}: "
                  f"prefill_time={prefill_time:.2f}, decode_time={decode_time:.2f}")
            if input_len not in focus_input:
                break

    return ret


def plot(names, x, ys, x_label, y_label, figname):
    FONTSIZE = 20
    MARKERSIZE = 8
    legend_x = 0.5
    legend_y = 1.1
    ylabel_x = -0.1
    ylabel_y = 0.5
    markers = ['v','s','o','+','s','D', 'P','X']

    legends = []
    curves = []
    fig, ax = plt.subplots()
    for i, (name, y) in enumerate(zip(names, ys)):
        curves.append(ax.plot(x, y, color=f"C{i}", marker=markers[i], markersize=MARKERSIZE)[0])
        legends.append(name)

    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color="black")
    y_format = StrMethodFormatter("{x:.1f}")

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(x_label, fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=2, width=1)
    ax.yaxis.set_major_formatter(y_format)
    if legends[0] is not None:
        fig.legend(curves, legends, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
                   ncol=4, fontsize=18)
    fig.text(ylabel_x, ylabel_y, y_label, va='center', rotation='vertical', fontsize=21)
    fig.subplots_adjust(wspace=0.2)

    # Save figure
    fig.set_size_inches((6, 4))
    figname = f"{figname}.pdf"
    plt.savefig(figname, bbox_inches="tight")
    print(f"Saved figure to {figname}")


if __name__ == "__main__":
    if os.path.exists("cost_profile.json"):
        with open("cost_profile.json", "r") as f:
            cost = json.loads(f.readline())
    else:
        import torch
        from slora.models.llama2.model import Llama2TpPartModel
        from slora.server.input_params import InputParams
        from slora.server.router.model_infer.model_rpc import start_model_process, ModelRpcServer
        sys.path.append("../test/model")
        from model_infer import tppart_model_infer

        # model_dir = "huggyllama/llama-7b"
        # model_class = Llama2TpPartModel
        # bs= 10
        # input_len = 256
        # output_len = 256
        # # not working for dummy model
        # tppart_model_infer(rank_id=0, world_size=1, ans_queue=Queue(),
        #                    model_dir=model_dir, model_class=model_class,
        #                    batch_size=bs, input_len=input_len, output_len=output_len)

        # profile on A10G (24GB)
        weight_dir = "huggyllama/llama-7b"
        max_total_token_num = 10000

        scheduler = "vtc_fair"
        max_req_total_len = 2048 + 1024 # default
        batch_max_tokens = int(1 / 6 * max_total_token_num)
        batch_max_tokens = max(batch_max_tokens, max_req_total_len)
        input_params = InputParams(max_req_total_len=max_req_total_len,
                                   # kv cache manager parameters
                                   max_total_token_num=max_total_token_num,
                                   pool_size_lora=0,
                                   batch_max_tokens=batch_max_tokens,
                                   running_max_req_size=1000,
                                   # heuristic
                                   swap=True,
                                   prefetch=False,
                                   prefetch_size=0,
                                   scheduler=scheduler,
                                   profile=False,
                                   batch_num_adapters=None,
                                   enable_abort=None,
                                   # mem_ratio=args.mem_ratio,
                                   dummy=True,
                                   no_lora_swap=False,
                                   no_lora_compute=False,
                                   no_kernel=False,
                                   no_mem_pool=False,
                                   bmm=False,
                                   no_lora=True,
                                   fair_weights=[1],
                                   rate_limit=None,
                                   predict_range=0,
                                  )

        model = ModelRpcServer()
        model.exposed_init_model(0, 1, weight_dir, adapter_dirs=[], max_total_token_num=max_total_token_num,
                load_way="HF", mode=[], input_params=input_params, prefetch_stream=None)

        tic = time.time()
        cost = profile(model)
        print(f"profiling takes time: {time.time() - tic:.2f}")

        with open("cost_profile.json", "w") as f:
            f.write(json.dumps(cost))
    print(cost)

    # Plot
    # cost[i][j]: (prefill time, decode time) for input length i and output length j
    x, y = [], []
    for input_len in cost.keys():
        x.append(int(input_len))
        y.append(cost[input_len]["1"][1])
    plot([None], x, [y], "Number of input tokens", "Prefill time (s)", "prefill_cost")

    x, ys = [], []
    names = []
    for input_key in cost.keys():
        if len(cost[input_key]) == 1: continue
        names.append(input_key)
        y = []
        for i, output_key in enumerate(cost[input_key].keys()):
            if len(ys) == 0:
                x.append(int(output_key))
            else:
                assert x[i] == int(output_key)
            bs, prefill_time, decode_time = cost[input_key][output_key]
            y.append(decode_time)
        ys.append(y)
    plot(names, x, ys, "Number of output tokens", "Decode time (s)", "decode_cost")

    # Get function
    x, y, h = [], [], []
    for input_key in cost.keys():
        for output_key in cost[input_key].keys():
            bs, prefill_time, decode_time = cost[input_key][output_key]
            x.append(int(input_key))
            y.append(int(output_key))
            h.append((prefill_time + decode_time) * 1000)

    def cost_func(X, a, b, c, d, e):
        x, y = X
        return a * x + b * y + c * x * y + d * y * y + e

    initial_guess = [1, 1, 1, 1, 1]
    popt, pcov = curve_fit(cost_func, (x, y), h, p0=initial_guess)
    print("Fitted parameters: a, b, c, d, e = ", popt)

    for i in range(len(x)):
        x_test, y_test = x[i], y[i]
        h_pred = cost_func((x_test, y_test), *popt)
        print(f"x, y, h, h_pred: {x_test}, {y_test}, {h[i]:.3f}, {h_pred:.3f}")
