import argparse
import os
import psutil
import sys

from exp_suite import BASE_MODEL, LORA_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-setting", type=str, default="S1")

    parser.add_argument("--num-adapter", type=int, default=10)
    parser.add_argument("--num-token", type=int, default=10000)

    parser.add_argument("--scheduler", type=str, default="vtc_fair")
    parser.add_argument("--fair-weights", type=int, default=[], action="append")
    parser.add_argument("--enable-abort", action="store_true")
    parser.add_argument("--rate-limit", type=int, default=None)
    parser.add_argument("--predict-range", type=float, default=0)
    args = parser.parse_args()

    base_model = BASE_MODEL[args.model_setting]
    adapter_dirs = LORA_DIR[args.model_setting]

    cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
    cmd += f" --model {base_model}"
    cmd += f" --tokenizer_mode auto"
    cmd += " --dummy"
    cmd += " --swap"
    cmd += f" --scheduler {args.scheduler}"
    cmd += " --no-lora"

    if args.enable_abort:
        cmd += " --enable-abort"
    if args.rate_limit is not None:
        cmd += f" --rate-limit {args.rate_limit}"
    cmd += f" --predict-range {args.predict_range}"

    num_iter = args.num_adapter // len(adapter_dirs) + 1
    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            cmd += f" --lora {adapter_dir}-{i}"
    for x in args.fair_weights:
        cmd += f" --fair-weights {x}"

    # print(cmd)
    os.system(cmd)
