## Instructions for Artifact Evaluation

Hardware and software requirements:
- One A10G(24GB) and one A100(80GB).
- Ubuntu installed with Pytorch 2.1.2 and Triton 2.1.0 would be recommended.

Install the S-LoRA runtime with VTC and baseline schedulers:
```
cd VTC-artifact
pip install -e .
```


## Section 5.2 Synthetic Workload
1. Launch the server with Llama-2-7b on an AWS instance of A10G(24GB) GPU using VTC scheduler.
```
cd VTC-artifact/fair_bench
python launch_server.py
```
Run experiments with synthetic workload.
```
# Figure 3 (around 25 mins)
python run_exp.py --suite overload --output VTC/all_results_overload.jsonl
# Figure 4 (around 13 mins)
python run_exp.py --suite proportional --output VTC/all_results_proportional.jsonl
# Figure 5
python run_exp.py --suite on_off_less --output VTC/all_results_on_off_less.jsonl
# Figure 6
python run_exp.py --suite on_off_overload --output VTC/all_results_on_off_overload.jsonl
# Figure 7
python run_exp.py --suite poisson_short_long --output VTC/all_results_poisson_short_long.jsonl
# Figure 8 (around 30 mins)
python run_exp.py --suite poisson_short_long_2 --output VTC/all_results_poisson_short_long_2.jsonl
# Figure 9
python run_exp.py --suite increase --output VTC/all_results_increase.jsonl
# Figure 10 (around 18 mins)
python run_exp.py --suite dist_shift --output VTC/all_results_dist_shift.jsonl
```

2. Launch the server with Llama-2-7b on an AWS instance of A10G(24GB) GPU using FCFS scheduler.
```
cd S-LoRA-fair/fair_bench
python launch_server.py --scheduler slora
```
Run experiments with synthetic workload.
```
# Figure 3 (around 25 mins)
python run_exp.py --suite overload --output FCFS/all_results_overload.jsonl
# Figure 7
python run_exp.py --suite poisson_short_long --output FCFS/all_results_poisson_short_long.jsonl
# Figure 8 (around 30 mins)
python run_exp.py --suite poisson_short_long_2 --output FCFS/all_results_poisson_short_long_2.jsonl
```

3. Launch the server with Llama-2-7b on an AWS instance of A10G(24GB) GPU using LCF scheduler.
```
cd S-LoRA-fair/fair_bench
python launch_server.py --scheduler lcf_fair
```
Run experiments with synthetic workload. For the baseline LCF, please **relaunch** the server for each experiment.
```
# Figure 10 (around 18 mins)
python run_exp.py --suite dist_shift --output LCF/all_results_dist_shift.jsonl
```

4. Plot the figure from the `jsonl` files.
```
cd S-LoRA-fair/fair_bench/plot
python plot_6.2.py
python plot_6.2_work_diff.py
```
Figure 3: `fair_bench/VTC/sec6.2_overload_service.pdf` and `fair_bench/plot/sec6.2_overload_acc_service_diff.pdf`

Figure 4: `fair_bench/VTC/sec6.2_proportional_service.pdf` and `fair_bench/VTC/sec6.2_proportional_response_time.pdf`

Figure 5: `fair_bench/VTC/sec6.2_on_off_less_service.pdf` and `fair_bench/VTC/sec6.2_on_off_less_response_time.pdf`

Figure 6: `fair_bench/VTC/sec6.2_on_off_overload_service.pdf` and `fair_bench/VTC/sec6.2_on_off_overload_response_time.pdf`

Figure 7: `fair_bench/VTC/sec6.2_poisson_short_long_service.pdf` and `fair_bench/plot/sec6.2_poisson_short_long_acc_service_diff.pdf`

Figure 8: `fair_bench/VTC/sec6.2_poisson_short_long_2_service.pdf` and `fair_bench/plot/sec6.2_poisson_short_long_2_acc_service_diff.pdf`

Figure 9: `fair_bench/VTC/sec6.2_increase_service.pdf` and `fair_bench/VTC/sec6.2_increase_response_time.pdf`

Figure 10: `fair_bench/VTC/sec6.2_dist_shift_service.pdf` and `fair_bench/LCF/sec6.2_dist_shift_service.pdf`


## Section 5.3 Real Workload
1. Launch the server with Llama-2-7b on an AWS instance of A10G(24GB) GPU using VTC scheduler.
```
cd S-LoRA-fair/fair_bench
python launch_server.py --num-adapter 50
```
Run experiments on a real trace from chat.lmsys.org.
```
python run_exp.py --suite real --output VTC/all_results_real.jsonl
```

2. Launch the server with Llama-2-7b on an AWS instance of A10G(24GB) GPU using RPM scheduler.
- Rate Limit 5:
  ```
  cd S-LoRA-fair/fair_bench
  python launch_server.py --num-adapter 50 --enable-abort --scheduler lshare_fair --rate-limit 5
  ```
  Run experiments on a real trace from chat.lmsys.org.
  ```
  python run_exp.py --suite real --output LShare/rpm5/all_results_real.jsonl
  ```

- Rate Limit 10:
  ```
  cd S-LoRA-fair/fair_bench
  python launch_server.py --num-adapter 50 --enable-abort --scheduler lshare_fair --rate-limit 10
  ```
  Run experiments on a real trace from chat.lmsys.org.
  ```
  python run_exp.py --suite real --output LShare/rpm10/all_results_real.jsonl
  ```

- Rate Limit 15:
  ```
  cd S-LoRA-fair/fair_bench
  python launch_server.py --num-adapter 50 --enable-abort --scheduler lshare_fair --rate-limit 15
  ```
  Run experiments on a real trace from chat.lmsys.org.
  ```
  python run_exp.py --suite real --output LShare/rpm15/all_results_real.jsonl
  ```

- Rate Limit 20:
  ```
  cd S-LoRA-fair/fair_bench
  python launch_server.py --num-adapter 50 --enable-abort --scheduler lshare_fair --rate-limit 20
  ```
  Run experiments on a real trace from chat.lmsys.org.
  ```
  python run_exp.py --suite real --output LShare/rpm20/all_results_real.jsonl
  ```

3. Launch the server with Llama-2-7b on an AWS instance of A10G(24GB) GPU using FCFS scheduler.
```
cd S-LoRA-fair/fair_bench
python launch_server.py --num-adapter 50 --scheduler slora
```
Run experiments on a real trace from chat.lmsys.org.
```
python run_exp.py --suite real --output FCFS/all_results_real.jsonl
```

4. Plot
```
cd S-LoRA-fair/fair_bench/plot
python plot_6.3_real.py
python plot_6.3_calculate_stats_with_abort.py
```
Figure 11: `fair_bench/VTC/sec6.3_real_response_time.pdf`, `fair_bench/FCFS/sec6.3_real_response_time.pdf`

Figure 12: `fair_bench/LShare/rpm5/sec6.3_real_response_time.pdf`, `fair_bench/LShare/rpm10/sec6.3_real_response_time.pdf`, `fair_bench/LShare/rpm15/sec6.3_real_response_time.pdf`, `fair_bench/LShare/rpm20/sec6.3_real_response_time.pdf`

Figure 13: `fair_bench/LShare/LShare_overall_throughput.pdf`

**Note**: We lost the original trace file, so the current trace file is resampled from a different range, which is different with the one we used for the submission. The plots are then looks different with the ones in the paper. The trends and conclusion are maintained same.


## Sectino 5.4 Ablation Study
Launch the server with Llama-2-13b on a GCP instance of A100(80GB) GPU using VTC scheduler.
1. Use a memory pool of size 35000.
```
cd S-LoRA-fair/fair_bench
python launch_server.py --model-setting S4 --num-token 35000
```
Run experiments with synthetic workload.
```
# Figure 14 (VTC-256-35000)
python run_exp.py --suite overload-s4-35000-256 --output sec6.4/35000/VTC-256/all_results_overload-s4.jsonl
# Figure 14 (VTC-512-35000)
python run_exp.py --suite overload-s4-35000-512 --output sec6.4/35000/VTC-512/all_results_overload-s4.jsonl
# Figure 14 (VTC-768-35000)
python run_exp.py --suite overload-s4-35000-768 --output sec6.4/35000/VTC-768/all_results_overload-s4.jsonl
```

2. Use a memory pool of size 65000.
```
cd S-LoRA-fair/fair_bench
python launch_server.py --model-setting S4 --num-token 65000
```
Run experiments with synthetic workload.
```
# Figure 14 (VTC-512-65000)
python run_exp.py --suite overload-s4-long --output sec6.4/65000/VTC-512/all_results_overload-s4.jsonl
```

3. Plot
```
cd S-LoRA-fair/fair_bench/plot
python plot_6.4_work_diff.py
```
Figure 14: `fair_bench/plot/sec6.4_overload-s4_acc_service_diff_mem_pool.pdf` and `fair_bench/plot/sec6.4_overload-s4_acc_service_diff_req_len.pdf`


## Comments
The configurations for experiments are in `exp_suite.py`.

The experiments are run on the S-LoRA runtime, in which different users are identified by different LoRA adapters. But the LoRA computations have been turned off for a vanilla evaluation, since this paper has no assumption of using LoRA adapters. This is controlled by the `--no-lora` option in `launch_server.py`.

The real trace file is masked. The timestamps and sequence lengths are real, but the prompts are replaced with "Hello Hello Hello ..." for protecting privacy.

We lost the original trace file for real world (chatbot arena) experiments, so the current trace file is resampled from a different time range, which is different with the one we used for the submission. The plots are then looks different with the ones in the paper. The trends and conclusion are maintained same.
