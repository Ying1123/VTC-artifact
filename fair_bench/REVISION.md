## VTC with Length Prediction (Synthetic)
2 clients:
```
python launch_server.py --scheduler vtc_fair
python run_exp.py --suite overload --output VTC/all_results_overload.jsonl

python launch_server.py --scheduler vtc_oracle
python run_exp.py --suite overload --output VTC_oracle/all_results_overload.jsonl

python launch_server.py --scheduler vtc_oracle --predict-range 0.5
python run_exp.py --suite overload --output VTC_pred_50/all_results_overload.jsonl
```

8 clients:
```
python launch_server.py --scheduler vtc_fair
python run_exp.py --suite overload-multi --output VTC/all_results_overload-multi.jsonl

python launch_server.py --scheduler vtc_oracle
python run_exp.py --suite overload-multi --output VTC_oracle/all_results_overload-multi.jsonl

python launch_server.py --scheduler vtc_oracle --predict-range 0.5
python run_exp.py --suite overload-multi --output VTC_pred_50/all_results_overload-multi.jsonl
```

## VTC with Length Prediction (Real)
```
python launch_server.py --num-adapter 50 --scheduler vtc_fair
python run_exp.py --suite real --output VTC/all_results_real.jsonl

python launch_server.py --num-adapter 50 --scheduler vtc_oracle
python run_exp.py --suite real --output VTC_oracle/all_results_real.jsonl

python launch_server.py --num-adapter 50 --scheduler vtc_len_predict
python run_exp.py --suite real --output VTC_predict/all_results_real.jsonl
```

other baselines:
```
python launch_server.py --num-adapter 50 --scheduler slora
python run_exp.py --suite real --output FCFS/all_results_real.jsonl

python launch_server.py --num-adapter 50 --scheduler lcf_fair
python run_exp.py --suite real --output LCF/all_results_real.jsonl

python launch_server.py --num-adapter 50 --enable-abort --scheduler lshare_fair --rate-limit 30
python run_exp.py --suite real --output LShare/rpm20/all_results_real.jsonl

python launch_server.py --num-adapter 50 --enable-abort --scheduler lshare_fair --rate-limit 20
python run_exp.py --suite real --output LShare/rpm20/all_results_real.jsonl

python launch_server.py --num-adapter 50 --enable-abort --scheduler lshare_fair --rate-limit 5
python run_exp.py --suite real --output LShare/rpm5/all_results_real.jsonl
```

Plot:
```
python plot_revision.py
```
