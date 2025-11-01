import sys
import json
import os
import glob

# 找出指定文件夹下所有的json文件，按照文件名排序
folder_path = "/root/python/artifacts_download"
json_files = glob.glob(os.path.join(folder_path, "**/*.json"), recursive=True)
json_files.sort()

# print("找到的JSON文件:")
# for json_file in json_files:
#     print(json_file)

# json_files = ["/root/python/artifacts_download/gpu-mode_discord-cluster-manager/run_18480579448/run-result/result.json"]
def extract_data(use_benchmark):
    map = []
    board_name = "benchmark" if use_benchmark else "leaderboard"
    for filename in json_files:
        json_file = open(filename, "r")
        # print(json_file, flush=True)
        json_data = json.load(json_file)
        import numpy as np
        # print(json_data)
        try:
            data = json_data['runs'][board_name]['run']['result']
        except: 
            pass
            continue
        # print(data["benchmark.0.spec"], filename)
        # k: 7168; m: 64; n: 18432; seed: 1212; has_bias: False; world_size: 8
        if 'check' in data and  data['check'] == 'pass':
            which = data['benchmark.0.spec']
            which_run = None
            if "hidden_dim: 6144" in which:
                which_run = "a2a"
            elif "k: 7168; m: 64; n: 18432; seed: 1212; has_bias: False; world_size: 8" in which:
                which_run = "ag-gemm"
            else:
                which_run = "gemm-rs"
            assert which_run is not None, f"{which_run} {which}"

            cnt = 5 if which_run == "a2a" else 6
            time = [float(data[f'benchmark.{i}.mean'])/1000 for i in range(cnt)]
            best_time = [float(data[f'benchmark.{i}.best'])/1000 for i in range(cnt)]
            worst_time = [float(data[f'benchmark.{i}.worst'])/1000 for i in range(cnt)]
            geo_mean = np.prod(time) ** (1 / len(time))
            geo_mean_best = np.prod(best_time) ** (1 / len(best_time))
            geo_mean_worst = np.prod(worst_time) ** (1 / len(worst_time))
            # if geo_mean < 300:
            if which_run == "gemm-rs" and geo_mean > 450:
                continue
            if which_run == "ag-gemm" and geo_mean > 400:
                continue
            if which_run == "a2a" and geo_mean > 300:
                continue

           
            # print(json_data['runs']['leaderboard'])
            map.append((which_run, geo_mean, json_data['runs'][board_name]['start'], which_run,filename, "gen_mean",  "full:",  f"{geo_mean:.2f}",[f'{i:.2f}' for i in time], "best:", f"{geo_mean_best:.2f}", [f'{i:.2f}' for i in best_time], "worst:",
                        f"{geo_mean_worst:.2f}",  [f'{i:.2f}' for i in worst_time]))
                # print(which_run,filename, "gen_mean", f"{geo_mean:.2f}us")
    return map
            
import typer
import os


app = typer.Typer()
@app.command()
def run(
    use_benchmark: bool = typer.Option(False, help="Whether to use benchmark data"),
    with_time: bool = typer.Option(False, help="Whether to print time"),
    # full_tokens: bool = typer.Option(False, help="Whether to use full expert"),
):
    ret = extract_data(use_benchmark)
    if not with_time:
        ret.sort(key=lambda x: x[1])
    index_start = 3 if not with_time else 2
    for i in ret: 
        print(*i[index_start:])
    
if __name__ == "__main__":
    app()
