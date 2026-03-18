# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import warnings
import shutil
from datetime import datetime

from typing import Dict, NoReturn

import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ts_benchmark.utils.get_file_name import get_unique_file_suffix
from ts_benchmark.report import report
from ts_benchmark.common.constant import CONFIG_PATH, THIRD_PARTY_PATH, ROOT_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend


sys.path.insert(0, THIRD_PARTY_PATH)

warnings.filterwarnings("ignore")


def str_to_bool(value: str) -> bool:
    """
    Converts a string to a boolean: True for 'True', '1', or 'T'; False for 'False', '0', or 'F'.
    """
    if value.lower() in ['true', '1', 't']:
        return True
    elif value.lower() in ['false', '0', 'f']:
        return False
    else:
        raise ValueError("Invalid boolean value. Please enter 'True' or 'False'.")


def build_data_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the data loader config from commandline arguments and configuration dict
    """
    data_config = config_data["data_config"]
    data_config["data_name_list"] = args.data_name_list
    if args.data_set_name is not None:
        data_config["data_set_name"] = args.data_set_name
    return data_config


def build_model_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the model config from commandline arguments and configuration dict
    """
    model_config = config_data.get("model_config", None)

    if args.adapter is not None:
        args.adapter = [None if item == "None" else item for item in args.adapter]
        if len(args.model_name) > len(args.adapter):
            args.adapter.extend([None] * (len(args.model_name) - len(args.adapter)))
    else:
        args.adapter = [None] * len(args.model_name)

    if args.model_hyper_params is not None:
        args.model_hyper_params = [
            None if item == "None" else item for item in args.model_hyper_params
        ]
        if len(args.model_name) > len(args.model_hyper_params):
            args.model_hyper_params.extend(
                [None] * (len(args.model_name) - len(args.model_hyper_params))
            )
    else:
        args.model_hyper_params = [None] * len(args.model_name)

    for adapter, model_name, model_hyper_params in zip(
        args.adapter, args.model_name, args.model_hyper_params
    ):
        model_config["models"].append(
            {
                "adapter": adapter,
                "model_name": model_name,
                "model_hyper_params": json.loads(model_hyper_params)
                if model_hyper_params is not None
                else {},
            }
        )

    return model_config


def build_evaluation_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the evaluation config from commandline arguments and configuration dict
    """
    evaluation_config = config_data["evaluation_config"]
    evaluation_config["save_path"] = args.save_path

    metric_list = []
    if args.metrics != "all" and args.metrics is not None:
        for metric in args.metrics:
            metric = json.loads(metric)
            metric_list.append(metric)
        evaluation_config["metrics"] = metric_list

    default_strategy_args = evaluation_config["strategy_args"]
    strategy_args_updates = (
        json.loads(args.strategy_args) if args.strategy_args else None
    )

    if strategy_args_updates is not None:
        default_strategy_args.update(strategy_args_updates)

    if args.seed is not None:
        default_strategy_args["seed"] = args.seed
    if args.save_true_pred is not None:
        default_strategy_args["save_true_pred"] = args.save_true_pred
    default_strategy_args["deterministic"] = args.deterministic

    return evaluation_config


def build_report_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the report config from commandline arguments and configuration dict
    """
    report_config = config_data["report_config"]
    report_config["aggregate_type"] = args.aggregate_type
    report_config["save_path"] = args.save_path

    return report_config


def init_worker(env: Dict) -> NoReturn:
    """
    An initializer function for each worker that does some global setup
    """
    sys.path.insert(0, THIRD_PARTY_PATH)
    torch.set_num_threads(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run_benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # script name
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Evaluation config file path",
    )

    parser.add_argument(
        "--data-name-list",
        type=str,
        nargs="+",
        default=None,
        help="List of series names entered by the user",
    )

    parser.add_argument(
        "--data-set-name",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset name names entered by the user,"
             "only takes effect when data_name_list is not specified",
    )

    # model_config
    parser.add_argument(
        "--adapter",
        type=str,
        nargs="+",
        default=None,
        help="Adapter used to adapt the method to our pipeline",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        required=True,
        help="The relative path of the model that needs to be evaluated",
    )
    parser.add_argument(
        "--model-hyper-params",
        type=str,
        nargs="+",
        default=None,
        help=(
            "The input parameters corresponding to the models to be evaluated "
            "should correspond one-to-one with the --model-name options."
        ),
    )

    # evaluation_config
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Evaluation metrics that need to be calculated",
    )
    parser.add_argument(
        "--strategy-args",
        type=str,
        default=None,
        help="Parameters required for evaluating strategies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed that is set before evaluating any model-series pair, "
             "by default, use the seed value in the config file"
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        default="full",
        choices=["full", "efficient", "none"],
        help="Specify the type of deterministic behavior for the algorithm. Options are: "
             "'full': Enables full deterministic mode. "
             "'efficient': Fixes only some seeds for efficiency. "
             "'none': No deterministic behavior is applied."
    )

    # evaluation engine
    parser.add_argument(
        "--eval-backend",
        type=str,
        default="sequential",
        choices=["sequential", "ray"],
        help="Evaluation backend, use ray for parallel evaluation",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="Number of cpus to use, only available in both backends",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="List of gpu devices to use, only available in ray backends",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of evaluation workers",
    )
    # TODO: should timeout be part of the configuration file?
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Time limit for each evaluation task, in seconds",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=100,
        help="Max tasks to run on a single worker when using parallel backends",
    )

    # report_config
    parser.add_argument(
        "--aggregate_type",
        default="mean",
        help="Select the baseline algorithm to compare",
    )

    parser.add_argument(
        "--report-method",
        type=str,
        default="csv",
        choices=[
            "dash",
            "csv",
        ],
        help="Presentation form of algorithm performance comparison results",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="The relative path for saving evaluation results, relative to the result folder",
    )

    parser.add_argument(
        "--save-true-pred",
        type=str_to_bool,
        default=None,
        help="If true, saves the model's prediction results "
             "and the true values in evaluation result file",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    torch.set_num_threads(3)
    with open(os.path.join(CONFIG_PATH, args.config_path), "r") as file:
        config_data = json.load(file)

    required_configs = [
        "data_config",
        "model_config",
        "evaluation_config",
        "report_config",
    ]
    for config_name in required_configs:
        if config_data.get(config_name) is None:
            raise ValueError(f"{config_name} is none")

    data_config = build_data_config(args, config_data)
    model_config = build_model_config(args, config_data)
    evaluation_config = build_evaluation_config(args, config_data)
    report_config = build_report_config(args, config_data)

    ParallelBackend().init(
        backend=args.eval_backend,
        n_workers=args.num_workers,
        n_cpus=args.num_cpus,
        gpu_devices=args.gpus,
        default_timeout=args.timeout,
        max_tasks_per_child=args.max_tasks_per_child,
        worker_initializers=[init_worker],
    )

    try:
        log_filenames = pipeline(
            data_config,
            model_config,
            evaluation_config,
        )

    finally:
        ParallelBackend().close(force=True)

    report_config["log_files_list"] = log_filenames
    if args.report_method == "csv":
        filename = get_unique_file_suffix()
        leaderboard_file_name = "test_report" + filename
        report_config["leaderboard_file_name"] = leaderboard_file_name
    report(report_config, report_method=args.report_method)

    # 生成汇总的 RESULTS.md
    generate_results_summary(args.save_path, args.model_name[0] if args.model_name else "Model")


def generate_results_summary(save_path: str, model_name: str) -> None:
    """
    从所有 test_report 文件生成汇总的 RESULTS.md
    """
    result_dir = os.path.join(ROOT_PATH, "result", save_path)

    # 查找所有 test_report CSV 文件
    if not os.path.exists(result_dir):
        return

    test_reports = [f for f in os.listdir(result_dir) if f.startswith("test_report") and f.endswith(".csv")]

    if not test_reports:
        return

    import csv

    # 收集每个 horizon 的指标
    horizon_results = {}  # {horizon: {"mse_norm": x, "mae_norm": x, "rmse_norm": x}}

    for report_file in test_reports:
        report_path = os.path.join(result_dir, report_file)
        try:
            with open(report_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # 跳过标题行

                for row in reader:
                    if len(row) < 3:
                        continue

                    # 提取 horizon（从第一列的JSON中）
                    strategy_args_str = row[0]
                    import re
                    match = re.search(r'"horizon"\s*:\s*(\d+)', strategy_args_str)
                    if not match:
                        continue

                    current_horizon = int(match.group(1))

                    # 提取指标名和值
                    metric_name = row[1].strip() if len(row) > 1 else ""
                    metric_value = row[2].strip() if len(row) > 2 else None

                    if metric_name in ["mse_norm", "mae_norm", "rmse_norm"]:
                        if current_horizon not in horizon_results:
                            horizon_results[current_horizon] = {}
                        horizon_results[current_horizon][metric_name] = metric_value

        except Exception as e:
            print(f"Warning: Failed to parse {report_file}: {e}")
            continue

    # 生成 RESULTS.md
    if horizon_results:
        # 按 horizon 排序
        sorted_horizons = sorted(horizon_results.keys())

        md_content = f"# {model_name} 实验结果\n\n"
        md_content += "## ETTh1 数据集\n\n"
        md_content += "| Horizon | MSE_norm | MAE_norm | RMSE_norm |\n"
        md_content += "|---------|----------|----------|-----------|\n"

        for h in sorted_horizons:
            results = horizon_results[h]
            mse = results.get("mse_norm", "N/A")
            mae = results.get("mae_norm", "N/A")
            rmse = results.get("rmse_norm", "N/A")
            md_content += f"|      {h} |   {mse} |   {mae} |    {rmse} |\n"

        # 添加配置说明（如果使用量子模块）
        if "quantum" in model_name.lower() or "DUET_quantum" in save_path:
            md_content += "\n## 配置说明\n\n"
            md_content += "- `use_quantum_block: true` - 启用量子 OTOC 块\n"
            md_content += "- 其他参数与原版 DUET 相同\n"

        md_content += "\n## 关于压缩文件\n\n"
        md_content += "输出目录中的 `.csv.tar.gz` 文件是预测结果（原始预测值），由 benchmark 框架自动压缩保存。\n"
        md_content += "如需查看具体预测值，可以解压：\n\n"
        md_content += f"```bash\ntar -xzf result/{save_path}/DUET.xxx.csv.tar.gz\n```\n"

        # 写入 RESULTS.md
        results_md_path = os.path.join(result_dir, "RESULTS.md")
        with open(results_md_path, 'w') as f:
            f.write(md_content)

        print(f"Results summary saved to: {results_md_path}")
