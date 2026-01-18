#!/usr/bin/env python3
"""
分析 bao_logs_all, lero_logs_all, balsa_logs_all 目录下的测试日志，输出每个数据集的执行时间

用法:
    python3 analyze_bao_logs.py              # 使用当前目录下的 *_logs_all 目录
    python3 analyze_bao_logs.py /path/to/dir # 指定基础目录
"""

import os
import re
import sys
import glob
from collections import defaultdict


def extract_bao_results(log_dir):
    """从 bao 日志文件中提取测试结果"""
    results = []

    for filepath in glob.glob(os.path.join(log_dir, '*test*job.log')):
        filename = os.path.basename(filepath)
        dataset = None
        time = None

        with open(filepath, 'r') as f:
            content = f.read()

        # Extract dataset
        match = re.search(r'^Dataset:\s+(\S+)', content, re.MULTILINE)
        if match:
            dataset = match.group(1)

        # Extract time
        match = re.search(r'Completed all queries, total time:\s+(\S+)', content)
        if match:
            time = match.group(1)

        if dataset and time:
            test_type = 'bao' if 'test_bao' in filename else 'pg'
            date_prefix = filename.split('_test_')[0]
            time_val = float(time.replace('ms', ''))
            results.append({
                'filename': filename,
                'dataset': dataset,
                'type': test_type,
                'time_ms': time_val,
                'date': date_prefix
            })

    return results


def extract_lero_results(log_dir):
    """从 lero 日志文件中提取测试结果"""
    results = []

    for filepath in glob.glob(os.path.join(log_dir, '*test_lero_output*.log')):
        filename = os.path.basename(filepath)

        # 从文件名提取 dataset: 20260116_173448_test_lero_output_imdb_2015_gen_job.log
        match = re.search(r'test_lero_output_(.+)_job\.log$', filename)
        if not match:
            continue
        dataset = match.group(1)

        # 读取文件内容
        with open(filepath, 'r') as f:
            content = f.read()

        total_time = None

        # 方式1: 优先查找 "Completed all queries, total time: XXXms" 格式
        match = re.search(r'Completed all queries, total time:\s+([\d.]+)ms', content)
        if match:
            total_time = float(match.group(1))
        else:
            # 方式2: 回退到逐行解析 "after writting write_latency_file"
            total_time = 0
            for line in content.split('\n'):
                if 'after writting write_latency_file' in line:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            total_time += float(parts[-1])
                        except ValueError:
                            pass

        if total_time and total_time > 0:
            date_prefix = filename.split('_test_lero')[0]
            results.append({
                'filename': filename,
                'dataset': dataset,
                'type': 'lero',
                'time_ms': total_time,
                'date': date_prefix
            })

    return results


def extract_balsa_results(log_dir):
    """从 balsa 日志文件中提取测试结果"""
    results = []

    for filepath in glob.glob(os.path.join(log_dir, '*test_balsa*.log')):
        filename = os.path.basename(filepath)

        # 从文件名提取数据集名，支持两种格式:
        # 格式1: 20260117_062756_test_balsa_imdb_2015_job.log
        # 格式2: 20260114_test_balsa_imdb_job.log
        match = re.match(r'(\d+_\d+)_test_balsa_(.+)_(\w+)\.log', filename)
        if not match:
            # 尝试另一种格式
            match = re.match(r'(\d+)_test_balsa_(.+)_(\w+)\.log', filename)
        if not match:
            continue

        date_prefix = match.group(1)
        dataset = match.group(2)

        # 读取文件内容，提取执行时间
        with open(filepath, 'r') as f:
            content = f.read()

        # 提取所有执行时间: "10a Execution time: 736.0 (predicted 727.5)"
        times = re.findall(r'(\w+) Execution time: ([\d.]+)', content)

        if times:
            total_time = sum(float(t[1]) for t in times)
            results.append({
                'filename': filename,
                'dataset': dataset,
                'type': 'balsa',
                'time_ms': total_time,
                'date': date_prefix
            })

    return results


def format_time(ms):
    """格式化时间，添加千位分隔符"""
    return f"{ms:,.0f}"


def print_comparison_table(results):
    """按数据集分组打印 PG vs Bao vs Balsa vs Lero 对比表"""
    # Group by (dataset, type) 并收集所有时间用于求平均
    times_by_dataset = defaultdict(lambda: defaultdict(list))
    for r in results:
        dataset = r['dataset']
        test_type = r['type']
        if r['time_ms'] > 0:  # 只统计有效的结果
            times_by_dataset[dataset][test_type].append(r['time_ms'])

    # 按 dataset 名字排序
    sorted_datasets = sorted(times_by_dataset.keys())

    # 表格绘制
    print()
    print("  ┌───────────────────────┬────────────┬─────────────┬─────────────┬─────────────┐")
    print("  │ 数据集                │ PG(s)      │ Bao(s)      │ Balsa(s)    │ Lero(s)     │")
    print("  ├───────────────────────┼────────────┼─────────────┼─────────────┼─────────────┤")

    for dataset in sorted_datasets:
        data = times_by_dataset[dataset]

        # 计算平均值
        pg_times = data.get('pg', [])
        bao_times = data.get('bao', [])
        balsa_times = data.get('balsa', [])
        lero_times = data.get('lero', [])

        pg_time = sum(pg_times) / len(pg_times) / 1000 if pg_times else 0
        bao_time = sum(bao_times) / len(bao_times) / 1000 if bao_times else 0
        balsa_time = sum(balsa_times) / len(balsa_times) / 1000 if balsa_times else 0
        lero_time = sum(lero_times) / len(lero_times) / 1000 if lero_times else 0

        pg_str = f"{pg_time:.2f}" if pg_time > 0 else "N/A"
        bao_str = f"{bao_time:.2f}" if bao_time > 0 else "N/A"
        balsa_str = f"{balsa_time:.2f}" if balsa_time > 0 else "N/A"
        lero_str = f"{lero_time:.2f}" if lero_time > 0 else "N/A"

        print(f"  │ {dataset:<21} │ {pg_str:>10} │ {bao_str:>11} │ {balsa_str:>11} │ {lero_str:>11} │")

    print("  └───────────────────────┴────────────┴─────────────┴─────────────┴─────────────┘")
    print(f"\n  共 {len(sorted_datasets)} 个数据集")


def print_summary(results):
    """打印汇总信息"""
    # Group by (dataset, type) 并收集所有时间用于求平均
    times_by_dataset = defaultdict(lambda: defaultdict(list))
    for r in results:
        dataset = r['dataset']
        test_type = r['type']
        if r['time_ms'] > 0:
            times_by_dataset[dataset][test_type].append(r['time_ms'])

    # 计算每个 (dataset, type) 的平均值，然后汇总
    totals = defaultdict(float)
    counts = defaultdict(int)

    for dataset, data in times_by_dataset.items():
        for test_type, times in data.items():
            if times:
                avg_time = sum(times) / len(times)
                totals[test_type] += avg_time
                counts[test_type] += 1

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │                    汇总统计                     │")
    print(f"  ├─────────────────────────────────────────────────┤")

    if totals['pg'] > 0:
        print(f"  │ PG 总计:    {counts['pg']:>3} 次, {totals['pg']/1000:>10.2f}s           │")
    if totals['bao'] > 0:
        print(f"  │ Bao 总计:   {counts['bao']:>3} 次, {totals['bao']/1000:>10.2f}s           │")
    if totals['balsa'] > 0:
        print(f"  │ Balsa 总计: {counts['balsa']:>3} 次, {totals['balsa']/1000:>10.2f}s           │")
    if totals['lero'] > 0:
        print(f"  │ Lero 总计:  {counts['lero']:>3} 次, {totals['lero']/1000:>10.2f}s           │")

    print(f"  └─────────────────────────────────────────────────┘")


def main():
    # 支持命令行参数指定目录
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
        bao_log_dir = base_dir
        lero_log_dir = base_dir.replace('bao_logs_all', 'lero_logs_all')
        balsa_log_dir = base_dir.replace('bao_logs_all', 'balsa_logs_all')
    else:
        base_dir = os.getcwd()
        bao_log_dir = os.path.join(base_dir, 'bao_logs_all')
        lero_log_dir = os.path.join(base_dir, 'lero_logs_all')
        balsa_log_dir = os.path.join(base_dir, 'balsa_logs_all')

    print(f"Bao 日志目录: {bao_log_dir}")
    print(f"Lero 日志目录: {lero_log_dir}")
    print(f"Balsa 日志目录: {balsa_log_dir}")

    # 提取结果
    results = []

    if os.path.exists(bao_log_dir):
        results.extend(extract_bao_results(bao_log_dir))

    if os.path.exists(lero_log_dir):
        results.extend(extract_lero_results(lero_log_dir))

    if os.path.exists(balsa_log_dir):
        results.extend(extract_balsa_results(balsa_log_dir))

    if not results:
        print("未找到测试日志文件")
        return

    # 打印对比表
    print_comparison_table(results)

    # 打印汇总
    print_summary(results)


if __name__ == '__main__':
    main()
