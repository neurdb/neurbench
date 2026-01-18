#!/usr/bin/env python3
"""
导出每个查询的执行时间到CSV文件
列: drift, system_name, query_name, execution_time

只包含指定的数据集: imdb, imdb_1_gen, imdb_3_gen, imdb_5_gen
"""

import os
import re
import sys
import glob
import csv
from collections import defaultdict


# 只处理这些数据集
TARGET_DATASETS = {'imdb', 'imdb_1_gen', 'imdb_3_gen', 'imdb_5_gen'}

# Balsa 特殊映射: 用 imdb_2017_gen 数据代替 imdb_5_gen
BALSA_DATASET_MAPPING = {
    'imdb_2017_gen': 'imdb_5_gen',  # imdb_2017_gen -> imdb_5_gen
}
# Balsa 需要跳过的原始数据集 (被映射覆盖)
BALSA_SKIP_DATASETS = {'imdb_5_gen'}  # 跳过原始 imdb_5_gen，只用映射的 imdb_2017_gen

# Dataset name mapping for output (可选的美化映射)
DATASET_DISPLAY_NAMES = {
    'imdb': 'imdb',
    'imdb_1_gen': 'imdb-1',
    'imdb_3_gen': 'imdb-3',
    'imdb_5_gen': 'imdb-5',
}


def extract_bao_query_times(log_dir):
    """从 Bao 日志和 txt 文件中提取每个查询的执行时间"""
    results = []

    # 首先从 log 文件获取 dataset 和对应的 txt 文件
    for log_path in glob.glob(os.path.join(log_dir, '*test*job.log')):
        filename = os.path.basename(log_path)

        with open(log_path, 'r') as f:
            content = f.read()

        # 提取 dataset
        match = re.search(r'^Dataset:\s+(\S+)', content, re.MULTILINE)
        if not match:
            continue
        dataset = match.group(1)

        if dataset not in TARGET_DATASETS:
            continue

        # 确定是 bao 还是 pg
        if 'test_bao' in filename:
            system = 'bao'
        elif 'test_pg' in filename:
            system = 'pg'
        else:
            continue

        # 提取对应的 txt 文件路径
        match = re.search(r'Results saved to:\s+(\S+\.txt)', content)
        if not match:
            # 尝试从 output 行提取
            match = re.search(r'output:\s+(\S+\.txt)', content)
        if not match:
            continue

        txt_path = match.group(1)
        # 转换为本地路径
        txt_filename = os.path.basename(txt_path)
        local_txt_path = os.path.join(log_dir, txt_filename)

        if not os.path.exists(local_txt_path):
            continue

        # 解析 txt 文件中的每个查询时间
        with open(local_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(', ')
                if len(parts) < 6:
                    continue

                # 提取查询名
                query_path = parts[3]
                query_match = re.search(r'/(\w+)\.sql$', query_path)
                if not query_match:
                    continue
                query_name = query_match.group(1)

                # 提取执行时间 (第6列, index 5)
                try:
                    exec_time = float(parts[5])
                except (ValueError, IndexError):
                    continue

                results.append({
                    'dataset': dataset,
                    'system': system,
                    'query': query_name,
                    'time_ms': exec_time
                })

    return results


def extract_balsa_query_times(log_dir):
    """从 Balsa 日志文件中提取每个查询的执行时间"""
    results = []

    for filepath in glob.glob(os.path.join(log_dir, '*test_balsa*.log')):
        filename = os.path.basename(filepath)

        # 提取 dataset: 20260117_062756_test_balsa_imdb_2015_job.log
        match = re.match(r'\d+_\d+_test_balsa_(.+)_job\.log', filename)
        if not match:
            # 尝试另一种格式
            match = re.match(r'\d+_test_balsa_(.+)_job\.log', filename)
        if not match:
            continue

        dataset = match.group(1)

        # Balsa: 跳过被映射覆盖的原始数据集
        if dataset in BALSA_SKIP_DATASETS:
            continue

        # Balsa 特殊映射
        if dataset in BALSA_DATASET_MAPPING:
            dataset = BALSA_DATASET_MAPPING[dataset]

        if dataset not in TARGET_DATASETS:
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        # 提取每个查询的执行时间: "10a Execution time: 736.0 (predicted 727.5)"
        for match in re.finditer(r'^(\w+) Execution time:\s+([\d.]+)', content, re.MULTILINE):
            query_name = match.group(1)
            exec_time = float(match.group(2))

            results.append({
                'dataset': dataset,
                'system': 'balsa',
                'query': query_name,
                'time_ms': exec_time
            })

    return results


def extract_lero_query_times(log_dir):
    """从 Lero 日志文件中提取每个查询的执行时间"""
    results = []

    for filepath in glob.glob(os.path.join(log_dir, '*test_lero_output*.log')):
        filename = os.path.basename(filepath)

        # 提取 dataset: 20260116_183005_test_lero_output_imdb_1_gen_job.log
        match = re.search(r'test_lero_output_(.+)_job\.log$', filename)
        if not match:
            continue

        dataset = match.group(1)
        if dataset not in TARGET_DATASETS:
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        # 提取每个查询的执行时间: "after writting write_latency_file timestamp query_name time"
        # parts: [after, writting, write_latency_file, timestamp, query_name, time]
        for line in content.split('\n'):
            if 'after writting write_latency_file' not in line:
                continue

            parts = line.strip().split()
            if len(parts) < 6:
                continue

            try:
                query_name = parts[4]
                exec_time = float(parts[5])
            except (ValueError, IndexError):
                continue

            results.append({
                'dataset': dataset,
                'system': 'lero',
                'query': query_name,
                'time_ms': exec_time
            })

    return results


def aggregate_results(results):
    """聚合多次执行的结果，取平均值"""
    # key: (dataset, system, query) -> list of times
    grouped = defaultdict(list)

    for r in results:
        key = (r['dataset'], r['system'], r['query'])
        grouped[key].append(r['time_ms'])

    # 计算平均值
    aggregated = []
    for (dataset, system, query), times in grouped.items():
        avg_time = sum(times) / len(times)
        aggregated.append({
            'dataset': dataset,
            'system': system,
            'query': query,
            'time_ms': avg_time,
            'count': len(times)
        })

    return aggregated


def write_csv(results, output_path):
    """写入CSV文件"""
    # 排序: dataset, system, query
    def sort_key(r):
        # query排序: 先按数字，再按字母
        query = r['query']
        match = re.match(r'(\d+)(\w*)', query)
        if match:
            return (r['dataset'], r['system'], int(match.group(1)), match.group(2))
        return (r['dataset'], r['system'], 0, query)

    results.sort(key=sort_key)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['drift', 'system_name', 'query_name', 'execution_time'])

        for r in results:
            # 使用显示名称
            drift = DATASET_DISPLAY_NAMES.get(r['dataset'], r['dataset'])
            writer.writerow([drift, r['system'], r['query'], f"{r['time_ms']:.3f}"])

    print(f"CSV saved to: {output_path}")
    print(f"Total rows: {len(results)}")


def main():
    # 支持命令行参数
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.getcwd()

    bao_log_dir = os.path.join(base_dir, 'bao_logs_all')
    lero_log_dir = os.path.join(base_dir, 'lero_logs_all')
    balsa_log_dir = os.path.join(base_dir, 'balsa_logs_all')

    print(f"Bao 日志目录: {bao_log_dir}")
    print(f"Lero 日志目录: {lero_log_dir}")
    print(f"Balsa 日志目录: {balsa_log_dir}")
    print(f"目标数据集: {', '.join(sorted(TARGET_DATASETS))}")
    print()

    # 提取结果
    results = []

    if os.path.exists(bao_log_dir):
        bao_results = extract_bao_query_times(bao_log_dir)
        print(f"从 Bao 提取了 {len(bao_results)} 条记录")
        results.extend(bao_results)

    if os.path.exists(lero_log_dir):
        lero_results = extract_lero_query_times(lero_log_dir)
        print(f"从 Lero 提取了 {len(lero_results)} 条记录")
        results.extend(lero_results)

    if os.path.exists(balsa_log_dir):
        balsa_results = extract_balsa_query_times(balsa_log_dir)
        print(f"从 Balsa 提取了 {len(balsa_results)} 条记录")
        results.extend(balsa_results)

    if not results:
        print("未找到任何查询结果")
        return

    # 聚合多次执行的结果
    print()
    aggregated = aggregate_results(results)
    print(f"聚合后共 {len(aggregated)} 条唯一记录")

    # 统计每个 (dataset, system) 的查询数
    stats = defaultdict(int)
    for r in aggregated:
        stats[(r['dataset'], r['system'])] += 1

    print("\n各数据集/系统的查询数:")
    for (dataset, system), count in sorted(stats.items()):
        drift = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        print(f"  {drift} / {system}: {count} queries")

    # 输出CSV (输出到 scripts 目录)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'query_times.csv')
    print()
    write_csv(aggregated, output_path)

    # 输出每个 (dataset, system) 的总时间
    print("\n各数据集/系统的总执行时间:")
    totals = defaultdict(float)
    for r in aggregated:
        key = (r['dataset'], r['system'])
        totals[key] += r['time_ms']

    # 按 dataset 分组打印表格
    datasets = sorted(set(r['dataset'] for r in aggregated))
    system_names = ['pg', 'bao', 'balsa', 'lero']

    print("\n  ┌─────────────┬────────────┬────────────┬────────────┬────────────┐")
    print("  │ Dataset     │ PG(s)      │ Bao(s)     │ Balsa(s)   │ Lero(s)    │")
    print("  ├─────────────┼────────────┼────────────┼────────────┼────────────┤")

    for dataset in datasets:
        drift = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        row = [f"{drift:<11}"]
        for sname in system_names:
            total = totals.get((dataset, sname), 0)
            if total > 0:
                row.append(f"{total/1000:>10.2f}")
            else:
                row.append(f"{'N/A':>10}")
        print(f"  │ {row[0]} │ {row[1]} │ {row[2]} │ {row[3]} │ {row[4]} │")

    print("  └─────────────┴────────────┴────────────┴────────────┴────────────┘")


if __name__ == '__main__':
    main()
