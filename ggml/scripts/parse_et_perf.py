#!/usr/bin/env python3
"""
Parse ET_PERF performance logs to CSV format.

Usage:
    python parse_et_perf.py <log_file> [output.csv]

The script parses pipe-delimited ET_PERF logs in the format:
    ET_PERF|op=MUL_MAT|kernel=mul_mat_f32|duration_us=1234|tensor=output|shape=[4096,4096,1,1]|start_us=1000000|end_us=1001234|...

And outputs CSV with columns for all fields found in the logs.
Also outputs JSON summary for tracking optimization progress.
"""

import sys
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_et_perf_line(line):
    """Parse a single ET_PERF log line into a dictionary."""
    if 'ET_PERF|' not in line:
        return None

    # Extract the ET_PERF portion
    match = re.search(r'ET_PERF\|(.+)', line)
    if not match:
        return None

    perf_data = match.group(1).strip()

    # Parse pipe-delimited fields
    result = {}
    for field in perf_data.split('|'):
        field = field.strip()
        if '=' in field:
            key, value = field.split('=', 1)
            result[key] = value

    return result if result else None


def parse_llama_perf(log_file):
    """Extract llama performance metrics from log file."""
    perf_info = {}

    with open(log_file, 'r') as f:
        for line in f:
            # prompt eval time = 30733.87 ms / 10 tokens ( 3073.39 ms per token, 0.33 tokens per second)
            if 'prompt eval time' in line:
                match = re.search(r'prompt eval time\s+=\s+[\d.]+\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s*([\d.]+)\s+ms per token,\s+([\d.]+)\s+tokens per second', line)
                if match:
                    perf_info['pp_tokens'] = int(match.group(1))
                    perf_info['pp_ms_per_token'] = float(match.group(2))
                    perf_info['pp_tps'] = float(match.group(3))

            # eval time = 67594.83 ms / 3 runs (22531.61 ms per token, 0.04 tokens per second)
            elif 'eval time' in line and 'prompt' not in line:
                match = re.search(r'eval time\s+=\s+[\d.]+\s+ms\s+/\s+(\d+)\s+runs\s+\(\s*([\d.]+)\s+ms per token,\s+([\d.]+)\s+tokens per second', line)
                if match:
                    perf_info['gt_tokens'] = int(match.group(1))
                    perf_info['gt_ms_per_token'] = float(match.group(2))
                    perf_info['gt_tps'] = float(match.group(3))

    return perf_info


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else log_file.with_suffix('.csv')

    if not log_file.exists():
        print(f"Error: Log file '{log_file}' not found")
        sys.exit(1)

    # First pass: collect all unique field names
    all_fields = set()
    perf_entries = []

    with open(log_file, 'r') as f:
        for line in f:
            entry = parse_et_perf_line(line)
            if entry:
                all_fields.update(entry.keys())
                perf_entries.append(entry)

    if not perf_entries:
        print(f"No ET_PERF entries found in '{log_file}'")
        sys.exit(1)

    # Standard fields first, then alphabetically sorted additional fields
    standard_fields = ['op', 'kernel', 'duration_us', 'tensor', 'shape', 'start_us', 'end_us']
    extra_fields = sorted(all_fields - set(standard_fields))
    fieldnames = [f for f in standard_fields if f in all_fields] + extra_fields

    # Write CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for entry in perf_entries:
            writer.writerow(entry)

    # Parse llama performance metrics
    llama_perf = parse_llama_perf(log_file)

    # Print summary statistics
    print(f"Parsed {len(perf_entries)} ET_PERF entries from '{log_file}'")
    print(f"Output written to '{output_file}'")

    if llama_perf:
        print()
        print("LLaMA Performance Metrics:")
        if 'pp_tokens' in llama_perf and 'pp_tps' in llama_perf:
            print(f"  Prompt Processing:  {llama_perf['pp_tokens']:3d} tokens @ {llama_perf['pp_ms_per_token']:8.2f} ms/token ({llama_perf['pp_tps']:6.2f} tokens/sec)")
        if 'gt_tokens' in llama_perf and 'gt_tps' in llama_perf:
            print(f"  Token Generation:   {llama_perf['gt_tokens']:3d} tokens @ {llama_perf['gt_ms_per_token']:8.2f} ms/token ({llama_perf['gt_tps']:6.2f} tokens/sec)")

    print()

    # Compute statistics by operation
    op_stats = defaultdict(lambda: {'count': 0, 'total_us': 0, 'min_us': float('inf'), 'max_us': 0, 'total_flops': 0})

    for entry in perf_entries:
        op = entry.get('op', 'UNKNOWN')
        duration = int(entry.get('duration_us', 0))

        op_stats[op]['count'] += 1
        op_stats[op]['total_us'] += duration
        op_stats[op]['min_us'] = min(op_stats[op]['min_us'], duration)
        op_stats[op]['max_us'] = max(op_stats[op]['max_us'], duration)

        # Accumulate FLOPs if provided in log
        if 'flops' in entry:
            flops = int(entry['flops'])
            op_stats[op]['total_flops'] += flops

    # Overall totals
    total_count = sum(s['count'] for s in op_stats.values())
    total_time_us = sum(s['total_us'] for s in op_stats.values())
    total_time_ms = total_time_us / 1000.0

    # Print statistics by operation
    print("Performance summary by operation:")
    print(f"{'Operation':<15} {'Count':>8} {'Total (ms)':>12} {'Total %':>10} {'GFLOPS':>10} {'Avg (us)':>12} {'Min (us)':>12} {'Max (us)':>12}")
    print("-" * 110)

    for op in sorted(op_stats.keys()):
        stats = op_stats[op]
        avg_us = stats['total_us'] / stats['count'] if stats['count'] > 0 else 0
        total_ms = stats['total_us'] / 1000.0
        pct = (stats['total_us'] / total_time_us * 100.0) if total_time_us > 0 else 0

        if stats['total_flops'] > 0 and stats['total_us'] > 0:
            gflops = (stats['total_flops']) / (stats['total_us'] / 1e6) / 1e9
            gflops_str = f"{gflops:>10.3f}"
        else:
            gflops_str = f"{'N/A':>10}"

        print(f"{op:<15} {stats['count']:>8} {total_ms:>12.2f} {pct:>9.1f}% {gflops_str} {avg_us:>12.1f} {stats['min_us']:>12} {stats['max_us']:>12}")

    print("-" * 110)
    print(f"{'TOTAL':<15} {total_count:>8} {total_time_ms:>12.2f} {100.0:>9.1f}%")
    print()

    # Compute statistics by kernel + shape
    kernel_shape_stats = defaultdict(lambda: {'count': 0, 'total_us': 0, 'min_us': float('inf'), 'max_us': 0, 'total_flops': 0})

    for entry in perf_entries:
        kernel = entry.get('kernel', 'UNKNOWN')
        shape = entry.get('shape', 'UNKNOWN')
        key = f"{kernel}|{shape}"
        duration = int(entry.get('duration_us', 0))

        kernel_shape_stats[key]['count'] += 1
        kernel_shape_stats[key]['total_us'] += duration
        kernel_shape_stats[key]['min_us'] = min(kernel_shape_stats[key]['min_us'], duration)
        kernel_shape_stats[key]['max_us'] = max(kernel_shape_stats[key]['max_us'], duration)

        # Accumulate FLOPs if provided in log
        if 'flops' in entry:
            flops = int(entry['flops'])
            kernel_shape_stats[key]['total_flops'] += flops

    # Print statistics by kernel + shape
    print("Performance summary by kernel and shape:")
    print(f"{'Kernel':<20} {'Shape':<30} {'Count':>8} {'Total (ms)':>12} {'Total %':>10} {'GFLOPS':>10} {'Avg (us)':>12} {'Min (us)':>12} {'Max (us)':>12}")
    print("-" * 150)

    for key in sorted(kernel_shape_stats.keys()):
        kernel, shape = key.split('|', 1)
        stats = kernel_shape_stats[key]
        avg_us = stats['total_us'] / stats['count'] if stats['count'] > 0 else 0
        total_ms = stats['total_us'] / 1000.0
        pct = (stats['total_us'] / total_time_us * 100.0) if total_time_us > 0 else 0

        if stats['total_flops'] > 0 and stats['total_us'] > 0:
            gflops = (stats['total_flops'] / (stats['total_us']/1e6)) / 1e9
            gflops_str = f"{gflops:>10.3f}"
        else:
            gflops_str = f"{'N/A':>10}"

        print(f"{kernel:<20} {shape:<30} {stats['count']:>8} {total_ms:>12.2f} {pct:>9.1f}% {gflops_str} {avg_us:>12.1f} {stats['min_us']:>12} {stats['max_us']:>12}")

    print("-" * 150)
    print(f"{'TOTAL':<52} {total_count:>8} {total_time_ms:>12.2f} {100.0:>9.1f}%")

    # Generate JSON summary for tracking optimization progress
    json_output = {
        'log_file': str(log_file),
        'total_entries': len(perf_entries),
        'total_time_ms': total_time_ms,
        'llama_perf': llama_perf if llama_perf else {},
        'by_operation': {},
        'by_kernel_shape': {}
    }

    for op in sorted(op_stats.keys()):
        stats = op_stats[op]
        gflops = (stats['total_flops'] / (stats['total_us']/1e6)) / 1e9 if stats['total_flops'] > 0 and stats['total_us'] > 0 else 0
        json_output['by_operation'][op] = {
            'count': stats['count'],
            'total_ms': stats['total_us'] / 1000.0,
            'total_pct': (stats['total_us'] / total_time_us * 100.0) if total_time_us > 0 else 0,
            'gflops': gflops,
            'avg_us': stats['total_us'] / stats['count'] if stats['count'] > 0 else 0,
            'min_us': stats['min_us'] if stats['min_us'] != float('inf') else 0,
            'max_us': stats['max_us']
        }

    for key in sorted(kernel_shape_stats.keys()):
        kernel, shape = key.split('|', 1)
        stats = kernel_shape_stats[key]
        gflops = (stats['total_flops'] / (stats['total_us']/1e6)) / 1e9 if stats['total_flops'] > 0 and stats['total_us'] > 0 else 0
        if kernel not in json_output['by_kernel_shape']:
            json_output['by_kernel_shape'][kernel] = {}
        json_output['by_kernel_shape'][kernel][shape] = {
            'count': stats['count'],
            'total_ms': stats['total_us'] / 1000.0,
            'total_pct': (stats['total_us'] / total_time_us * 100.0) if total_time_us > 0 else 0,
            'gflops': gflops,
            'avg_us': stats['total_us'] / stats['count'] if stats['count'] > 0 else 0,
            'min_us': stats['min_us'] if stats['min_us'] != float('inf') else 0,
            'max_us': stats['max_us']
        }

    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)

    print()
    print(f"JSON summary written to '{json_file}'")


if __name__ == '__main__':
    main()
