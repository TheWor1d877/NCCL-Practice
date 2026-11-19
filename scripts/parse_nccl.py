"""把 nccl-tests 的输出转成 markdown 的 busbw 表。

用法:
    python parse_nccl.py logs/m4_baseline.log > results/m4_baseline_table.md

nccl-tests 每个消息大小输出一行，列为:
    size count type redop root time algbw busbw #wrong（后接 in-place 一组，本脚本取 out-of-place）
"""
import sys

print("| size(B) | algbw(GB/s) | busbw(GB/s) | #wrong |")
print("|---:|---:|---:|---:|")
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) >= 9 and f[0].isdigit():
        print(f"| {f[0]} | {f[6]} | {f[7]} | {f[8]} |")
