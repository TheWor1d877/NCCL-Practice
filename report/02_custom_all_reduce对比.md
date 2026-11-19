# 模块 2：custom all-reduce 对比

## 做了什么

同一台机器、同一个模型、同样的 TP=2，起两组服务：

- Run A：默认配置，vLLM 的 custom all-reduce 处于开启状态。
- Run B：加 `--disable-custom-all-reduce`，强制走 NCCL。

两组各跑四遍 decode 计时，第一遍热身丢弃，取后三遍的中位数比较。

## 开关取证

vLLM 0.34 的参数入口只有 CLI 参数。启动时同时设 `VLLM_USE_CUSTOM_ALL_REDUCE=0` 会得到这条警告，说明环境变量在这个版本不存在：

```text
WARNING 11-19 17:36:14 [interface.py:1500] Unknown vLLM environment variable detected: VLLM_USE_CUSTOM_ALL_REDUCE
INFO 11-19 17:36:14 [startup.py:96] non-default args: {'model_tag': 'Qwen/Qwen3-0.6B', 'tensor_parallel_size': 2, 'disable_custom_all_reduce': True}
```

第二行是真正生效的证据：`non-default args` 里 `disable_custom_all_reduce` 为 `True`。只看到参数不报错不够，必须确认它进了这一行。

Run A 的启动日志里 `disable_custom_all_reduce` 为 `False`，且没有出现 "custom all-reduce is disabled" 一类提示。

## 数据

| 组 | 第 1 遍 | 第 2 遍 | 第 3 遍 | 中位数 |
|---|---:|---:|---:|---:|
| Run A（custom 开） | 3.97ms | 4.27ms | 3.07ms | 3.97ms |
| Run B（走 NCCL） | 4.60ms | 4.82ms | 4.09ms | 4.60ms |

按 Run B 为基准：`(4.60 − 3.97) ÷ 4.60 ≈ 13.7%`。关掉 custom all-reduce 后每 token 多花 0.63ms，解码吞吐从 252 tok/s 降到 217 tok/s。

组内波动比组间差异大：Run A 从 3.07ms 到 4.27ms，跨度 39%。13.7% 的差异只作为方向性证据，单独一组样本不足以排除干扰。要定量，需要把 `n` 提高到上千 token、每组重复更多轮，并且用同一 prompt 长度。本次的主要价值在于确认开关生效、两组路径不同。

## custom all-reduce 与 NCCL 的差别

NCCL 的 all-reduce 要覆盖单机多卡、多机、NVLink、PCIe、IB 各种组合，通用实现包含算法选择、协议切换、通道调度。vLLM 的 custom all-reduce 只处理一个场景：同一节点内、卡间 P2P 可用、张量并行规模为 2 到 8。范围缩小后，实现改成单次 kernel launch 完成整个 all-reduce，去掉多次小 kernel 的启动开销与流同步。

TP 解码每层做两次 all-reduce，消息大小等于 hidden size 乘 batch，0.6B 模型下只有几 KB。这个尺寸落在延迟主导区间，启动开销占大头，所以少一次 launch 就能省下可观时间。

## 结论

在 P2P 可用的 A100 上，custom all-reduce 打开比强制走 NCCL 快约 14%。这个收益来自路径特化和 kernel 数量减少，与带宽无关。

反过来的场景同样成立：GeForce 4090/5090 被驱动禁用 P2P，vLLM 会自动关闭 custom all-reduce，回落到 NCCL。此时模块 2 的对照组跑不出差异，属于硬件限制。

## 说明了 NCCL 的什么

1. NCCL 是通用库，通用性与单场景最优之间有取舍。框架在关键路径上绕过 NCCL 做特化实现，是常见的性能手段，判断依据是消息尺寸与拓扑是否固定。
2. custom all-reduce 的前提是 P2P。`isAllCudaP2p 1` 来自模块 1 的日志，这个值为 0 时框架不会启用它。
3. 比较通信优化时先确定瓶颈在带宽还是延迟。小消息看延迟，优化方向是减少 launch 与同步次数；大消息看带宽，优化方向是提高链路利用率。模块 4 的基线表给出了这条分界线。
