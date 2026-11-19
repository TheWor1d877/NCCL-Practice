# 模块 5 ⑤-1：作业级弹性重调度 workflow

状态：未实跑，原因：无 K8s 多节点环境。

## 目标

某个 rank 通信失败导致整个训练作业退出时，作业不被人工干预地重建，并从最近的 checkpoint 继续，而不是从 step 0 重来。

## 需要的环境

- Kubernetes 集群，GPU Operator 就绪，多节点多卡。
- torchrun 的 elastic agent（`--rdzv-backend=c10d` 或 etcd）。
- 共享存储存放 checkpoint，所有 Pod 可读写同一路径。
- K8s Job 配置 `backoffLimit`，允许失败后重建 Pod。

## 命令或流程设计

1. 启动：`torchrun --nnodes=N --rdzv-backend=c10d --rdzv-endpoint=<svc>:<port> train.py`，每节点一个 torchrun 进程。
2. 触发：某个 rank 的 NCCL 通信失败，PyTorch 抛 `DistBackendError`，elastic agent 捕获后让全体 rank 退出。
3. 重调度：K8s Job 按 `backoffLimit` 重建 Pod；新 Pod 起来后全部 rank 重新 rendezvous，从同一 rendezvous 地址重新组网。
4. 恢复：作业启动第一步先探测共享存储上最新的完整 checkpoint（写临时文件加 rename 的提交协议，避免读到写了一半的目录），存在则加载；加载后从该 checkpoint 的 step 继续，不是 step 0。

## 成功判据

- Rank 抛 `DistBackendError` 后 Job 自动重建 Pod，无需人工重新提交。
- 新作业日志显示加载了 checkpoint，起始 step 等于 checkpoint 的 step，而不是 0。
- 全体 rank 重新组网成功，训练继续推进。

## 为什么没实跑

当前没有 K8s 多节点集群，无法制造真实的 Pod 重建与重新 rendezvous。按验收要求不写恢复时间，只保留上述流程与判据。
