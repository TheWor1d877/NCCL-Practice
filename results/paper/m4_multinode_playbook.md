# 模块 4：多节点 all_reduce playbook

状态：未实跑，原因：单节点。

## 目标

两个节点各出 2 张卡，共 4 个 rank 建立同一个通信器，跑通一次跨机 all_reduce，拿到跨机 busbw 的量级。

## 需要的环境

- 两个节点，GPU 型号一致（本实验为 A100-SXM4-40GB，sm_80），驱动与 NCCL 版本一致。
- 节点间网络可互通。有 IB/RoCE 时 NCCL 走 IB；没有则走 Socket，需要知道业务网卡名，避免选中管理口或 docker0。
- 每节点有 `~/nccl-tests/build/all_reduce_perf`，有 `mpirun` 或 `torchrun`。
- 节点间 root 免密 SSH，mpirun 才能远程拉起进程。

## 命令或流程设计

指定网卡和调试级别，跨机拉进程：

```bash
mpirun --allow-run-as-root -np 4 \
  -H node0:2,node1:2 \
  -x NCCL_SOCKET_IFNAME=eth1 \
  -x NCCL_DEBUG=INFO \
  ~/nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

- `-H node0:2,node1:2` 表示每节点起 2 个进程，配合 `-g 1` 让每进程占一张卡，共 4 个 rank。
- `NCCL_SOCKET_IFNAME=eth1` 指定 NCCL 的 IP 层走哪块网卡。不指定时 NCCL 可能选到管理网口，多机连不上的第一排查点就是它。
- rendezvous 等价入口：设 `NCCL_COMM_ID=<node0_ip>:<port>`，所有 rank 先在这个地址互相发现再建通信器；mpirun 的 `-H` 只是把进程送过去，发现仍走这条路径。

## 成功判据

1. 两个节点各自打印出同一通信器的通道建立行（`Channel 00/..` 与 `via NET/IB` 或 `via NET/Socket`），不是单边成功。
2. `all_reduce_perf` 所有数据行的 `#wrong` 列全为 0。
3. 进程退出码 0，末尾有 `Avg bus bandwidth` 汇总行。
4. 大消息档的 busbw 落在节点间物理带宽的量级，而不是单机共享内存量级。

## 为什么没实跑

当前只有一台双卡实例，没有第二个节点，跨机部分不写数字。命令与判据按上述设计保留，拿到第二台机器可直接执行。
