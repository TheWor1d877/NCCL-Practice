# 阶段 2 教程：模块 4–6（nccl-tests 基线、ncclCommShrink、纸面账）

覆盖模块 4（单机 busbw 基线与调优对比实跑，多节点 playbook 和慢节点扫描写纸面设计）、模块 5（⑤-2 `ncclCommShrink` 实跑，⑤-1 重调度与 ⑤-3 NVCRE 认证写纸面）、模块 6（纯纸面账）。

前置条件：阶段 1 已完成，云主机实例未释放，SSH 别名 `nccl` 可用，云主机上已有 `~/exp-logs/`、`~/nccl-src/` 和 conda 环境 `nccl`。阶段 1 记录表里的驱动版本、拓扑格子值、系统 NCCL 版本这三项，本阶段直接引用，不用重查。

路径约定（沿用阶段 1，新增两行）：

| 位置 | 路径 | 用途 |
|---|---|---|
| 本地 | `~/Project/NCCL-Practice/results/paper/` | 四份纸面设计文档 |
| 云主机 | `~/nccl-tests/` | nccl-tests 仓库 |

## 步骤 1：无卡机器上的准备

### 1.1 四份纸面设计文档

本地建 `../paper`，写四份文档。每份按同一套骨架写五段：目标、需要的环境、命令或流程设计、成功判据、为什么没实跑。每份末尾标注"未实跑"和对应原因，与 requirement.md 模块 4/5 的标注文案一致。要写清的内容清单：

| 文件 | 必须写清的内容 |
|---|---|
| `m4_multinode_playbook.md` | 若有两节点：`NCCL_SOCKET_IFNAME` 指定走哪块网卡（不指定时 NCCL 可能选到管理网口，多机连不上第一怪它）；rendezvous 地址怎么给（`-H` 或 `NCCL_COMM_ID`，多机 NCCL 先在这个地址上互相发现再建通信器）；`all_reduce_perf` 经 mpirun 跨机启动的命令长什么样；成功判据（两节点各自打印同一通信器的通道建立、`#wrong` 全零、退出码 0） |
| `m4_slowpair_scan.md` | 用 `sbatch --array` 对 N 节点提交成对 all-reduce 作业；每对跑 `all_reduce_perf -b 32M -e 32M -g <每节点卡数>` 固定大消息（带宽主导，慢链路才现形）；`process_nccl_results.sh` 导出 CSV；判据：busbw 低于全体均值 ~5% 的 pair 记慢，交叉比对一个节点参与的所有 pair，都慢才定位到该节点，否则定位到链路 |
| `m5_reschedule_workflow.md` | 触发条件（某 rank 抛 `DistBackendError`，torchrun 的 elastic agent 捕获）；重调度流程（agent 停全体、K8s Job 按 backoffLimit 重建 Pod、新 rank 重新 rendezvous）；checkpoint 读取路径（作业启动先探测共享存储上最新完整 checkpoint）；恢复粒度（从最近 checkpoint 继续，非 step 0）；K8s Job/Pod 重试策略字段 |
| `m5_nvcre_certification.md` | 认证流程（装 `nvcrectl` → `kubectl nvcre setup init` → 提交一个 communication 测试 → 读报告）；报告字段（每节点 pass/fail 与原因）；隔离一个节点后重跑的预期结论（该节点被跳过或标红，健康节点结论不受影响） |

### 1.2 busbw 解析脚本

用现成的 `../../scripts/parse_nccl.py`，把 nccl-tests 输出转成 markdown 表。拉回日志后这样用：

```bash
python scripts/parse_nccl.py logs/m4_baseline.log > results/m4_baseline_table.md
```

它从日志里挑以数字开头的行（nccl-tests 每个消息大小打一行数据），取 size、algbw、busbw、#wrong 四列出表。nccl-tests 每行数据其实有两组 time/algbw/busbw/#wrong（前一组 out-of-place，后一组 in-place，见 3.1），脚本取前一组。输出长这样：

```
| size(B) | algbw(GB/s) | busbw(GB/s) | #wrong |
|---:|---:|---:|---:|
| 8 | 0.05 | 0.05 | 0 |
| ... |
```

跑出来只有表头没有数据行，说明日志里没有数字开头的行——多半编译或运行根本没成功，先回看 `../../logs/m4_build.log` 和 `m4_baseline.log` 的末尾。

### 1.3 模块 6 纸面账

本地直接写完，存 `../test/m6_checkpoint.md`，三个小节：

- **容量账**：70B 参数 × 每参数 16 字节（fp32 主权重 4 + fp16 梯度和参数副本 2 + Adam 一阶二阶动量各 4 + 杂项 2）= 1.12TB；分片到 1024 卡 ≈ 每卡 1.1GB。16 字节的拆分是混合精度 + Adam 训练的标准显存布局，逐项写进文档。为什么每 rank 写自己的分片：所有 rank 读同一份对象存储，元数据服务先扛不住。
- **保存间隔**：公式 T = √(2·C·F)，C 是存一次 checkpoint 耗时，F 是平均故障间隔。代入示例：C=120s、F=4 小时（14400s），T = √(2×120×14400) ≈ 1859s ≈ 31 分钟。C 和 F 是假设值，写明"假设"，不写成实测。
- **慢节点 vs 网络故障**：训练变慢，先查单节点自身（CPU 占用、磁盘 IO、数据加载速度），排除后再怀疑网络。顺序反了会把数据加载慢误判成链路故障。

## 步骤 2：有卡机器上的准备

登录云主机，`tmux attach -t work` 接回原会话（没了就 `tmux new -s work` 重开）。新开的窗口不继承旧窗口的激活环境，重新 `conda activate nccl`。

### 2.1 编译 nccl-tests

```bash
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests
cd ~/nccl-tests && make 2>&1 | tee ~/exp-logs/m4_build.log
ls ~/nccl-tests/build/all_reduce_perf
```

`make` 一次编出全部性能测试程序（all_reduce_perf、all_gather_perf 等一整套），都落在 `build/` 下，本模块只用 all_reduce_perf。最后一行 `ls` 能列出文件即编译成功；报 No such file 就翻 `m4_build.log` 找第一处 error，对照下表。

| 报错 | 处理 |
|---|---|
| nvcc: command not found | `which nvcc` 确认；没有就找 CUDA toolkit：镜像一般有 `/usr/local/cuda/bin/nvcc`，`export PATH=/usr/local/cuda/bin:$PATH` 后重跑 make |
| nccl.h 找不到 | libnccl-dev 没装或阶段 1 之后环境变了，重装：`apt-get install -y libnccl-dev` |
| 链接 -lnccl 失败 | 同上，或加 `make NCCL_HOME=/usr` |

### 2.2 确认 ncclCommShrink 可用性

```bash
grep -n "ncclCommShrink" /usr/include/nccl.h
```

这条命令在系统 NCCL 头文件里找这个 API 的声明。三种结果：

- **有输出**：形如 `ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags)` 的一行声明（参数名以头文件为准）。⑤-2 走实跑，3.3 照做，写示例前把整行声明和它上面的注释读一遍，注释里写了调用方职责（哪些 rank 要调、被排除的 rank 该做什么）。
- **无输出，但 `/usr/include/nccl.h` 存在**（`ls /usr/include/nccl.h` 确认）：系统 NCCL 版本太旧没有这个 API。两个选择。其一源码编译阶段 1 克隆的 `~/nccl-src`：`make -j src.build`，产物在 `~/nccl-src/build/`，编的要花一段时间，⑤-2 编译时加 `-I ~/nccl-src/build/include -L ~/nccl-src/build/lib -lnccl`；其二按 requirement 兜底，⑤-2 降级为 API 语义加示例骨架，标注"未实跑，原因：NCCL 版本不支持"。
- **报 No such file or directory**：连头文件都没有，libnccl-dev 没装上，回 2.1 的报错表装好再查。

## 步骤 3：正式运行与数据采集

日志命名：

| 文件 | 内容 |
|---|---|
| `~/exp-logs/m4_build.log` | nccl-tests 编译输出 |
| `~/exp-logs/m4_baseline.log` | 单机 busbw 基线 |
| `~/exp-logs/m4_1M_*.log` / `m4_1G_*.log` | 调优对比六组 |
| `~/exp-logs/m5_shrink_build.log` / `m5_shrink.log` | ⑤-2 编译输出 / 运行输出 |

### 3.1 模块 4：单机基线（实跑）

```bash
cd ~/nccl-tests
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 2 2>&1 | tee ~/exp-logs/m4_baseline.log
```

四个参数：`-b 8` 从 8 字节消息测起，`-e 8G` 测到 8GB 封顶，`-f 2` 每档消息量乘 2，`-g 2` 用两张卡（每张卡一个 rank）。8B 到 8G 翻倍共约 28 档，每档反复跑取平均。8G 档两卡各要放得下 8GB 的收发 buffer，40GB 卡够用。全程输出都进日志，跑的过程中屏幕按档滚动。

输出怎么读。开头是几行 `#` 注释（版本、卡数、拓扑），然后每个大小一行数据：

```
#       size   count    type  redop   root     time    algbw    busbw   #wrong     time    algbw    busbw   #wrong
        (B)  (element)   (f)    (f)     (us)   (GB/s)   (GB/s)            (us)   (GB/s)   (GB/s)
     1048576     262144  float     sum     -1    ...
```

每行有两组 time/algbw/busbw/#wrong：前一组 out-of-place（发送和接收用两块 buffer），后一组 in-place（同一块 buffer 收发）。看前一组即可，步骤 1 的解析脚本也取这一组。关注三列：size 对档位，busbw 看带宽，#wrong 验正确。

跑完检查两点：末尾有 `Avg bus bandwidth : xx.xx` 汇总行；所有数据行的 `#wrong` 列全零。#wrong 非零说明该档结果错误，数据无效，整档重跑（把 `-b`、`-e` 都设成那个大小单跑一档）。

| 现象 | 处理 |
|---|---|
| 启动报 ncclInvalidUsage | 参数格式错，对照上面四个参数逐个检查 |
| 中途某档卡住不动 | 别 kill 整轮，记下卡住的 size，重跑时用 `-b/-e` 跳过该档，其余先拿到 |

### 3.2 模块 4：调优对比（实跑，有限）

挑两个代表尺寸：1MB 是 decode 量级（小消息，预期 LL 协议占优），1GB 是训练梯度量级（大消息，预期 Simple 协议 + Ring 占优）。每个尺寸跑默认、Tree、LL 或 Simple 共三组。`NCCL_ALGO`、`NCCL_PROTO` 是强制指定算法/协议的环境变量，放在命令最前面只对本条命令生效：

```bash
cd ~/nccl-tests
./build/all_reduce_perf -b 1M -e 1M -g 2 2>&1 | tee ~/exp-logs/m4_1M_default.log
NCCL_ALGO=Tree ./build/all_reduce_perf -b 1M -e 1M -g 2 2>&1 | tee ~/exp-logs/m4_1M_tree.log
NCCL_PROTO=LL  ./build/all_reduce_perf -b 1M -e 1M -g 2 2>&1 | tee ~/exp-logs/m4_1M_ll.log
./build/all_reduce_perf -b 1G -e 1G -g 2 2>&1 | tee ~/exp-logs/m4_1G_default.log
NCCL_PROTO=Simple ./build/all_reduce_perf -b 1G -e 1G -g 2 2>&1 | tee ~/exp-logs/m4_1G_simple.log
NCCL_ALGO=Tree ./build/all_reduce_perf -b 1G -e 1G -g 2 2>&1 | tee ~/exp-logs/m4_1G_tree.log
```

每组记下 busbw 一行。`-b` 和 `-e` 相同就只测一档，每份日志只有一行数据，提取用：

```bash
for f in ~/exp-logs/m4_1M_*.log ~/exp-logs/m4_1G_*.log; do
  echo "== $f"; awk '$1 ~ /^[0-9]+$/ {print "size="$1, "busbw="$8, "wrong="$9}' $f
done
```

（`$8` 是 out-of-place 组的 busbw 列，`$9` 是 #wrong 列。）

一个预期内的现象：Tree 算法在只有 2 个 rank 时可能不被 NCCL 采纳，日志里会出现 "Ignoring" 之类的提示，或者跑出来的 busbw 和默认组几乎一样。这本身就是调优观察的一部分，如实记录"双卡下 NCCL_ALGO=Tree 未生效/无差异"，不硬凑结论。双卡 NVLink 下 Ring 和 Tree 往往拉不开差距，差异不显著就如实写"单机 NVLink 下该组参数差异不显著，多节点场景的调优结论未实跑"。

### 3.3 模块 5 ⑤-2：ncclCommShrink 最小示例（实跑）

目标：2 rank 建好父通信器后把 rank1 踢掉，rank0 不重新分配 uniqueId（建通信器时全体交换的握手凭证，重新分配等于全体重来）、不重启作业，在缩容后的通信器上完成一次 allreduce。

写 `~/shrink_demo.c`，结构自上而下六段：

1. MPI 初始化，按 rank 号各绑一张 GPU。
2. `cudaMalloc` 分配发送和接收 buffer。
3. rank0 调 `ncclGetUniqueId` 拿 uniqueId，`MPI_Bcast` 广播给 rank1。
4. 两 rank 各调 `ncclCommInitRank` 建父通信器；先做一次 `ncclAllReduce` 验证父通信器正常。
5. rank1 到此退出，不再参与任何调用（模拟该 rank 故障）。
6. rank0 对父通信器调 `ncclCommShrink` 排除 rank1，得到 shrunk 通信器，在它上面再做一次 `ncclAllReduce`，打印返回值和结果。

第 3、4 段照 `~/nccl-src/docs/examples/01_communicators` 的 MPI 子项抄，shrink 部分核心调用（完整参数列表和调用方职责以 2.2 查到的 nccl.h 声明及其注释为准）：

```c
int exclude = 1;                                  // 要踢掉的 rank
ncclComm_t shrunk;
ncclCommShrink(parent, &exclude, 1, &shrunk, NULL, NCCL_SHRINK_ABORT);
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, shrunk, stream);
```

`NCCL_SHRINK_ABORT` 表示父通信器上未完成的通信直接中止。

编译、运行、留痕：

```bash
mpicc ~/shrink_demo.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lnccl \
  -o ~/shrink_demo 2>&1 | tee ~/exp-logs/m5_shrink_build.log
mpirun --allow-run-as-root -np 2 ~/shrink_demo 2>&1 | tee ~/exp-logs/m5_shrink.log
grep -iE "shrink|success|error" ~/exp-logs/m5_shrink.log
```

示例里有 MPI 调用，用 `mpicc` 编译，再手动补 CUDA runtime 的头文件路径与 `-lcudart`。只用 `nvcc` 会找不到 `mpi.h`。

日志里确认两点：shrink 调用返回 success；rank0 在 shrunk 通信器上完成了 collective。注意单 rank 的 allreduce 语义上就是把输入原样给出（求和只有一个加数），NCCL 很可能短路成本地拷贝没走任何通信路径——这个现象本身要如实记录，配上对 API 语义的理解（缩容的价值在于通信器还在、作业没重启），这就是有效产出。

| 现象 | 处理 |
|---|---|
| mpirun 几十秒不退出 | 多半是 rank1 还堵在父通信器的某次 collective 里，或 shrink 的调用方和头文件注释要求不一致。`Ctrl+C`，把终止前最后一行输出抄进笔记 |
| 编译报 undefined reference: ncclCommShrink | 链接的 libnccl 太旧，回 2.2 走源码编译那条路 |
| 运行报 unhandled cuda error | 头文件和链接库版本对不上，检查 2.2 选了哪条路，编译参数要与之一致 |

### 3.4 收尾

关掉所有跑着的程序，退出 tmux（`Ctrl+B D`）。**先别释放实例**，等步骤 4 的 rsync 拉完数据。

## 步骤 4：将数据拉取到本地并分析

本地执行（参数含义和拉完对账的文件数检查见阶段 1 步骤 4，命令相同）：

```bash
rsync -avP nccl:~/exp-logs/ ~/Project/NCCL-Practice/logs/
python scripts/parse_nccl.py logs/m4_baseline.log > results/m4_baseline_table.md
```

### 4.1 基线表分析

两个带宽概念：algbw = 数据量 ÷ 耗时，描述这次 collective 整体多快；busbw = 每张卡链路实际承担的带宽，all-reduce 换算系数 algbw × 2(n−1)/n，n=2 时两者数值相等。

拿 busbw 列按消息大小分段看形状：小消息段（8B 到几十 KB）busbw 低是延迟主导，属正常形态，不是故障；大消息段（8MB 往上）busbw 爬升并趋平，平台值对硬件。NVLink 机器（A800/H800 SXM）的聚合带宽约 400GB/s（双向口径），大消息段 busbw 到理论值六到八成说明 NVLink 正常。PCIe 双卡机器（`nvidia-smi topo -m` 为 PHB 或 PIX）的平台值在 10–16GB/s，同样属正常。大消息段爬不上去才需要回头查拓扑（阶段 1 的 2.1 记录）。基线表标注实际拓扑，例如"单节点双卡、PCIe（PHB）拓扑下实跑"。

### 4.2 调优对比分析

六份日志各取 busbw 行（3.2 的 awk 命令在拉回本地后同样可用，把路径换成本地 logs/），拼一张 尺寸 × 配置 的对比表进 `../test/m4_tuning_table.md`，每格写 busbw 数值，行末写相对默认组的差异百分比。结论按 3.2 的口径，差异不显著或参数未生效就明说。

### 4.3 ⑤-2 分析

从 `../../logs/m5_shrink.log` 摘 shrink 调用和 collective 完成两段，配三行说明：踢掉了谁、没重新分配 uniqueId、剩余 rank 完成了什么。若观察到单 rank allreduce 被短路成本地拷贝，把这句加进说明。

### 4.4 拼阶段 2 报告并释放实例

报告分两栏：实跑部分（基线表、调优对比表、shrink 日志）和纸面部分（`../paper` 四份、`m6_checkpoint.md`），每部分明确"实跑"或"未实跑 + 原因"。对照 requirement.md 模块 4/5/6 验收条目逐条勾。全部勾完、确认本地 logs/ 和 results/ 齐了，去平台控制台释放云主机实例。

## 本阶段名词速查

| 名词 | 含义 |
|---|---|
| nccl-tests | NVIDIA 官方通信性能测试套件，产出 busbw 表 |
| algbw / busbw | 前者是操作整体带宽，后者是每卡链路承担的有效带宽，见 4.1 |
| out-of-place / in-place | 两种测量：发送和接收用两块 buffer / 同一块 buffer 收发，本教程取 out-of-place 组 |
| rendezvous | 多机 NCCL 的会合过程：先在一块约定的网卡和地址上互相发现，再建通信器 |
| NCCL_SOCKET_IFNAME | 指定 NCCL 走哪块网卡的环境变量 |
| NCCL_ALGO / NCCL_PROTO | 强制指定集合通信算法（Ring/Tree）/ 传输协议（Simple/LL/LL128）的环境变量，默认 NCCL 自动选 |
| ncclCommShrink | 从一个通信器里排除指定 rank、生成新通信器的 API，父通信器和作业进程都不动 |
| GIN | GPU 在 kernel 内直接发起的 RDMA 通信，要多机 RDMA 网络 |
| LSA | Load/Store Accessible，同机 GPU 间直接读写对方显存的设备端通信方式 |
| multimem | NVSwitch 多播内存操作，Hopper 起才有 |
| DistBackendError | PyTorch 分布式后端异常的异常类型，重调度以它为触发信号 |
| checkpoint | 训练中间态快照，存参数和优化器状态供故障后恢复 |
| MTBF（公式里的 F） | 平均故障间隔，一台机器两次故障之间的平均时间 |
| NVCRE | NVIDIA 集群就绪引擎，跑标准化测试给每个节点出健康认证报告 |
