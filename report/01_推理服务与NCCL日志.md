# 模块 1：vLLM TP=2 推理与 NCCL 日志

## 做了什么

在单节点 2×A100-SXM4-40GB 上起了一个张量并行的推理服务，模型 Qwen3-0.6B，服务就绪后发三条 chat 请求，各限 128 token，随后跑四遍 decode 计时探针，第一遍作热身丢弃。

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
  vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2
```

日志落到 `logs/m1_startup.log`，请求原文在 `logs/m1_requests.log`，计时在 `logs/m1_bench.log`。

## 启动过程

启动按固定顺序走完六步，屏幕停在哪一步都有对应解释：

1. 模型从本地缓存加载，全程不联网。
2. 权重按 TP=2 切分，一半放 GPU0，一半放 GPU1。
3. NCCL 初始化，INIT 与 GRAPH 两个子系统打出通道建立行。
4. KV cache 预分配。
5. CUDA graph 捕获。这一步几十秒没有新日志，属于正常行为。
6. 打印 `Application startup complete`，服务可用。

## 日志原文

下面五行来自 `logs/m1_startup.log`，分别对应版本、通信器初始化、拓扑带宽、通道与路径类型：

```text
NCCL INFO NCCL version 2.30.7+cuda13.3
NCCL INFO [Rank 1] ncclCommInitRank comm 0x594cef582b20 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 90 commId 0xb315b6794fb097ab - Init START
NCCL INFO === System : maxBw 24.0 totalBw 24.0 ===
NCCL INFO Channel 00/02 : 0 1
NCCL INFO Channel 01/02 : 0 1
NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 1
```

逐行读：

- 第一行的版本是 torch 自带的 2.30.7，系统 `libnccl` 是 2.32.6。`import torch` 时动态库搜索路径里 torch 目录在前，vLLM 实际加载的是前者。排查版本问题时以这一行为准，`dpkg -l | grep nccl` 看到的是系统库，不代表实际运行版本。
- 第二行 `rank 1 nranks 2 cudaDev 1 busId 90` 说明两卡分属两个 rank，`commId` 两边一致，表示两个进程加入的是同一个通信器。
- 第三行 `System : maxBw 24.0` 是 NCCL 按探测到的链路估出的系统带宽，这个数值对应 PCIe 主机桥下的双卡，与 `nvidia-smi topo -m` 的 PHB 一致。
- 第四、五行建了两条通道：`Channel 00/02` 与 `Channel 01/02`，斜杠后的 02 是通道总数，冒号后 `0 1` 是通道上 rank 的排列。2.30 起不再打印 `via P2P/CUMEM` 字样，路径类型改由第七行给出。
- 第六行 `Trees [0] 1/-1/-1->0->-1` 是树形算法的链接关系，`1 -> 0` 表示 rank1 把数据交给 rank0 汇总；`[1]` 组方向相反。两条通道各建一条树。
- 第七行 `isAllCudaP2p 1` 表示两卡能直接读写对方显存，`isAllDirectP2p 0` 表示没有走无中转的直连模式。PCIe 主机桥下的双卡就是这个状态：P2P 可用，但每次访问经过桥，带宽低于 NVLink。

## 数据

三条请求的返回都正常，`completion_tokens` 为 128。解码计时三遍：TPOT 2.76ms、3.87ms、3.93ms，对应 362.5、258.2、254.2 tok/s。

三遍之间差 40%，单条请求的样本就是这个波动幅度。模块 2 用中位数做组间比较，原因在这里。

## 结论

TP=2 生效。两卡各自持有一半权重，是一个通信器的两个 rank，通道按显存直连建立，解码时每层在两张卡之间交换数据。TPOT 里包含这部分通信时间。

## 说明了 NCCL 的什么

1. 通信器建立有固定链路：探测网卡插件与 GPU 拓扑，按 rank 数划分通道，为每条通道选算法与路径。日志按这个顺序留痕，哪一段缺失就说明卡在哪。
2. 路径由物理拓扑决定，软件只做选择。同一份 vLLM 代码在 NVLink 机型和 PCIe 机型上跑，日志里的通道数、`maxBw`、`isAllCudaP2p` 都会不同，比较性能数字前先对齐这三个值。
3. `NCCL_DEBUG=INFO` 只覆盖初始化阶段，不会为每次 collective 打一行。请求期间日志没有新增行属于正常，判断 NCCL 是否参与要看路径证据（模块 2 的对照组）或 `NCCL_DEBUG_SUBSYS=COLL`。
4. 推理场景的通信特征是消息小、次数多、延迟敏感。TP=2 的 LLM 每层做两次 all-reduce，一次在 attention 输出，一次在 MLP 下投影。这个特征决定了模块 2 的自定义 all-reduce 有收益，也决定了模块 4 里 LL 协议在 PCIe 上不一定更快。
