# NCCL 推理侧与通信工程指南

## 这个项目由哪些并列模块组成

六个模块并列，各自有目标、硬件门槛和验收。可以全做，也可以按硬件条件选做。模块之间不互相混结论。

- **模块 1：vLLM TP=2 推理 + NCCL 日志阅读**
- **模块 2：custom all-reduce vs NCCL 的 decode 延迟对比**
- **模块 3：NCCL 官方 example 跑通**
- **模块 4：通信性能基线、多节点与慢节点定位**
- **模块 5：生产级故障恢复与集群健康认证**
- **模块 6：训练侧纸面账（不跑程序）**

硬件总要求：一台至少 2 张 GPU 的云主机，T4、A10 均可满足模块 1–3；当前实机为单节点双卡 A800-SXM4-40GB（NVLink 互联），模块 4 的单机基线与模块 5 的 ⑤-2 在其上实跑。模块 1–3 不需要 RDMA 网卡。模块 4 的多节点部分需要两节点以上和多机环境；模块 5 的作业级重调度和健康认证需要 K8s/GPU Operator 和多节点。没有对应硬件时，写纸面设计并明确标注“未实跑”，不硬写数字。

**各模块共同边界：**

- 模块 1–3 不跑 nccl-tests，不做故障恢复，不出现 busbw 曲线。
- 模块 4 只记单机/多节点 busbw 基线表和慢 pair，不画曲线；不碰 RoCE/IB 参数调优。
- 模块 5 只写能实跑或能解释清楚的恢复行为；没实跑就不写恢复时间。
- 模块 6 只做纸面账，不跑训练程序。
- 任何模块都不硬写没有硬件支持的数字。


## 模块 1：vLLM TP=2 推理 + NCCL 日志阅读

### 这个模块要干什么

跑通 vLLM 双卡 TP 推理，看懂 NCCL 日志。确认单机双卡走的是 P2P/SHM 而不是网络，看推理请求进来时 NCCL 在不在干活。

### 启动服务

用一个小模型，两张卡各放一半。命令：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2
```

`--tensor-parallel-size 2` 是必须的，vLLM 0.4.0 以后不显式指定就不会走多卡。

### 看日志的什么

NCCL 的日志分段可以用 `NCCL_DEBUG_SUBSYS` 过滤。你要看的两段：

**INIT 段**：NCCL 建通信路径时打印的。单机双卡应该看到 `P2P` 或 `SHM`，**不应该出现 `NET/IB` 或 `NET/Socket`**。如果出现了，说明它试图走网络，单机配置有问题。

**GRAPH 段**：NCCL 选了哪个算法和协议。推理的小消息通常走 LL 或 LL128，不是 Simple。看日志里实际选了什么。

### 发请求验证

服务起来后，发几个请求。观察：**请求处理过程中日志里有没有新的 NCCL 通信记录**。如果有，说明 NCCL 确实在推理的 critical path 上，不是只在启动时初始化一次就完事。

### 这个模块的产出

一段注解过的日志片段，标出 INIT 段走的是什么路径，以及推理请求时 NCCL 有没有活动。

### 这个模块的验收

- vLLM TP=2 跑通。
- NCCL 日志 INIT 段能解释走了什么路径，应该是 P2P/SHM，不是网络。
- 能指出推理请求时 NCCL 有没有活动。


## 模块 2：custom all-reduce vs NCCL 的 decode 延迟对比

### 这个模块要干什么

对比 custom all-reduce 和 NCCL 在 decode 阶段的差异。vLLM 默认走自己的 custom all-reduce 绕过 NCCL ring，关掉它强制走 NCCL，看 decode 延迟差多少。这个实验解释“为什么推理引擎不直接用 NCCL 做小消息”。

### 为什么要做这个

NCCL 的 ring all-reduce 是为**大消息**设计的，带宽最优。但推理 decode 阶段的消息很小：batch=1 时，每层 all-reduce 的消息可能只有十几 KB。ring 算法要串行走多跳，延迟是瓶颈，不是带宽。

所以 vLLM 在支持 P2P/NVLink 的卡上默认启用 **custom all-reduce**，用一次 kernel launch 直接读对端 buffer 做归约，绕过 NCCL ring。你要验证的就是这个差异。

### 怎么对比

**第一遍**：默认启动，custom all-reduce 开启，发固定请求，记录 decode 阶段的 tok/s 或每 token 延迟（TPOT）。

**第二遍**：关掉 custom all-reduce，强制走 NCCL：

```bash
VLLM_USE_CUSTOM_ALL_REDUCE=0 \
NCCL_DEBUG=INFO \
vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2
```

同样请求，同样记录 decode 指标。

**注意**：只对比 decode。prefill 阶段消息大，NCCL ring 反而是合适的，custom all-reduce 的优势不明显。报告里要写清楚“这个差异只在 decode 小消息场景下成立”。

### 如果卡不支持 custom all-reduce

custom all-reduce 需要 GPU 之间有完整的 P2P 访问，NVLink 或 PCIe full peer。如果你的租用实例是 PCIe 部分连接，vLLM 会静默回退到 NCCL。这种情况下对比做不了，在报告里写“实例拓扑不支持 custom all-reduce，强制走 NCCL 是唯一路径”。

### 这个模块的产出

一句话结论：decode 阶段，custom all-reduce 开启和关闭的 tok/s 或 TPOT 差异是多少；或“因拓扑不支持无法对比”。

### 这个模块的验收

- custom all-reduce 开/关的 decode 延迟对比有数，或明确写“拓扑不支持”。
- 结论限定在 decode 小消息场景，不外推到 prefill 或大消息。


## 模块 3：NCCL 官方 example 跑通

### 这个模块要干什么

跑 NVIDIA NCCL 仓库的官方 example，以推理相关的为主，训练相关的为辅。跑通、能解释每个 example 教了什么。

### 跑哪些、怎么选

NCCL 仓库 `docs/examples/` 下的示例，按推理相关度排：

**推理直接相关：**

- **`04_user_buffer_registration`**：用户预注册 buffer 省掉 NCCL 内部拷贝。推理的权重和 KV cache 都是预分配的，这个 example 教的就是怎么让 NCCL 直接操作它们。
- **`05_symmetric_memory`**：NCCL 2.27 的 window registration，专门为小消息低延迟优化，目标就是推理场景。需要较新 NCCL 和 GPU 支持，跑之前确认版本。
- **`06_device_api` 的单机 LSA 部分**：在 GPU kernel 内直接发起通信。推理的每层 forward 都通信，降低启动开销有直接价值。**但注意**：其中的 GIN 子项需要多机 RDMA，单机跑不了，跳过。

**跑一下知道是什么就行：**

- **`01_communicators`**：创建通信组。NCCL 的 Hello World，任何卡都能跑。
- **`02_point_to_point`**：两个 rank 互发数据。推理的 pipeline parallel 用这个模式。
- **`03_collectives`**：最小 all-reduce。知道 all-reduce 长什么样即可，不用深究梯度叙事，那是训练侧的核心，推理侧 TP 也用但消息模式不同。
- **`07_kernel_fusion`**：通信和计算融合。里面的 RMSNorm + all-reduce 例子跟推理每层都相关，但部分子项需要 Hopper 以上架构，T4 可能跑不了，以实际编译结果为准。

### 每个 example 记什么

三句话，不要多：

1. 它调了哪几个 NCCL 函数/API。
2. 它教了什么，比如“预注册 buffer 消除了什么开销”。
3. 跑的时候有没有报错，怎么解决的。

**不记带宽数字，不画曲线。** 官方明确说了这些是教学模板，不是性能基准。跑 example 的产出是“跑通 + 能解释”，不是数据。

### 这个模块的验收

- 至少跑通 `01/02/03/04`。
- `05/06` 视 NCCL 版本和 GPU 支持情况尽力跑。
- 每个能跑通的 example 能说清教了什么。
- 跑不通的写清楚原因，比如版本、架构、缺少多机 RDMA。



## 模块 4：通信性能基线、多节点与慢节点定位

### 这个模块要干什么

做通信性能基线和多节点验证，目标不是“扫一条漂亮曲线”，而是拿到可解释的 busbw 基线、验证多节点 collective、识别慢节点或慢 pair。本模块与模块 1–3 的推理结论分开写。当前可用硬件为单节点双卡 A800-SXM4-40GB，NVLink 互联；单机 nccl-tests 基线实跑，多节点 collective 和慢节点定位写实验设计并标注“未实跑”。

### 硬件门槛

- 单机 nccl-tests：当前双卡 A800-SXM4-40GB 即可做，实跑。
- 多节点：需要两节点以上，能改 `NCCL_SOCKET_IFNAME` 和 rendezvous 地址。当前无第二节点，多节点部分写实验设计，标注“未实跑，原因：单节点”，不写多节点数字。
- 拓扑感知成对扫描：最好有 Slurm；当前无多节点、无 Slurm，写扫描设计，不硬写多节点数字。
- 不碰 RoCE/IB 参数调优；多节点只用环境已有网络。

### 跑什么

**单机 nccl-tests 基线（实跑）**

编译 `https://github.com/NVIDIA/nccl-tests` 后跑：

```bash
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 2
```

完成标志：拿到单机 busbw 基线表，能解释 `-b` / `-e` / `-f` / `-g` 的含义，并说明该基线是在 NVLink 拓扑下测得。

**多节点 playbook（未实跑，纸面设计）**

不跑 `https://github.com/NVIDIA/dgx-spark-playbooks` 中 NCCL 多节点 playbook 的 `setup.sh` / `launch.sh`。写清楚：若有两节点，需要修改 `NCCL_SOCKET_IFNAME` 和 rendezvous 地址；`all_reduce` 或 `all_gather` 的启动命令长什么样；成功判据是什么。标注“未实跑，原因：单节点”。

**拓扑感知慢节点定位（未实跑，纸面设计）**

不跑 `awslabs/awsome-distributed-training` 的 `micro-benchmarks/nccl-tests/slurm/topology-aware-nccl-tests/`。写清楚扫描设计：用 `sbatch --array` 对 N 节点跑成对 all-reduce 扫描，再用 `process_nccl_results.sh` 导出 CSV；以明显低于均值的 pair 为慢 pair，AWS 验证脚本常用 ~5% 为阈值；能定位到具体节点，而不是笼统判断“网络不行”。标注“未实跑，原因：无多节点 + 无 Slurm”。

**调优参数入口**

完成慢节点识别之后，再进入细粒度调优；当前单机可做有限对比。参考 `https://github.com/ml-engineering-curriculum/training-pipeline-engineer-learning` 的 `mod-105-networking-storage-for-training`，理解 `NCCL_ALGO`（Ring / Tree / CollNet）、`NCCL_PROTO`（Simple / LL / LL128）、rail alignment、PXN、subcommunicators 的适用场景。

完成标志：能根据消息大小和拓扑给出至少一组有理由的 `NCCL_ALGO` / `NCCL_PROTO` 设置，并在单机 nccl-tests 上做对比验证，记录 busbw 差异；若单机 NVLink 下该组参数差异不显著，如实写“单机 NVLink 下该组参数差异不显著，多节点场景的调优结论未实跑”。不硬写多节点调优数字。

### 这个模块的产出

- 单机 busbw 基线表（实跑）。
- 多节点 all_reduce/all_gather 成功记录，或“未实跑，原因：单节点”，附启动命令与成功判据的纸面设计。
- 慢 pair 识别结果：未实跑，原因：无多节点；附成对扫描设计和 ~5% 阈值判据，能说明如何定位到具体节点。
- 一组有理由的 `NCCL_ALGO` / `NCCL_PROTO` 设置和单机对比结果，或纸面理由。

### 这个模块的验收

- 能解释 `-b/-e/-f/-g`。
- 拿到单机 busbw 基线（实跑）。
- 多节点 `all_reduce` 或 `all_gather`：明确写“未实跑，原因：单节点”，并给出可执行的实验设计，不写多节点数字。
- 慢 pair 定位：明确写“未实跑，原因：无多节点”，给出扫描设计，不写慢节点数字。
- 能给出至少一组有理由的 `NCCL_ALGO` / `NCCL_PROTO` 设置；有单机对比就写单机结果，没有就写纸面理由。
- 不画 busbw 曲线进模块 1–3。


## 模块 5：生产级故障恢复与集群健康认证

### 这个模块要干什么

前几个模块的验收标准是“一次运行成功”；这个模块的验收标准是**在人为制造故障或拓扑不匹配时，系统行为仍然可解释、可恢复**。三个能力并列，按硬件条件做：⑤-2 在双卡 A800 上写 `ncclCommShrink` 最小示例并实跑语义验证；⑤-1 和 ⑤-3 因无 K8s 多节点集群，写纸面 workflow / 认证流程，标注“未实跑”。不写未实跑的恢复时间。

### 硬件门槛

- ⑤-1 作业级重启：需要 K8s 和多节点训练环境；当前无，写 workflow 和恢复路径，标注“未实跑，原因：无 K8s 多节点环境”，不写恢复时间。
- ⑤-2 通信器级缩容：需要较新 NCCL 支持 `ncclCommShrink` / `NCCL_SHRINK_ABORT`；当前双卡 A800 可写最小示例并实跑语义验证。若 NCCL 版本不支持，写 API 语义和示例骨架，标注“未实跑，原因：NCCL 版本不支持”。
- ⑤-3 集群健康认证：需要 K8s + GPU Operator + 多节点；当前无，写认证流程和报告字段，标注“未实跑，原因：无 K8s + GPU Operator + 多节点”，不写认证耗时。
- kind 仅用于 K8s 控制面流程演示，不替代硬件实跑；若使用，报告中单独标注“流程演示，非硬件结论”，不写恢复时间或认证耗时。

### ⑤-1 弹性重启：任务失败后自动重调度

入口：`https://github.com/NVIDIA/OSMO` 的 `cookbook/dnn_training/torchrun_reschedule`，或官方 TorchRun Rescheduling 教程。

动作：按其 workflow 跑一次 `torchrun` 训练；故意在某个 rank 抛 `DistBackendError`，观察是否自动重新调度并从 checkpoint 恢复。当前无 K8s 多节点环境，不实跑。

写清楚：重调度触发条件（如 `DistBackendError`）、checkpoint 读取路径、恢复粒度（从最近 checkpoint 继续而不是从 step 0 开始）、K8s Job/Pod 重试策略。标注“未实跑，原因：无 K8s 多节点环境”，不写恢复时间。

若用 kind 做流程演示：只记录“Pod 失败后 Job 是否重建、checkpoint 路径是否被引用”，标注“K8s 控制面流程验证，非真实训练恢复实跑”，不写恢复耗时数字。

### ⑤-2 通信器级故障恢复：不重启整个作业，把坏 rank 踢出去

入口：NVIDIA 技术博客 *Building Scalable and Fault-Tolerant NCCL Applications* + NCCL 仓库。

动作：在双卡 A800 上基于 `ncclCommShrink` + `NCCL_SHRINK_ABORT` 写最小示例：建 communicator 后模拟某 rank 失败，调用 shrink 后剩余 rank 继续完成一次 allreduce。若 NCCL 版本不支持该 API，则写 API 语义和示例骨架，标注“未实跑，原因：NCCL 版本不支持”。

完成标志：剩余 rank 能在**不重新分配 uniqueId、不全体重启**的情况下完成一次 collective；或写清 API 语义与最小示例骨架，并标注未实跑原因。

### ⑤-3 集群健康认证：生产前先查出坏节点

入口：`https://github.com/NVIDIA/cluster-readiness-engine`（NVCRE）。

动作：安装 `nvcrectl` 后跑首次 certification：`kubectl nvcre setup init` → 按 your-first-certification 执行一个 communication 测试并读取报告。当前无 K8s + GPU Operator + 多节点，不实跑。

写清楚：认证流程、报告字段（每个目标节点有明确 pass/fail 及原因）、隔离一个节点后重跑的预期结论（该节点被跳过或标红，健康节点结论不受影响）。标注“未实跑，原因：无 K8s + GPU Operator + 多节点”，不写认证耗时。

若用 kind 做流程演示：只演示 NVCRE 安装和报告框架，标注“流程演示，底层硬件依据不真实，不作为节点健康结论”。

### 这个模块的产出

- 恢复层级图：作业级重启是兜底，通信器级缩容目标是最小化恢复范围。
- ⑤-2：双卡实跑日志，或 API 语义 + 最小示例骨架（标注未实跑原因）。
- ⑤-1 / ⑤-3：纸面 workflow / 认证流程，明确标注“未实跑”；kind 若使用，单独标注“流程演示，非硬件结论”。
- 明确区分哪些是实跑，哪些是纸面。

### 这个模块的验收

- ⑤-1：写清重调度触发条件、checkpoint 读取路径、恢复粒度，标注“未实跑，原因：无 K8s 多节点环境”；不写恢复时间。若用 kind，只写 K8s 控制面流程验证，不写恢复耗时。
- ⑤-2：双卡上实跑剩余 rank 不重新分配 uniqueId、不全体重启完成一次 collective；若 NCCL 版本不支持，写 API 语义和最小示例骨架并标注未实跑原因。
- ⑤-3：写清认证流程、报告字段、隔离节点后的预期结论，标注“未实跑，原因：无 K8s + GPU Operator + 多节点”；不写认证耗时。若用 kind，只写流程演示，不作为硬件健康结论。
- 不写未实跑的恢复时间。


## 模块 6：训练侧纸面账（不跑程序）

### 这个模块要干什么

不跑程序，纸面算账。不需要额外硬件，跟其他模块放一起，但结论独立。

### checkpoint 的账

70B 参数，混合精度 Adam，每参数 16 字节，总 checkpoint 1.12 TB。分片到 1024 卡，每卡约 1.1 GB。不能所有 rank 读同一份对象存储，元数据先扛不住，所以每 rank 写自己的分片。

保存间隔公式：√(2 × 存一次耗时 × 平均故障间隔)。把你实际测的存 checkpoint 耗时代进去算一个数。

### 慢节点 vs 网络故障的区分

训练变慢，先看是不是某个节点自己慢，CPU/磁盘/数据加载，不要一上来就说网络有问题。这是训练侧运维的基本判断，跟 NCCL 本身无关，但面试会问。

### 这个模块的验收

- checkpoint 容量账写得出来。
- 保存间隔公式写得出来，并能代入参数算一个数。
- 能说清慢节点和网络故障的区分顺序。


## 总体验收

按模块列，不混在一起。

**模块 1：** vLLM TP=2 跑通；NCCL 日志 INIT 段能解释走了什么路径，应该是 P2P/SHM，不是网络；能指出推理请求时 NCCL 有没有活动。

**模块 2：** custom all-reduce 开/关的 decode 延迟对比有数，或明确写“拓扑不支持”；结论限定在 decode 小消息场景。

**模块 3：** NCCL example 至少跑通 `01/02/03/04`；`05/06` 视 NCCL 版本和 GPU 支持情况尽力跑；每个能说清教了什么。

**模块 4：** 能解释 `-b/-e/-f/-g`；拿到单机 busbw 基线；多节点 `all_reduce` 或 `all_gather` 成功，或明确写“未实跑，原因：单节点”；能识别慢 pair 并定位节点，或写扫描设计；能给出有理由的 `NCCL_ALGO` / `NCCL_PROTO` 设置并对比验证，或写纸面理由。

**模块 5：** ⑤-1 有重跑记录并从 checkpoint 恢复，或写清恢复路径并标注未实跑；⑤-2 剩余 rank 不重新分配 uniqueId、不全体重启完成一次 collective，或写 API 语义和最小示例骨架；⑤-3 每节点 pass/fail 及原因，隔离节点后健康节点结论不受影响，或写认证流程和预期报告。不写未实跑的恢复时间。

**模块 6：** checkpoint 容量账和保存间隔公式写得出来。

**共同不出现：** 没有硬件支持却硬写的数字；未实跑却写恢复时间；把模块 4/5 的结论混进模块 1–3 的推理结论；模块 1–3 里出现 busbw 曲线。

测完释放实例。

## 被问到怎么讲

“我做的 NCCL 是拆成并列模块的。模块 1–3 在 vLLM 上跑 TP 推理，看 NCCL 日志里单机双卡走什么路径，对比 custom all-reduce 和 NCCL ring 在 decode 小消息上的延迟差异，再跑 NCCL 官方 example 理解 API 用法。模块 4 在双卡 A800 上实跑 nccl-tests 单机 busbw 基线，多节点 collective 和拓扑感知慢节点定位写成可执行的实验设计并标注未实跑，结论不混入推理模块。模块 5 在双卡上实跑 `ncclCommShrink` 通信器级缩容最小示例，作业级重调度和 NVCRE 健康认证写纸面流程并标注未实跑，不写恢复时间。模块 6 只保留 checkpoint 容量和保存间隔的纸面账。模块 1–3 没做带宽测试，因为推理对 NCCL 的诉求是延迟不是带宽；模块 4 才看 busbw。没碰 RoCE/IB 参数调优，单机双卡走 NVLink P2P 就够了。”