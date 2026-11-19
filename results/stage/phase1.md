# 阶段 1 教程：模块 1–3（vLLM TP 推理、custom all-reduce 对比、NCCL example）

覆盖模块 1（vLLM TP=2 推理 + NCCL 日志阅读）、模块 2（custom all-reduce 开关对比 decode 延迟）、模块 3（NCCL 官方 example 跑通）。全部在一台双卡 A800 云主机上完成。

路径约定：

| 位置 | 路径 | 用途 |
|---|---|---|
| 本地 | `~/Project/NCCL-Practice/logs/` | 从云主机拉回的原始日志 |
| 本地 | `~/Project/NCCL-Practice/results/` | 记录表、笔记、报告 |
| 本地 | `~/Project/NCCL-Practice/scripts/` | bench_decode.py 等脚本 |
| 云主机 | `~/exp-logs/` | 所有运行日志，跑完整目录拉回 |
| 云主机 | `~/nccl-src/` | NCCL 仓库，example 在 `docs/examples/` |

## 步骤 1：无卡机器上的准备

### 1.1 租机

规格按这个挑，每一项都有后置依赖，租错一步后面全卡住：

| 项目       | 挑什么         | 为什么                                                                                                   |
|------------|----------------|----------------------------------------------------------------------------------------------------------|
| GPU        | 2× A800 40GB   | TP=2 要两张卡；0.6B 模型 40GB 显存绰绰有余                                                               |
| 卡间互联   | NVLink         | 模块 2 的 custom all-reduce 依赖两卡完整 P2P 访问，PCIe 互联的实例大概率被 vLLM 自动禁用，对比直接做不了 |
| 系统盘剩余 | ≥ 50GB         | vLLM + PyTorch + CUDA 依赖 + 模型缓存合计 20GB 往上，编译 example 还要占几个 GB                          |
| 镜像       | 预装 CUDA 驱动 | 云主机上自装驱动容易和内核版本错配，装坏了这台就废了                                                     |

租完记下三个值：**主机地址、SSH 端口、登录用户名**。后面 1.2 写 SSH 配置就填这三个值，本教程所有连云主机的命令都走配置里的别名 `nccl`，之后换实例也只用改这一处。

### 1.2 SSH 打通

本地已经有 SSH 密钥对，直接用现成的。公钥文件在 `~/.ssh/` 下，常见名 `id_ed25519.pub` 或 `id_rsa.pub`，看一眼内容：

```bash
cat ~/.ssh/id_ed25519.pub    # 没有这个文件就 ls ~/.ssh/*.pub 看你有什么
```

输出是一整行以 `ssh-ed25519` 或 `ssh-rsa` 开头的字符串，完整复制它（一整行，中间不能有换行）。

把这一行放进云主机的 `~/.ssh/authorized_keys`，两种方式选一种：

- **平台控制台**：找到"SSH 公钥 / 密钥对"入口粘贴保存，平台开机时自动写进实例。优先用这个，不用碰密码。
- **ssh-copy-id**（需要实例的登录密码）：

  ```bash
  ssh-copy-id -p <端口> <用户>@<主机地址>
  ```

  输一次密码，它把本地公钥**追加**进云主机的 `authorized_keys`，之后连接不再要密码。手动粘贴时注意是追加到文件末尾，直接覆盖会把平台上原有的其他公钥弄丢。

写本地 `~/.ssh/config`（没有就新建），之后统一用别名 `nccl`：

```
Host nccl
  HostName <主机地址>
  Port <端口>
  User <用户>
```

三个字段就是 1.1 记的三个值。写这个文件的意义：后面 scp、rsync、ssh 的命令里全部用 `nccl` 顶替"地址+端口+用户名"三件套，抄命令时不用改任何东西。

验证一条命令完成：

```bash
ssh nccl 'nvidia-smi -L'
```

单引号里的命令在云主机上执行，结果传回本地打印。第一次连接会问一句 host authenticity（是否信任这台机器），输 `yes` 回车，之后不再问。预期输出：

```
GPU 0: NVIDIA A800-SXM4-40GB (UUID: GPU-xxxxxxxx-...)
GPU 1: NVIDIA A800-SXM4-40GB (UUID: GPU-xxxxxxxx-...)
```

两行就是免密登录和 GPU 驱动一次验完。如果还提示输密码，公钥没进 `authorized_keys`，回去重做粘贴那步。连不上对照下表：

| 现象 | 处理 |
|---|---|
| Permission denied (publickey) | 公钥没进 authorized_keys，或 config 里 User 填错。重新粘贴一次，确认复制的是 .pub 文件内容且是完整一行 |
| Connection refused | 端口填错。租用平台给的常是跳板端口，不一定是 22，去控制台核对 |
| Host key verification failed | 这台实例重装过系统，主机指纹变了。`ssh-keygen -R [<地址>:]<端口>` 删掉本地存的旧指纹再连 |
| 连上但 `nvidia-smi: command not found` | 镜像没装驱动，按 1.1 的规格重新租，别自己装 |

### 1.3 本地目录与记录表

```bash
mkdir -p ~/Project/NCCL-Practice/{logs,results}
```

`-p` 对已存在的目录静默跳过，重复执行无害。logs 放步骤 4 从云主机拉回的原始日志，results 放你写的记录表、笔记和报告，两类东西分开，成稿时不用在日志堆里翻结论。

建 `../记录表.md`，骨架如下，每跑一遍填一行：

| 模块 | 第几遍 | 启动参数 | 关键指标 | 日志文件名 |
|---|---|---|---|---|
| 2 | Run A | 默认（custom all-reduce 开） | TPOT 中位数（跑完填） | m2_runA.log |

当场填的原因：步骤 4 拉回来的是二十几个日志文件，"哪个 TPOT 对应哪组启动参数"这层关系只存在于你按下回车的那一刻，事后靠猜文件名对不上。

计时脚本用现成的 `../../scripts/bench_decode.py`（2.5 传到云主机，3.2 运行）。它向 vLLM 发一条流式请求，最后打印一行：

```
tokens=256 TTFT=182.3ms TPOT=11.42ms decode_tok/s=87.6
```

四个字段：tokens 是实际收到的正文 token 数；TTFT 是请求发出到第一个 token 的时间，含 prefill 计算；TPOT 是第一个 token 之后每个 token 的平均间隔，这就是 decode 延迟，模块 2 对比的指标；decode_tok/s 是 TPOT 的倒数，用来交叉核对没抄错数。脚本内部把 `temperature` 固定为 0、`max_tokens` 固定为 256，两组实验发完全相同的请求，数字才可比。

## 步骤 2：有卡机器上的准备

登录云主机后，所有操作放进 tmux 会话。原因：SSH 断线时云主机给前台进程发挂断信号，vllm 一跑几小时，断一次线全白干；tmux 会话挂在云主机侧，断线后任务照跑。

```bash
tmux new -s work        # 开会话
# Ctrl+B 松开再按 D     # 离开会话（任务继续跑）
tmux attach -t work     # 断线后接回来
tmux ls                 # 忘了会话名时列出全部
# Ctrl+B 松开再按 C     # 在会话内开第二个窗口（发请求用）
```

### 2.1 GPU 与拓扑确认

```bash
nvidia-smi
nvidia-smi topo -m
```

`nvidia-smi` 看三点：右上角 Driver Version 有版本号；下方表格里两行 A800-SXM4-40GB；每张卡的显存占用接近 0MiB/40960MiB、没有残留进程。显存被占说明实例是共享的或上一手没清，记下 PID `kill -9`，杀不掉就找平台换实例。

`topo -m` 打印一个矩阵，行和列都是设备，看 GPU0 行 × GPU1 列那个格子：

| 格子值 | 含义 | 对本实验的影响 |
|---|---|---|
| NV4 / NV6 / NV12 | NVLink 直连，数字是链路条数，NV12 是满配 | 模块 2 正常做 |
| PIX / PHB | 走 PCIe（同一交换机下 / 跨桥） | custom all-reduce 大概率被 vLLM 禁用，准备兜底结论 |
| SYS | 跨 NUMA 走 CPU，最慢 | 同上，且基线带宽会明显偏低 |

把看到的格子值抄进记录表，这一步的结论决定模块 2 走正常对比还是兜底路径。

| 报错 | 处理 |
|---|---|
| Failed to initialize NVML: Driver/library version mismatch | 驱动和内核版本错配。先重启实例；重启无效说明镜像本身有问题，找平台 |
| INFO: No devices were found | 驱动装了但看不到卡，多是虚拟化直通配置问题，找平台 |

### 2.2 装 vLLM

```bash
conda create -n nccl python=3.11 -y
conda activate nccl
pip install vllm
```

conda 环境把依赖圈在独立目录里，不碰系统 python，装坏了删环境重来。python 3.11 在 vllm 各版本 wheel 的覆盖范围内。`pip install vllm` 会连带拉 PyTorch 和一批 CUDA 运行库，下载量按 GB 计，转圈久属正常。

装完验证：

```bash
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.nccl.version())"
```

打印三个值（数字以实际安装为准，形如 `0.9.2 2.7.1 (2, 26, 6)`）：vllm 版本、PyTorch 版本、PyTorch 自带的 NCCL 版本。三个都抄进记录表。NCCL 那个元组管 vllm 侧的通信行为；模块 3 的 example 链接的是系统 libnccl，版本在 2.4 单独查，两者互不相干。

| 报错 | 处理 |
|---|---|
| pip 超时或反复重试 | 换国内镜像：`pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| import 时报 CUDA driver version is insufficient | 装的 torch 要求的驱动比镜像里的高。记下 2.1 查到的驱动版本，对照 vllm 发布说明换版本或找平台升驱动 |
| `torch.cuda.is_available()` 返回 False | torch 看不到 GPU，先按 2.1 排除驱动问题，再 `pip install vllm --force-reinstall` 重装 |

### 2.3 下载模型

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
huggingface-cli download Qwen/Qwen3-0.6B
ls ~/.cache/huggingface/hub        # 确认有 models--Qwen--Qwen3-0.6B
```

为什么要 `HF_ENDPOINT`：huggingface.co 在国内直连多数超时，hf-mirror 是它的镜像站。写进 `~/.bashrc` 并 source，是为了之后每个新开的 shell（包括 tmux 新窗口）自动带上这个变量；vllm 启动时先查本地缓存，命中就不碰网络。`huggingface-cli` 随 vllm 的依赖装好，找不到就 `pip install -U "huggingface_hub[cli]"`。

预期输出：下载进度条走完后（0.6B 模型约 1.5GB），`ls` 能看到 `models--Qwen--Qwen3-0.6B` 目录，文件落在它的 `snapshots/<一串哈希>/` 下。

**`Qwen/Qwen3-0.6B` 这个字符串后面每次启动都要原样用**，大小写和斜杠都不能变，写进记录表。

| 报错 | 处理 |
|---|---|
| 下载仍超时 | `echo $HF_ENDPOINT` 确认变量已生效；没生效就检查 ~/.bashrc 那行拼写 |
| 下到一半断 | 重跑同一条命令，它从断点续传 |

### 2.4 example 编译环境

```bash
apt-get install -y build-essential libnccl-dev libopenmpi-dev openmpi-bin
git clone https://github.com/NVIDIA/nccl.git ~/nccl-src
```

四个包各管一块：build-essential 提供 gcc/make；libnccl-dev 提供 `nccl.h` 头文件和可链接的 libnccl；libopenmpi-dev 提供 `mpi.h` 和 mpicc；openmpi-bin 提供 `mpirun` 命令本身（libopenmpi-dev 不含它，漏装会在 3.3 报 command not found）。克隆 `~/nccl-src` 是为了拿 `docs/examples/`，模块 3 在那里逐个编。

查系统 NCCL 版本，记进记录表——example 链接的是这个版本，05_symmetric_memory 要 2.27 以上看的是它，和 2.2 记的 torch 自带版本无关：

```bash
dpkg -s libnccl-dev | grep -i version
```

没有 root 权限就改用 conda：`conda install -c conda-forge nccl-cxx-headers openmpi`。

example 不在这里一次性全量编译，留到 3.3 逐个 topic 编。原因：05/06/07 的部分子项在 A800（Ampere 架构）上编不过，在顶层 `make` 会因第一处失败中断其余全部，逐 topic 编只损失失败的那一个。

| 报错 | 处理 |
|---|---|
| nccl.h: No such file | libnccl-dev 没装上 |
| mpi.h 找不到 | libopenmpi-dev 没装上 |
| mpirun: command not found | openmpi-bin 没装上 |
| 链接找不到 -lcudart | CUDA 运行库路径问题：`export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |

### 2.5 建目录、传脚本

云主机上：

```bash
mkdir -p ~/exp-logs
```

本地执行（把计时脚本放上去）：

```bash
scp ~/Project/NCCL-Practice/scripts/bench_decode.py nccl:~/
```

`nccl:` 就是 1.2 配的别名，scp 复用同一份 `~/.ssh/config`。传完验一下：

```bash
ssh nccl 'ls -l ~/bench_decode.py'
```

能看到文件即完成。

## 步骤 3：正式运行与数据采集

所有运行输出都 tee 进 `~/exp-logs/`。`2>&1` 把标准错误并入标准输出——NCCL 和 vllm 的日志主要走标准错误，少了这段重定向，落盘文件几乎是空的；`tee` 让同一份输出既显示在屏幕又写进文件，跑的时候能盯，跑完有存档。

本阶段会产生的日志：

| 文件 | 内容 |
|---|---|
| `~/exp-logs/m1_startup.log` | 模块 1 服务启动与 NCCL 初始化日志 |
| `~/exp-logs/m1_requests.log` | 模块 1 三条请求的响应 |
| `~/exp-logs/m2_runA.log` / `m2_runA_bench.log` | Run A 启动日志 / 三遍计时输出 |
| `~/exp-logs/m2_runB.log` / `m2_runB_bench.log` | Run B 启动日志 / 三遍计时输出 |
| `~/exp-logs/m3_0X_build.log` | 模块 3 各 topic 编译输出 |
| `~/exp-logs/m3_0X.log` | 模块 3 各 topic 运行输出 |

### 3.1 模块 1：起服务看日志

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
  vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2 2>&1 | tee ~/exp-logs/m1_startup.log
```

命令各部分：`NCCL_DEBUG=INFO` 打开 NCCL 日志输出；`NCCL_DEBUG_SUBSYS=INIT,GRAPH` 限定只打通道建立（INIT）和拓扑图选择（GRAPH）两段的日志，用 ALL 会打几十万行没法看；`--tensor-parallel-size 2` 让 vLLM 把每层权重切开放到两张卡，vLLM 0.4.0 以后不显式指定就不走多卡。

启动过程按顺序会经过这些阶段，屏幕卡在哪一阶段都能对上：

1. 从本地缓存加载模型（2.3 已下好，这里不联网）。
2. 权重分载到两张卡。
3. 中间穿插 NCCL 初始化行，形如：

   ```
   NCCL INFO [Rank 0] ncclCommInitRank comm 0x... rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 80 commId 0x... - Init START
   NCCL INFO Channel 00/02 : 0 1
   NCCL INFO Channel 01/02 : 0 1
   NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
   NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 1
   ```

   rank 是每个参与进程的编号（0 和 1），nranks 2 表示这个通信组共两个参与者。NCCL 2.27 及以后的版本不再打印 `via P2P/CUMEM` 字样，通道建立行改为 `Channel 00/02 : 0 1`，斜杠后是通道总数，冒号后是通道上各 rank 的排列。路径类型看 `isAllCudaP2p`：值为 1 表示两卡能直接读写对方显存；`isAllDirectP2p 0` 表示没有走无中转的直连模式，PCIe 主机桥下的双卡就是这个状态。
4. KV cache 预分配。
5. CUDA graph 捕获——**这一步几十秒没有新日志属正常**，它在把一串 kernel 录成整图，别急着 kill。
6. 出现 `Application startup complete` 才算就绪。

服务就绪后，tmux 第二个窗口（`Ctrl+B C` 开）发三条请求，响应落盘：

```bash
for i in 1 2 3; do
  curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"介绍一下NCCL"}],"max_tokens":128}' \
    | tee -a ~/exp-logs/m1_requests.log
done
```

请求体里 `model` 必须和启动时的模型名字符串完全一致，不一致返回 404；`max_tokens=128` 固定生成长度，三条请求一样，输出才有可比性。每条预期返回一段 JSON，`choices[0].message.content` 是正文，`usage.completion_tokens` 接近 128。三条都有正文，模块 1 的"跑通"就算成立。

回第一个窗口 `Ctrl+C` 关服务，确认退干净：

```bash
ps aux | grep -i vllm | grep -v grep    # 应无输出
nvidia-smi                               # 两张卡显存回到 0MiB
```

| 报错 | 处理 |
|---|---|
| CUDA out of memory | 显存被残留进程占着。`nvidia-smi` 查 PID，`kill -9` 后重启服务 |
| Address already in use | 上一个 vllm 没退干净还占着 8000 端口。`fuser -k 8000/tcp` 杀掉占用进程 |
| 请求返回 404 model not found | curl 里 model 名和启动命令不一致 |
| 启动中途 Traceback 退出 | 整段报错抄进记录表对应行，常见原因是 2.2 的版本三元组不匹配 |

### 3.2 模块 2：custom all-reduce 开关联跑

两组实验唯一差别是 custom all-reduce 开关，模型、请求、机器都不动，差异才能归因到开关上。Run A 不加 NCCL_DEBUG 环境变量，保持默认配置，日志也干净；Run B 加，日志里出现 NCCL 通信行本身就是"真的走了 NCCL"的证据。

**Run A（默认，custom all-reduce 开）：**

```bash
vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2 2>&1 | tee ~/exp-logs/m2_runA.log
```

起来后先确认开关状态，别默认它一定开着：

```bash
grep -iE "custom.?all.?reduce" ~/exp-logs/m2_runA.log
```

预期：出现启用相关行（具体措辞随 vLLM 版本变化，抓 custom allreduce 这个关键词）。如果出现的是禁用行并给了原因（比如 P2P 访问不完整、拓扑不支持），两组对比失去意义，直接按 requirement 兜底写"实例拓扑不支持 custom all-reduce"，Run B 不用跑。

计时。第一遍是热身，不记录：首个请求会触发 CUDA graph 缓存命中、文件页缓存加载这类一次性开销，TPOT 虚高，计入会污染对比。之后正式三遍：

```bash
python ~/bench_decode.py                      # 热身，数字丢弃
for i in 1 2 3; do python ~/bench_decode.py 2>&1 | tee -a ~/exp-logs/m2_runA_bench.log; done
```

每遍输出一行（字段含义见 1.3），形如 `tokens=256 TTFT=182.3ms TPOT=11.42ms decode_tok/s=87.6`。三个 TPOT 抄进记录表。

**Run B（强制走 NCCL）：** 先 `Ctrl+C` 关掉 Run A 的服务并确认端口释放（方法同 3.1 末尾），再启动。vLLM 0.34 只有 CLI 参数入口：

```bash
NCCL_DEBUG=INFO \
vllm serve Qwen/Qwen3-0.6B --tensor-parallel-size 2 --disable-custom-all-reduce \
  2>&1 | tee ~/exp-logs/m2_runB.log
```

启动日志出现 `Unknown vLLM environment variable detected: VLLM_USE_CUSTOM_ALL_REDUCE` 说明该版本没有这个环境变量，加上也不生效。启动后找 `non-default args` 那行，确认里面有 `'disable_custom_all_reduce': True`。若报 `unrecognized arguments: --disable-custom-all-reduce`，说明当前版本只认环境变量，改用环境变量，并用 `vllm serve --help | grep -i custom` 查该版本支持的写法。

同样 grep，这次**必须**看到 custom allreduce 被禁用的行，确认了开关才继续，否则两组都开着 custom 跑出来的差异毫无意义。然后热身一遍、正式三遍：

```bash
python ~/bench_decode.py
for i in 1 2 3; do python ~/bench_decode.py 2>&1 | tee -a ~/exp-logs/m2_runB_bench.log; done
```

两组数字分别记录，不跨组混。预期 Run B 的 TPOT 高于 Run A；真出现相反结果也照实记，步骤 4 分析时不硬凑。

| 现象 | 处理 |
|---|---|
| bench 报 Connection refused | 服务没起来或已被关掉，回窗口一确认 `Application startup complete` 后再跑 |
| grep 在 Run B 日志里找不到禁用行 | 开关没生效，两组配置相同，对比作废。检查用的是环境变量还是参数，换另一个入口重跑 |

### 3.3 模块 3：example 逐个跑

01 到 04 必跑，05/06/07 尽力。

先说清目录结构：每个 topic（如 `01_communicators`）下分若干子项（如 `01_allreduce`），子项里是 `c/`（部分还有 `python/`，本教程全跑 C 版）。在 topic 目录 `make` 会编该 topic 全部子项。二进制名跟目录名对不上（04 的二进制就叫 `allreduce_ub`），找不到时用 `find ~/nccl-src/docs/examples/<topic> -type f -executable -not -name "*.sh"` 列出编出来的可执行文件，以各子项 README 里的运行命令为准。example 自动使用全部可见 GPU，双卡正好，运行命令里不用指定卡。

怎么判断一个 example 跑成功了：这些程序验证正确性，不测性能，运行时间是秒级。退出码 0（跑完立刻 `echo $?` 看）且输出里有各 rank 的结果行，就算通过；失败时输出里会有 "Test NCCL failure" 或 ncclInvalidUsage 之类的报错，**把报错第一行抄进笔记**，报错第一行信息量最大。

**01_communicators（三个子项：单进程多卡 / pthread / MPI 一进程一卡）：**

看什么：同一个通信组用三种进程/线程组织方式建立，每种方式里 uniqueId（建通信器的握手凭证）的广播路径不同。

```bash
cd ~/nccl-src/docs/examples/01_communicators
make MPI=1 2>&1 | tee ~/exp-logs/m3_01_build.log
cd 01_multiple_devices_single_process/c
./multiple_devices_single_process 2>&1 | tee ~/exp-logs/m3_01.log
cd ../../02_one_device_per_pthread/c
./one_device_per_pthread 2>&1 | tee -a ~/exp-logs/m3_01.log
cd ../../..
mpirun --allow-run-as-root -np 2 ./03_one_device_per_process_mpi/c/one_device_per_process_mpi 2>&1 | tee -a ~/exp-logs/m3_01.log
```

`make` 不加 `MPI=1` 只编前两个子项，01 目录的 Makefile 默认跳过 MPI 那个，`mpirun` 会报 `could not access or execute an executable`、退出码 132。缺这个二进制时补 `cd 03_one_device_per_process_mpi/c && make` 单独编一次。

`mpirun -np 2` 起两个进程，每进程占一张卡，对应 rank0 和 rank1。

**02_point_to_point（ring 点对点互发）：**

看什么：两个 rank 沿环形互相收发，send/recv 成对出现，这是 pipeline 并行里相邻段通信的最小模式。

```bash
cd ~/nccl-src/docs/examples/02_point_to_point
make 2>&1 | tee ~/exp-logs/m3_02_build.log
cd 01_ring_pattern/c
./ring_pattern 2>&1 | tee ~/exp-logs/m3_02.log
```

**03_collectives（最小 all-reduce）：**

看什么：一次 `ncclAllReduce` 调用的完整参数，两卡数据求和后各自拿到全量结果。

```bash
cd ~/nccl-src/docs/examples/03_collectives
make 2>&1 | tee ~/exp-logs/m3_03_build.log
cd 01_allreduce/c
./allreduce 2>&1 | tee ~/exp-logs/m3_03.log
```

**04_user_buffer_registration（预注册用户 buffer）：**

看什么：通信前先把显存注册给 NCCL（`ncclCommRegister`），通信时 NCCL 直接读写这块显存，省掉内部中转拷贝。推理的权重和 KV cache 都是预分配的，这正是它们的使用方式。

```bash
cd ~/nccl-src/docs/examples/04_user_buffer_registration
make 2>&1 | tee ~/exp-logs/m3_04_build.log
cd 01_allreduce/c
./allreduce_ub 2>&1 | tee ~/exp-logs/m3_04.log
```

**05_symmetric_memory（对称内存，要求系统 NCCL ≥ 2.27）：** 先看 2.4 记进记录表的 libnccl-dev 版本，低于 2.27 整个跳过，笔记写明版本号。版本够才跑：

```bash
cd ~/nccl-src/docs/examples/05_symmetric_memory
make 2>&1 | tee ~/exp-logs/m3_05_build.log
cd 01_allreduce/c
./allreduce_sm 2>&1 | tee ~/exp-logs/m3_05.log
cd ../../02_allgather/c
./allgather_ce 2>&1 | tee -a ~/exp-logs/m3_05.log
```

看什么：各卡把 buffer 注册成对称内存窗口后，通信路径为小消息低延迟优化，这是 NCCL 面向推理场景的新机制。

**06_device_api（kernel 内直接通信，只跑 LSA 子项）：**

```bash
cd ~/nccl-src/docs/examples/06_device_api
make 2>&1 | tee ~/exp-logs/m3_06_build.log
cd 01_allreduce_lsa/c
./allreduce_lsa 2>&1 | tee ~/exp-logs/m3_06.log
```

看什么：通信发起点挪进 GPU kernel 内部（LSA：同机 GPU 间直接读写对方显存），省掉 kernel 外发起通信的启动开销。

06 的其余子项不跑，笔记里写明原因：`02_alltoall_gin`、`03_alltoall_hybrid` 依赖 GIN（GPU kernel 内直接发起的 RDMA 通信），单机没有 RDMA 网络；`05/06` 两个 CFT 子项涉及 multimem，Ampere 不支持。

**07_kernel_fusion（RMSNorm + all-reduce 融合，只跑 LSA 子项）：**

```bash
cd ~/nccl-src/docs/examples/07_kernel_fusion
make -C 01_rmsnorm_lsa/c 2>&1 | tee ~/exp-logs/m3_07_build.log
cd 01_rmsnorm_lsa/c
./rmsnorm_lsa 2>&1 | tee ~/exp-logs/m3_07.log
```

看什么：归一化计算和 all-reduce 融合进同一个 kernel，一次 launch 干完两件事，对应推理每层 forward 里的真实模式。

这里用 `make -C 01_rmsnorm_lsa/c` 只编这一个子项，其余子项要 multimem 或 GIN，编了也跑不了。01 编译失败（Ampere 不支持所需指令）属预期，把报错第一行抄进笔记即可。

**三条通用注意：**

- root 身份跑 mpirun 必须带 `--allow-run-as-root`，不带它 mpirun 直接报错退出。
- mpirun 启动后几十秒没有任何输出，多半是某个 rank 卡死，`Ctrl+C` 终止，把终止前最后一行输出抄进笔记。
- 三句话笔记（调了什么 API、教了什么、报错与解决）当场写进本地 `../test/m3_01.md` 到 `m3_07.md`。云主机输出在云主机屏幕上看着，笔记文件在本地写，笔记不用拉回。

### 3.4 收尾

关掉 vllm（窗口一 `Ctrl+C`），确认 `nvidia-smi` 显存回到 0MiB。tmux 会话可以 `Ctrl+B D` 离开留着，阶段 2 直接接回。**实例先别释放**，阶段 2 还要用。

## 步骤 4：将数据拉取到本地并分析

本地执行：

```bash
rsync -avP nccl:~/exp-logs/ ~/Project/NCCL-Practice/logs/
```

参数含义：`-a` 保留时间戳和权限，`-v` 列出每个传的文件，`-P` 显示进度且断点续传。源路径末尾的 `/` 表示"传目录里的内容"，不带 `/` 会把 exp-logs 目录本身套一层放进去，命令里已带上。rsync 只传新增和变过的文件，中途断了重跑同一条即可。

拉完对账，两边文件数应一致：

```bash
ls ~/Project/NCCL-Practice/logs/ | wc -l
ssh nccl 'ls ~/exp-logs | wc -l'
```

### 4.1 模块 1：日志分析

```bash
grep -E "Channel [0-9]+/" logs/m1_startup.log | head -30
grep -E "isAllCudaP2p|NET/(IB|Socket)" logs/m1_startup.log
grep -iE "Algorithm|Protocol" logs/m1_startup.log | head -20
```

- 第一条找通道建立行，形如 `Channel 00/02 : 0 1`。NCCL 2.27 以前的版本写作 `Channel 00 : 0[GPU0] -> 1[GPU1] via P2P/CUMEM`，之后不再给 `via` 字样，路径类型改看下一条。
- 第二条 `isAllCudaP2p 1` 表示两卡能直接读写对方显存；`isAllDirectP2p 0` 表示没走无中转的直连模式，PCIe 主机桥下的双卡正常如此。`NET/IB : No device found` 与 `NET/Socket` 只说明 NCCL 探测过网卡插件，单机双卡不走网络，这类行可以忽略。
- 第三条在 NCCL 2.30 上可能空手，算法与协议的自动选择不再逐行打印。要比较算法协议，用模块 4 的 `NCCL_ALGO` / `NCCL_PROTO` 变量实测。

产出格式：从 m1_startup.log 摘通道建立段几行，每行后面加一句注解，形如：

```
NCCL INFO Channel 00/02 : 0 1   # 通道 0 覆盖 rank0 与 rank1，单机路径正常
NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 1   # 显存可直接互访，未走直连模式
```

**请求时 NCCL 在不在干活：** 先明确一个坑——INIT、GRAPH 子系统只在初始化阶段打印，INFO 级别不为每次 collective 输出，所以请求期间日志没有新行属正常，拿"日志没动"当"NCCL 没活动"的证据会得出错误结论。取证三选一：

1. 发请求的同时开个窗口跑 `nvidia-smi dmon -s u`。它每秒打一行各卡利用率，请求处理期间两张卡的 sm 列（SM 利用率）应同步拉起。TP 推理每层都要两卡交换数据，一卡忙一卡闲说明没走 TP。
2. 加 `NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL` 重跑一次。COLL 子系统会为每次集合通信打一行记录，量很大，只为这次验证用，验完就关。
3. 用模块 2 的 Run B 反证：强制走 NCCL 后 decode 变慢，说明 NCCL 就在关键路径上。

### 4.2 模块 2：延迟对比分析

每组三个 TPOT 取中位数：三个数排序取中间那个。用中位数而不用平均值，因为偶发干扰（后台进程抢了一次 GPU）会把单个样本拉高一大截，平均值被这种尖峰带偏，中位数不受影响。

差异百分比按 Run B 为基准算：`(B中位 − A中位) ÷ B中位 × 100%`。例：Run A 中位 11.4ms，Run B 中位 15.2ms，(15.2−11.4)÷15.2 ≈ 25%。

结论句式限定死："decode 小消息场景下，custom all-reduce 比 NCCL ring 的 TPOT 低 X%"，不外推到 prefill 或大消息。

### 4.3 模块 3：笔记检查

逐份检查 `../test/m3_01.md` 到 `m3_07.md`：三句话是否齐全（调了什么 API、教了什么、报错与解决）；跑不通的子项是否写清原因（系统 NCCL 版本低于 2.27 / Ampere 架构不支持 / 缺多机 RDMA），原因要能对上 2.4 和 2.1 记录的实际版本与拓扑。

### 4.4 拼阶段 1 报告

对照 requirement.md 模块 1/2/3 的验收条目逐条勾：每条写"达成，证据在 logs/xxx.log 第几段"或"未达成 + 原因"。附 4.1 的注解日志片段、4.2 的对比数字、4.3 的笔记要点。

## 本阶段名词速查

| 名词                   | 含义                                                                                   |
|------------------------|----------------------------------------------------------------------------------------|
| rank                   | 每个通信参与进程的编号，双卡即 rank0、rank1                                            |
| 通信器（communicator） | NCCL 的"群组"对象，所有通信都要声明在哪个组里进行                                      |
| uniqueId               | 建通信器的握手凭证，全体 rank 拿到同一个才能互相找到                                   |
| collective             | 全体一起参与的通信操作，all-reduce 是最常见的一种                                      |
| all-reduce             | 每卡出一份数据，求和之后每卡都拿到完整结果                                             |
| P2P                    | GPU 直接读写另一张 GPU 的显存                                                          |
| CUMEM                  | CUDA 的虚拟内存分配机制，NCCL 用它把对端显存映射进本进程；2.27 以前日志写 via P2P/CUMEM，之后看 `isAllCudaP2p` |
| SHM                    | P2P 用不了时走主机内存中转                                                             |
| NVLink                 | NVIDIA 自有 GPU 间高速专线，比 PCIe 快一个量级                                         |
| TP（张量并行）         | 把每一层的权重切开放到两张卡上一起算                                                   |
| custom all-reduce      | vLLM 自研小消息归约 kernel，一次 kernel launch 直接读对端 buffer，绕开 NCCL ring       |
| LL / LL128 / Simple    | NCCL 三种传输协议：前两种把数据和标志打包传，低延迟适合小消息；Simple 吞吐高适合大消息 |
| Ring / Tree            | NCCL 集合通信算法：Ring 带宽最优、多跳延迟高；Tree 延迟低                              |
| TTFT                   | 请求发出到第一个 token 返回的时间，含 prefill                                          |
| TPOT                   | 第一个 token 之后每个 token 的平均间隔，decode 延迟指标                                |
| MPI                    | 多进程拉起工具，example 用它单机开 2 个 rank                                           |
| mpirun                 | MPI 的启动命令，`-np 2` 表示起两个进程                                                 |
| CUDA graph             | vLLM 把一串 kernel 录成图整体回放，启动开销小，首次捕获较慢                            |
| dmon                   | nvidia-smi 的逐秒利用率监控子命令                                                      |
