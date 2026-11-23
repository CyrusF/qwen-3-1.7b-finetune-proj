export const lessons = [
  {
    id: "setup", nav: "环境检查", title: "环境检查",
    intro: "让 PyTorch 在 Apple GPU 上跑通一小段计算，再下载课程使用的 Qwen3-1.7B 权重。",
    flow: [
      ["Python", "用 Python 3.13 和 PyTorch 2.15.0 创建独立环境。"],
      ["MPS probe", "MPS（Metal Performance Shaders）是 PyTorch 在 Mac 上使用 Apple GPU（Graphics Processing Unit）的后端。"],
      ["Weights", "从 Hugging Face 下载固定 revision 的权重和 tokenizer。后续三章读取本地文件。"],
    ],
    checks: [
      ["磁盘空间", "模型约 4.08 GB。下载前至少保留 5 GiB。计划跑完整课程时，安装和下载前预留约 12 GiB。"],
      ["MPS", "探针会跑 FP32 matmul、BF16 SDPA（Scaled Dot-Product Attention）、AdamW 和 multinomial sampling，并检查实际设备和梯度。"],
      ["权重", "setup 成功后，结果里会写入权重路径及 Hugging Face revision。"],
    ],
    info: ["统一内存", "CPU（Central Processing Unit）和 GPU 共享本机内存。课程用 torch.mps 查询分配量和推荐工作集。它们与原 CUDA 日志里的 VRAM peak 是不同指标。"],
    empty: "运行左侧命令，查看 MPS 探针、版本和权重路径。你可以先浏览后续章节和历史曲线。",
  },
  {
    id: "parity", nav: "Forward parity", title: "Forward parity",
    intro: "比较自定义实现与 Hugging Face 的 forward pass，验证相同输入和精度下的 logits 是否在容差内。",
    flow: [
      ["Tokenize", "同一个 tokenizer 生成 input_ids、attention_mask 和 position_ids。"],
      ["Forward", "两个实现都使用 FP32（32-bit Floating Point）和 eager attention。"],
      ["Compare", "对所有非 padding 位置逐元素比较 logits，再检查 KV cache（Key–Value Cache）。"],
    ],
    checks: [
      ["HF / single prompt", "用同一句 prompt 比较 Hugging Face 和自定义 forward。"],
      ["HF / ragged batch", "把长短不同的 prompt 放入同一个 batch，只比较非 padding 位置。"],
      ["KV cache / single prompt", "逐 token 更新 KV cache，并与一次性 forward 比较。"],
      ["KV cache / ragged batch", "重复 cache 检查，验证 left padding 和 position IDs。"],
    ],
    info: ["logits 数据说明", "logits 是模型给词表中每个 token 的未归一化分数。HF（Hugging Face）是本章的 reference implementation。四项检查的容差均为 1e-3。"],
    empty: "运行左侧命令。参考模型和自定义模型会依次加载，最后显示四项最大 logits 误差。",
  },
  {
    id: "sft", nav: "LoRA SFT", title: "小规模 LoRA SFT",
    intro: "用 LoRA（Low-Rank Adaptation）做 SFT（Supervised Fine-Tuning），用已知的无 e 回答监督模型。",
    flow: [
      ["Labels", "prompt token 的 label 是 -100。回答 token 与 EOS（End of Sequence）参与 loss。"],
      ["Loss", "把预测移动一位，让每个 token 学会预测下一个目标 token。"],
      ["Adapter", "冻结 base weights，只为 LoRA 矩阵 A 和 B 更新参数。"],
    ],
    checks: [
      ["Data split", "从已有 split 按 token 数取 4 条训练样本和 2 条验证样本。跑 1 个 epoch。"],
      ["Training loss", "每个 batch 记录 loss、gradient norm 和监督 token 数。"],
      ["Checkpoint", "LoRA B 从零开始。训练后 B 的 L2 norm 大于零，并保存 latest.pt。"],
    ],
    info: ["Adapter 数据说明", "MPS 训练用 BF16（Brain Floating Point 16-bit）base weights 和 FP32 adapter。Validation loss 衡量目标文本的预测效果，回答质量在最后一章用独立 eval 衡量。"],
    empty: "运行小规模 SFT 后可查看 validation loss、LoRA 更新量和真实样本的 token labels。",
  },
  {
    id: "grpo-one", nav: "GRPO 单步", title: "GRPO 单步",
    intro: "用 GRPO（Group Relative Policy Optimization）从同一个 prompt 采样四个回答，再执行一次 optimizer step。",
    flow: [
      ["Rollout", "从同一个 prompt 采样四个候选。训练和采样使用同一张 no-e token mask。"],
      ["Advantage", "用每个 reward 减去组均值，再除以组内标准差。"],
      ["Update", "结合 advantage、policy ratio 和 KL（Kullback–Leibler Divergence）执行一次更新。"],
    ],
    checks: [
      ["Sampling", "加载小规模 SFT 的 adapter。最多生成 16 个新 token，记录每个 candidate。"],
      ["Reward", "固定教学 reward 为 0、0.25、0.75、1。无需 API（Application Programming Interface）key。"],
      ["Optimizer step", "确认 gradient norm 和 adapter 参数变化都有限且大于零，保存 step_0001.pt。"],
    ],
    info: ["为什么看 advantage", "同组相对分数决定哪些回答要提高概率。KL 用于限制 policy 相对起始 adapter 的漂移。"],
    empty: "先跑小规模 SFT，再选择它的 adapter。运行后这里会展示真实候选和参数更新。",
  },
  {
    id: "grpo300", nav: "300-step 复现", title: "300-step 复现",
    intro: "在 MPS 上重跑仓库的两条 judge-based GRPO 路线，对照 Base 和 SFT 180 的学习曲线。",
    flow: [
      ["Step 0", "用同一份 eval set 和相同规则测量起点。"],
      ["Train", "每步采样 4 prompts × 8 candidates，使用 gpt-6-luna judge 打分。"],
      ["Eval", "每 25 steps 在 50 条 held-out prompts 上用 4 个 seeds 评估并保存 checkpoint。"],
    ],
    checks: [
      ["Judge", "在仓库根目录的 .env 或终端配置 OPENAI_API_KEY。先用一条样本确认账号可以调用 gpt-6-luna。"],
      ["Base / SFT 180", "Base 从新 LoRA adapter 开始。SFT 180 需要先训练和选择 best.pt。"],
      ["完整复现", "每条路线约 12,200 次 judge 请求，另计重试。终点需要 step_0300.pt 和 step 300 eval report。"],
    ],
    info: ["Usable rate", "回答通过 no-e check，且 coherence 和 responds 均达到 4/5 才计入 usable。rl_check 统计有 usable 候选的 prompts 占比及组内 reward 差异。judge_check 用已知好坏样本检验 judge 的评分标准。"],
    empty: "运行后可查看 step、judge reward、KL 和最近回答。原 CUDA 结果可先在下方浏览。",
  },
];
