export function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

const stateLabels = { running: "运行中", passed: "已通过", failed: "失败", interrupted: "已中断" };
export const routeNames = { base: "Base", sft180: "SFT 180" };
export function statusLabel(state) { return stateLabels[state] || state; }
const number = (value, digits = 4) => Number.isFinite(value) ? Number(value.toPrecision(digits)).toString() : "—";
const percent = value => Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "—";

function table(headings, rows) {
  const box = element("div", null, "table-wrap");
  const table = element("table");
  const header = table.createTHead().insertRow();
  for (const name of headings) {
    const th = element("th", name);
    th.scope = "col";
    header.append(th);
  }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const value of row) tr.append(element("td", value));
  }
  box.append(table);
  return box;
}

function disclosure(title, content) {
  const details = element("details");
  details.append(element("summary", title), content);
  return details;
}

function chart(title, series, xMax, range) {
  const box = element("section", null, "experiment-section");
  const canvas = element("canvas", null, "chart");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", title + "，横轴为 step，数据见下表");
  const values = series.flatMap(item => item.values.map(point => point.y));
  let [yMin, yMax] = range || [Math.min(...values), Math.max(...values)];
  if (yMin === yMax) yMax = yMin + 1;
  canvas.drawChart = () => {
    const { width, height } = canvas.getBoundingClientRect();
    if (width < 40 || height < 40) return;
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    const left = 48, top = 10, right = width - 14, bottom = height - 30;
    const px = x => left + (right - left) * x / xMax;
    const py = y => bottom - (bottom - top) * (y - yMin) / (yMax - yMin);
    ctx.font = "11px Menlo, monospace";
    ctx.fillStyle = "#4e6278";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const y = yMin + (yMax - yMin) * (i / 4);
      ctx.strokeStyle = "#bfcedd";
      ctx.beginPath();
      ctx.moveTo(left, py(y));
      ctx.lineTo(right, py(y));
      ctx.stroke();
      ctx.fillText(range ? Math.round(y * 100) + "%" : number(y, 3), left - 7, py(y) + 4);
    }
    ctx.textAlign = "center";
    for (const x of new Set([0, Math.round(xMax / 2), xMax])) {
      ctx.fillText(String(x), px(x), bottom + 18);
    }
    for (const item of series) {
      ctx.strokeStyle = ctx.fillStyle = item.color;
      ctx.lineWidth = 2;
      ctx.setLineDash(item.dashed ? [5, 4] : []);
      ctx.beginPath();
      item.values.forEach(({ x, y }, i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)));
      ctx.stroke();
      ctx.setLineDash([]);
      for (const { x, y } of item.values) {
        ctx.beginPath();
        ctx.arc(px(x), py(y), item.dashed ? 2 : 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };
  const legend = element("div", null, "chart-legend");
  for (const item of series) {
    const mark = element("span", item.name);
    mark.style.setProperty("--series", item.color);
    legend.append(mark);
  }
  box.append(element("h3", title), canvas, legend);
  requestAnimationFrame(canvas.drawChart);
  return box;
}
window.addEventListener("resize", () => document.querySelectorAll(".chart").forEach(canvas => canvas.drawChart()));

function memoryTable(memory) {
  return table(["MPS 内存", "GiB"], [
    ["Tensor allocated", number(memory.allocated_gib)],
    ["Driver allocated", number(memory.driver_gib)],
    ["Recommended working set", number(memory.recommended_gib)],
  ]);
}

export function renderResult(root, lesson, detail, overview) {
  const run = detail?.run;
  const data = detail?.report;
  const previousLogOpen = root.querySelector(".run-log")?.open;
  root.replaceChildren();
  if (!run) {
    root.append(element("span", "待运行", "status"), element("p", lesson.empty));
  } else {
    root.append(element("span", statusLabel(run.state), "status " + run.state));
    root.append(element("div", run.id, "trace"));
    if (run.state === "failed") {
      root.append(element("p", "命令退出码 " + run.exit_code + "。展开终端日志查看原因。"));
    } else if (run.state === "interrupted") {
      root.append(element("p", "实验已中断，已保存的输出仍可查看。再次执行命令会生成新的 run ID。"));
    }
  }

  if (lesson.id === "setup" && overview) {
    root.append(table(["当前环境", "值"], [
      ["可用空间", overview.free_gib + " GiB"],
      ["Qwen3-1.7B 权重", overview.weights_ready ? "已就绪" : "待下载"],
    ]));
    root.append(element("p", overview.data_dir, "trace"));
    if (data) {
      root.append(table(["MPS probe", "结果"], data.probes.map(probe => [probe, "PASS"])));
      root.append(element("p", "PyTorch " + data.torch + " · Transformers " + data.transformers));
      root.append(element("p", "revision " + data.revision, "trace"), memoryTable(data.memory));
    }
  }
  if (lesson.id === "parity" && data) {
    root.append(table(["检查", "Max logits diff", "结果"], data.checks.map(check => [
      check.name, number(check.max_diff, 6), check.passed ? "PASS" : "FAIL",
    ])));
    root.append(element("p", data.device.toUpperCase() + " / " + data.dtype + " / tol " + data.tolerance));
  }
  if (lesson.id === "sft" || run?.stage === "sft180") {
    const summary = data || detail?.summary;
    if (summary) {
      root.append(table(["指标", "值"], [
        ["Optimizer steps", summary.global_step], ["Initial validation loss", number(summary.val_loss_before)],
        ["Best validation loss", number(summary.best_val_loss)], ["Best epoch", summary.best_epoch],
        ...("adapter_delta_l2" in summary ? [["LoRA B update · L2", number(summary.adapter_delta_l2)]] : []),
      ]));
    }
    const train = detail?.metrics.filter(row => row.event === "train") || [];
    if (train.length) {
      root.append(chart("Training loss", [{
        name: "loss", color: "#174ac7", values: train.filter(r => r.loss != null).map(r => ({ x: r.global_step, y: r.loss })),
      }], train.at(-1).global_step));
      root.append(table(["Step", "Loss", "Grad norm", "Target tokens"], train.slice(-8).map(r => [
        r.global_step, number(r.loss), number(r.grad_norm), r.supervised_tokens,
      ])));
      const memory = train.at(-1).mps_memory;
      if (memory) root.append(memoryTable(memory));
    }
    if (detail?.tokens) {
      root.append(element("p", "灰色 token 属于 context，蓝色 token 参与 loss。"));
      for (const sample of detail.tokens) {
        const section = element("div");
        const tokens = element("div", null, "tokens");
        for (const token of sample.tokens) tokens.append(element("span", token.text, token.target ? "token target" : "token"));
        section.append(element("p", sample.prompt), element("p", sample.answer), tokens);
        root.append(disclosure(sample.split + " / " + sample.id, section));
      }
    }
  }
  if (lesson.id === "grpo-one" && data) {
    root.append(element("p", data.prompt), table(["Candidate", "Reward", "Advantage", "回答"],
      data.answers.map((row, i) => [i + 1, number(row.reward), number(row.advantage), row.answer || "（空回答）"])));
    root.append(table(["更新证据", "值"], [
      ["Optimizer step", data.step], ["Gradient norm", number(data.grad_norm)],
      ["LoRA update · L2", number(data.adapter_delta_l2)], ["KL", number(data.kl)],
      ["Sampling / training logprob gap", number(data.logprob_gap?.max)],
    ]));
    if (data.memory) root.append(memoryTable(data.memory));
  }

  if (run?.stage === "judge-check" && data) {
    root.append(element("p", [data.model, data.error].filter(Boolean).join(" · ")),
      element("p", (data.request_ids || []).join("、"), "trace"));
  }
  if (run?.stage === "grpo300") {
    const last = detail.metrics.at(-1);
    const progress = element("progress", null, "progress");
    progress.max = 300;
    progress.value = last?.step || 0;
    progress.setAttribute("aria-label", "GRPO 训练进度");
    root.append(element("p", routeNames[run.route] + " · " + progress.value + " / 300 steps"), progress);
    if (last) {
      root.append(table(["Step", "Mean reward", "KL", "Judge errors"], detail.metrics.slice(-6).map(row => [
        row.step, number(row.mean_reward), number(row.kl), row.judge_errors,
      ])));
      if (last.mps_memory) root.append(memoryTable(last.mps_memory));
    }
    if (detail.answers.length) {
      root.append(disclosure("最近 candidates", table(["Step", "Reward", "回答"], detail.answers.map(row => [
        row.step, number(row.reward), row.answer || "（空回答）",
      ]))));
    }
  }

  if (run?.checkpoint) root.append(element("p", run.checkpoint, "trace"));
  if (detail?.config) {
    const parameters = element("div");
    parameters.append(element("pre", JSON.stringify(detail.config, null, 2), "code"));
    root.append(disclosure("运行参数", parameters));
  }
  if (run) {
    const text = element("div");
    text.append(element("pre", detail.log || "正在等待终端输出。", "log"));
    const log = disclosure("终端日志", text);
    log.className = "run-log";
    log.open = previousLogOpen ?? (run.state === "running" || run.state === "failed");
    root.append(log);
  }
}

export function renderHistory(root, history, detail, includeHistorical) {
  root.replaceChildren();
  const series = [];
  const local = detail?.evals || [];
  const points = rows => rows.map(r => ({ x: r.step, y: r.usable_rate }));
  if (includeHistorical) {
    for (const [route, color] of [["base", "#637a99"], ["sft180", "#167665"]]) {
      if (history?.[route]?.length) series.push({
        name: "CUDA · " + routeNames[route], color, dashed: true, values: points(history[route]),
      });
    }
  }
  if (local.length) series.push({
    name: "本机 MPS · " + routeNames[detail.run.route], color: "#174ac7", values: points(local),
  });
  if (series.length) {
    root.append(chart("Usable rate", series, 300, [0, 1]));
    const steps = [...new Set(series.flatMap(item => item.values.map(point => point.x)))].sort((a, b) => a - b);
    root.append(disclosure("查看 eval 数据", table(["Step", ...series.map(item => item.name)], steps.map(step =>
      [step, ...series.map(item => percent(item.values.find(point => point.x === step)?.y))]))));
  }
  if (includeHistorical && local.length) {
    const final = local.find(row => row.step === 300);
    const reference = history?.[detail.run.route]?.find(row => row.step === 300);
    if (Number.isFinite(final?.usable_rate) && Number.isFinite(reference?.usable_rate)) {
      const delta = 100 * (final.usable_rate - reference.usable_rate);
      root.append(element("p", "Step 300 · 本机 " + percent(final.usable_rate) + " · 历史 " +
        percent(reference.usable_rate) + " · 差值 " + (delta > 0 ? "+" : "") + delta.toFixed(1) + " 个百分点", "muted"));
    }
  }
}

export function rewardExercise(root) {
  const box = element("div");
  const start = [0, 0.25, 0.75, 1];
  const sliders = [];
  const values = [];
  const summary = element("p", null, "muted");
  summary.setAttribute("aria-live", "polite");

  function update() {
    const rewards = sliders.map(input => Number(input.value));
    const mean = rewards.reduce((sum, score) => sum + score, 0) / rewards.length;
    const std = Math.sqrt(rewards.reduce((sum, score) => sum + (score - mean) ** 2, 0) / rewards.length);
    for (const [i, score] of rewards.entries()) values[i].textContent = ((score - mean) / Math.max(std, 1e-6)).toFixed(3);
    summary.textContent = "组均值 " + mean.toFixed(3) + " · 组内标准差 " + std.toFixed(3) +
      (std === 0 ? " · 同分时每个 advantage 都是 0。" : "");
  }
  box.append(element("p", "拖动任意 reward，观察组内 advantage。滑块只改这里的计算示例。"));
  for (const [i, score] of start.entries()) {
    const row = element("label", null, "reward-row");
    const range = element("input");
    Object.assign(range, { type: "range", min: "0", max: "1", step: "0.05", value: String(score) });
    range.addEventListener("input", update);
    const output = element("output", null, "numeric");
    sliders.push(range);
    values.push(output);
    row.append(element("span", "Reward " + (i + 1)), range, output);
    box.append(row);
  }
  const buttons = element("div", null, "source-tabs");
  for (const [label, preset] of [["设为同分", [0.5, 0.5, 0.5, 0.5]], ["重置 reward", start]]) {
    const button = element("button", label, "quiet");
    button.onclick = () => {
      sliders.forEach((slider, i) => { slider.value = preset[i]; });
      update();
    };
    buttons.append(button);
  }
  box.append(summary, buttons);
  const details = disclosure("练习：组内 advantage", box);
  details.open = true;
  root.append(details);
  update();
}
