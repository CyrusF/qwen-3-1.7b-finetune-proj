import { lessons } from "/lessons.js";
import { element, statusLabel, renderResult, renderHistory, rewardExercise, routeNames } from "/results.js";

const get = id => document.getElementById(id);
const python = ".venv/bin/python -m mps_lab ";
const runChoice = new Map();
const adapters = new Map();
let lesson;
let overview;
let detail;
let history;
let includeHistorical = true;
let polling = false;
let commandKey = "";
let resultKey = "";
let historyKey = "";

async function api(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw Error("HTTP " + response.status);
  return response.json();
}

function quote(word) {
  return "'" + word.replaceAll("'", "'\"'\"'") + "'";
}
function localCommand(text) {
  const prefix = overview?.custom_dir ? "MPS_LAB_DIR=" + quote(overview.data_dir) + " " : "";
  return prefix + python + text;
}

function command(title, text, help) {
  const area = element("section", null, "command");
  if (title) area.append(element("h3", title));
  const shell = element("div", null, "command-box");
  const code = element("pre", text);
  const button = element("button", "复制");
  button.type = "button";
  button.setAttribute("aria-label", "复制" + (title || "运行命令"));
  button.onclick = async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "已复制";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = document.execCommand("copy") ? "已复制" : "已选中";
    }
    setTimeout(() => { button.textContent = "复制"; }, 1600);
  };
  shell.append(code, button);
  area.append(shell);
  if (help) area.append(element("p", help, "command-note"));
  return area;
}

function selectMenu(label, options, selected, onchange) {
  const wrapper = element("label", label, "input-row");
  const list = element("select");
  for (const [id, text] of options) {
    const option = element("option", text);
    option.value = id;
    option.selected = id === selected;
    list.append(option);
  }
  list.onchange = () => onchange(list.value);
  wrapper.append(list);
  return wrapper;
}

function adapterCommand(stage, args, title) {
  const choices = overview?.runs.filter(run => run.stage === stage && run.checkpoint) || [];
  const selected = choices.find(run => run.id === adapters.get(stage)) || choices[0];
  const area = element("div", null, "command");
  if (selected) {
    area.append(selectMenu("选择 " + stage + " checkpoint", choices.map(run => [
      run.id, run.id + " / " + run.checkpoint.split("/").at(-1),
    ]), selected.id, id => {
      adapters.set(stage, id);
      renderCommands(true);
      get("commands").querySelectorAll("select")[0]?.focus();
    }));
    area.append(command(title, localCommand(args + " --adapter " + quote(selected.checkpoint))));
  } else {
    area.append(element("p", args === "grpo-one"
      ? "先完成小规模 LoRA SFT。成功运行的 latest.pt 会出现在这里。"
      : "先完成 SFT 180。成功运行的 best.pt 会出现在这里。", "command-note"));
  }
  return area;
}

function renderCommands(force = false) {
  const choices = overview?.runs.filter(run => run.checkpoint).map(run => run.id) || [];
  const key = JSON.stringify([lesson.id, overview?.custom_dir, overview?.data_dir, choices]);
  if (key === commandKey && !force) return;
  commandKey = key;
  const root = get("commands");
  root.replaceChildren(element("h2", "运行命令"));
  switch (lesson.id) {
    case "setup":
      root.append(command("安装 Mac 依赖", "uv venv --python 3.13 .venv &&\nuv pip install --no-cache --python .venv/bin/python -r requirements-mps.txt", "在仓库根目录执行。"),
        command("运行探针并下载权重", localCommand("setup"), "如需换卷，在网页服务和终端命令中设置同一个 MPS_LAB_DIR。"));
      break;
    case "parity":
    case "sft":
      root.append(command(null, localCommand(lesson.id)));
      break;
    case "grpo-one":
      root.append(adapterCommand("sft", "grpo-one", "运行一次 GRPO update"));
      break;
    case "grpo300":
      root.append(command("Judge probe", localCommand("judge-check"), "先设置 OPENAI_API_KEY。该命令发送一条测试请求。"),
        command("路线 A · Base", localCommand("grpo300 base")),
        command("路线 B · 训练 SFT 180", localCommand("sft180")),
        adapterCommand("sft180", "grpo300 sft180", "路线 B · GRPO 300 steps"));
      break;
  }
}

function renderRunSelector(runs, selected) {
  const root = get("select-run");
  const values = runs.map(run => [
    run.id, [run.route ? routeNames[run.route] : run.stage, run.id, statusLabel(run.state)].join(" · "),
  ]);
  const key = JSON.stringify([values, selected]);
  if (key === root.dataset.key) return;
  root.dataset.key = key;
  root.replaceChildren();
  if (values.length) {
    root.append(selectMenu("选择本机运行", values, selected, id => {
      runChoice.set(lesson.id, id);
      refresh();
    }));
  }
}

function updateHistory() {
  if (lesson.id !== "grpo300") return;
  const key = JSON.stringify([includeHistorical, detail?.run?.id, detail?.evals, history]);
  if (key === historyKey) return;
  historyKey = key;
  renderHistory(get("history-charts"), history, detail, includeHistorical);
}

async function refresh() {
  if (polling) return;
  polling = true;
  const id = lesson.id;
  try {
    overview = await api("/api/overview");
    if (id !== lesson.id) return;
    renderCommands();
    const runs = overview.runs.filter(run => run.stage === id ||
      (id === "grpo300" && ["sft180", "judge-check"].includes(run.stage)));
    const selected = runs.find(run => run.id === runChoice.get(id)) || runs[0];
    renderRunSelector(runs, selected?.id);
    const value = selected ? await api("/api/runs/" + selected.id) : null;
    if (id !== lesson.id || (runChoice.has(id) && selected?.id !== runChoice.get(id)
        && runs.some(run => run.id === runChoice.get(id)))) return;
    detail = value;
    const key = JSON.stringify([id, value, id === "setup" ? overview : null]);
    if (key !== resultKey) {
      renderResult(get("result"), lesson, detail, overview);
      resultKey = key;
    }
    if (id === "grpo300" && !history) history = await api("/api/history");
    updateHistory();
    get("refresh").textContent = "刷新";
  } catch (error) {
    get("refresh").textContent = "读取失败，重试";
    get("refresh").title = error.message;
  } finally {
    polling = false;
    if (id !== lesson.id) refresh();
  }
}

async function loadSource() {
  const id = lesson.id;
  const container = get("source-body");
  if (!get("source").open || container.dataset.lesson === id) return;
  container.textContent = "读取源码…";
  try {
    const excerpts = await api("/api/source/" + id);
    if (id !== lesson.id) return;
    container.dataset.lesson = id;
    const tabs = element("div", null, "source-tabs");
    const path = element("p", null, "trace");
    const code = element("pre", null, "code");
    code.tabIndex = 0;
    const buttons = excerpts.map((ex, i) => {
      const button = element("button", ex.symbol, "quiet");
      button.onclick = () => show(i);
      tabs.append(button);
      return button;
    });
    function show(i) {
      const ex = excerpts[i];
      path.textContent = ex.path + ":" + ex.line;
      code.textContent = ex.text + (ex.truncated ? "\n…" : "");
      buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(i === index)));
      code.scrollTop = code.scrollLeft = 0;
    }
    container.replaceChildren(tabs, path, code);
    show(0);
  } catch {
    if (id === lesson.id) container.textContent = "源码读取失败。重新展开以重试。";
  }
}

const parityDiffNotes = {
  "scripts/check_parity.py": [
    "Parity runner",
    "Forward 在选定的设备上跑 FP32。Hugging Face reference 处理两种 batch 后释放，再加载自定义模型。两边 logits 搬到 CPU 比较，用 masked_fill 排除 padding，四项检查写入 report.json。",
  ],
  "core/devices.py": [
    "设备选择",
    "课程传入 mps。resolve_device() 先解析设备，再检查对应后端是否可用。只有 device.type 等于 cuda 时才会执行 CUDA 可用性检查。",
  ],
  "core/utils.py": [
    "本地 tokenizer",
    "get_encoder(weights_dir) 使用 setup 已下载的 tokenizer。Hugging Face reference 和自定义模型也从同一个 weights_dir 加载。",
  ],
};

async function loadParityDiff() {
  const section = get("parity-diff");
  const root = get("parity-diff-body");
  if (!section.open || root.dataset.loaded || root.getAttribute("aria-busy") === "true") return;
  root.textContent = "读取 Git diff…";
  root.setAttribute("aria-busy", "true");
  try {
    const { base, head, entry, files } = await api("/parity-diff.json");
    const comparison = element("a", base.slice(0, 7) + " → " + head.slice(0, 7));
    comparison.href = "https://github.com/CyrusF/qwen-3-1.7b-finetune-proj/compare/" + base + "..." + head;
    const revision = element("p", "上游与 MPS 适配的版本对比 ", "muted diff-description");
    revision.append(comparison);
    const entryCode = element("pre", entry.text, "code diff-entry");
    entryCode.tabIndex = 0;
    const tabs = element("div", null, "source-tabs");
    const note = element("p", null, "diff-description");
    const path = element("p", null, "trace");
    const code = element("pre", null, "code");
    code.tabIndex = 0;
    code.setAttribute("aria-label", "Git diff");
    const buttons = files.map((file, index) => {
      const button = element("button", parityDiffNotes[file.path][0], "quiet");
      button.type = "button";
      button.onclick = () => show(index);
      tabs.append(button);
      return button;
    });
    function show(index) {
      const file = files[index];
      note.textContent = parityDiffNotes[file.path][1];
      path.textContent = file.path;
      code.replaceChildren(...file.patch.trimEnd().split("\n").map(line => {
        const kind = /^(diff |index |new file |--- |\+\+\+ |@@)/.test(line) ? "diff-header"
          : line.startsWith("+") ? "diff-added" : line.startsWith("-") ? "diff-removed" : null;
        return element("span", line + "\n", kind);
      }));
      buttons.forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
      code.scrollTop = code.scrollLeft = 0;
    }
    root.replaceChildren(revision, element("p", "课程入口把 mps、权重目录和 report.json 路径传给 parity runner。", "diff-description"),
      element("p", entry.path + ":" + entry.line, "trace"), entryCode, tabs, note, path, code);
    root.dataset.loaded = "true";
    show(0);
  } catch {
    root.textContent = "Git diff 读取失败。重新展开以重试。";
  } finally {
    root.removeAttribute("aria-busy");
  }
}

function showLesson() {
  lesson = lessons.find(item => "#" + item.id === location.hash) || lessons[0];
  detail = null;
  resultKey = historyKey = "";
  get("lessons").replaceChildren(...lessons.map((item, i) => {
    const link = element("a");
    link.href = "#" + item.id;
    if (item === lesson) link.setAttribute("aria-current", "page");
    link.append(element("span", String(i).padStart(2, "0"), "number"), element("span", item.nav));
    return link;
  }));
  get("title").textContent = lesson.title;
  get("intro").textContent = lesson.intro;
  const note = get("pipeline-note");
  note.hidden = true;
  const pipeline = get("pipeline");
  pipeline.replaceChildren();
  for (const [i, [title, explanation]] of lesson.flow.entries()) {
    const button = element("button", title);
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-controls", "pipeline-note");
    button.onclick = () => {
      const expand = button.getAttribute("aria-expanded") === "false";
      pipeline.querySelectorAll("button").forEach(node => node.setAttribute("aria-expanded", "false"));
      button.setAttribute("aria-expanded", String(expand));
      note.hidden = !expand;
      note.textContent = explanation;
    };
    if (i) {
      const arrow = element("span", null, "connector");
      arrow.setAttribute("aria-hidden", "true");
      pipeline.append(arrow);
    }
    pipeline.append(button);
  }
  get("checks").replaceChildren(...lesson.checks.map(([name, text]) => {
    const line = element("div", null, "check-row");
    line.append(element("dt", name), element("dd", text));
    return line;
  }));
  get("explanation").replaceChildren(element("h2", lesson.info[0]), element("p", lesson.info[1]));
  get("source").open = false;
  get("source-body").replaceChildren();
  get("source-body").dataset.lesson = "";
  const diff = get("parity-diff");
  diff.hidden = lesson.id !== "parity";
  if (diff.hidden) diff.open = false;
  const exercise = get("exercise");
  exercise.replaceChildren();
  if (lesson.id === "grpo-one") rewardExercise(exercise);
  const past = get("history");
  past.replaceChildren();
  if (lesson.id === "grpo300") {
    const label = element("label", null, "muted");
    const input = element("input");
    input.type = "checkbox";
    input.checked = includeHistorical;
    input.onchange = () => {
      includeHistorical = input.checked;
      updateHistory();
    };
    const chartRoot = element("div");
    chartRoot.id = "history-charts";
    label.append(input, " 显示仓库 CUDA 曲线");
    past.append(element("div", null, "experiment-section"), label, chartRoot);
  }
  const next = lessons[lessons.indexOf(lesson) + 1];
  const link = get("next");
  link.hidden = !next;
  if (next) {
    link.textContent = "下一步 · " + next.nav;
    link.href = "#" + next.id;
  }
  get("select-run").replaceChildren();
  get("select-run").dataset.key = "";
  renderResult(get("result"), lesson, null, overview);
  renderCommands();
  refresh();
  window.scrollTo(0, 0);
}

get("refresh").onclick = refresh;
get("source").addEventListener("toggle", loadSource);
get("parity-diff").addEventListener("toggle", loadParityDiff);
window.addEventListener("hashchange", showLesson);
showLesson();
setInterval(refresh, 2200);
