(function () {
  const API_BASE = window.FERB_API_BASE || "";

  const form = document.getElementById("chatForm");
  const input = document.getElementById("input");
  const submitBtn = document.getElementById("submit");
  const messagesEl = document.getElementById("messages");
  const codePanel = document.getElementById("codePanel");
  const ideMeta = document.getElementById("ideMeta");
  let runningOptimize = false;
  let typingRunId = 0;

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function nl2br(text) {
    return escapeHtml(text).replace(/\n/g, "<br>");
  }

  function renderRichText(text) {
    // Tiny safe markdown subset: bold, italic, inline code + newlines.
    let html = escapeHtml(text || "");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  function addMessage(role, content, options = {}) {
    const div = document.createElement("div");
    div.className = "message message--" + role + (options.loading ? " message--loading" : "");
    div.setAttribute("data-role", role);

    const bubble = document.createElement("div");
    bubble.className = "message__bubble";
    if (options.markdown) {
      bubble.innerHTML = renderRichText(content);
    } else {
      bubble.innerHTML = "<p>" + renderRichText(content) + "</p>";
    }
    div.appendChild(bubble);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function updateIDE(metaText, codeText) {
    if (ideMeta && typeof metaText === "string") ideMeta.textContent = metaText;
    if (codePanel && typeof codeText === "string") codePanel.textContent = codeText;
  }

  function isNearBottom(el, thresholdPx = 48) {
    if (!el) return true;
    return el.scrollTop + el.clientHeight >= el.scrollHeight - thresholdPx;
  }

  function typeIntoCodePanel(fullText, opts = {}) {
    if (!codePanel) return;
    const text = String(fullText || "");
    const chunkSize = Math.max(1, Number(opts.chunkSize || 12));
    const tickMs = Math.max(8, Number(opts.tickMs || 16));

    const myRun = ++typingRunId;
    codePanel.classList.add("typing");
    codePanel.textContent = "";

    let i = 0;
    const timer = setInterval(() => {
      if (myRun !== typingRunId) {
        clearInterval(timer);
        return;
      }
      if (i >= text.length) {
        clearInterval(timer);
        codePanel.classList.remove("typing");
        return;
      }

      const keepScroll = isNearBottom(codePanel);
      codePanel.textContent += text.slice(i, i + chunkSize);
      i += chunkSize;
      if (keepScroll) codePanel.scrollTop = codePanel.scrollHeight;
    }, tickMs);
  }

  function setMessageContent(el, content) {
    const bubble = el.querySelector(".message__bubble");
    if (!bubble) return;
    el.classList.remove("message--loading");
    bubble.innerHTML = "<p>" + renderRichText(content) + "</p>";
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendMessage(text) {
    if (!text.trim()) return;

    addMessage("user", text.trim());
    input.value = "";
    input.style.height = "auto";

    const trimmed = text.trim();
    if (trimmed.toLowerCase().startsWith("/optimize ")) {
      const parsed = parseOptimizeCommand(trimmed);
      await runOptimizeStream(parsed.objective, parsed.options);
      return;
    }

    const loadingEl = addMessage("assistant", "", { loading: true });
    submitBtn.disabled = true;

    try {
      const res = await fetch(API_BASE + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text.trim() }),
      });
      const data = await res.json();
      if (data.reply) {
        setMessageContent(loadingEl, data.reply);
      } else {
        setMessageContent(loadingEl, "Sorry, something went wrong. " + (data.detail || ""));
      }
    } catch (err) {
      setMessageContent(
        loadingEl,
        "Could not reach the API. Is it running? Start with: cd api && uvicorn main:app --reload"
      );
    } finally {
      submitBtn.disabled = false;
      input.focus();
    }
  }

  function tokenizeCommand(cmd) {
    const re = /"[^"]*"|\S+/g;
    const out = [];
    const m = cmd.match(re) || [];
    for (const t of m) {
      if (t.startsWith("\"") && t.endsWith("\"") && t.length >= 2) out.push(t.slice(1, -1));
      else out.push(t);
    }
    return out;
  }

  function parseOptimizeCommand(cmd) {
    // Supported:
    // /optimize [--backend triton|reference] [--problem N] [--iters N] [--nproc N] [--model NAME] [--no-eval] <objective...>
    const tokens = tokenizeCommand(cmd);
    const options = {
      problem_id: 1,
      iterations: 3,
      model: "gpt-4o-mini",
      target_backend: "triton",
      nproc_per_node: 1,
      no_eval: false,
    };

    let i = 1; // skip /optimize
    while (i < tokens.length) {
      const tok = tokens[i];
      if (!tok.startsWith("--")) break;
      const key = tok.slice(2).toLowerCase();
      if (key === "no-eval") {
        options.no_eval = true;
        i += 1;
        continue;
      }
      const val = tokens[i + 1];
      if (val == null) break;
      if (key === "backend") options.target_backend = String(val);
      else if (key === "problem") options.problem_id = Number(val);
      else if (key === "iters") options.iterations = Number(val);
      else if (key === "nproc") options.nproc_per_node = Number(val);
      else if (key === "model") options.model = String(val);
      i += 2;
    }

    const objective = tokens.slice(i).join(" ").trim();
    return { objective, options };
  }

  async function runOptimizeStream(objective, options = {}) {
    if (!objective) {
      addMessage(
        "assistant",
        "Usage: /optimize [--backend triton|reference] [--problem N] [--iters N] [--nproc N] [--model NAME] [--no-eval] <objective>"
      );
      return;
    }
    if (runningOptimize) {
      addMessage("assistant", "An optimize run is already in progress.");
      return;
    }

    runningOptimize = true;
    submitBtn.disabled = true;
    const statusEl = addMessage("assistant", "Starting agentic optimization run...", { loading: true });

    try {
      const target_backend = options.target_backend || "triton";
      const problem_id = Number.isFinite(options.problem_id) ? options.problem_id : 1;
      const iterations = Number.isFinite(options.iterations) ? options.iterations : 3;
      const model = options.model || "gpt-4o-mini";
      const nproc = Number.isFinite(options.nproc_per_node) ? options.nproc_per_node : 1;
      const noEval = !!options.no_eval;

      const useModalEval = target_backend === "triton";

      const res = await fetch(API_BASE + "/optimize/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          objective,
          problem_id,
          iterations,
          model,
          target_backend,
          evaluator_python: window.FERB_EVAL_PYTHON || null,
          evaluator_command: noEval
            ? null
            : useModalEval
              ? `modal run scripts/modal_benchmark.py --problem ${problem_id} --candidate {candidate_path} --rows 1024 --cols 1024 --dtype float32 --warmup 3 --iters 10`
              : `python -m torch.distributed.run --nproc-per-node ${nproc} scripts/benchmark_candidate.py --problem ${problem_id} --candidate {candidate_path} --rows 1024 --cols 1024 --dtype float32 --warmup 3 --iters 10`,
        }),
      });

      if (!res.ok || !res.body) {
        const txt = await res.text();
        setMessageContent(statusEl, "Optimize stream failed: " + txt);
        return;
      }

      setMessageContent(
        statusEl,
        `Agent run started. Streaming steps below.\n\nbackend=${target_backend} problem=${problem_id} iters=${iterations} eval=${noEval ? "disabled" : (useModalEval ? "modal H100x8" : `local nproc=${nproc}`)}`
      );

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6);
          if (!raw) continue;
          let event;
          try {
            event = JSON.parse(raw);
          } catch {
            continue;
          }
          renderOptimizeEvent(event);
        }
      }
    } catch (err) {
      addMessage(
        "assistant",
        "Could not stream optimization. Ensure API is running and OPENAI_API_KEY is set."
      );
    } finally {
      runningOptimize = false;
      submitBtn.disabled = false;
      input.focus();
    }
  }

  function renderOptimizeEvent(event) {
    const t = event && event.type;
    if (!t) return;
    if (t === "run_started") {
      addMessage("assistant", `Run ${event.run_id} started.\nModel: ${event.model}\nIterations: ${event.iterations}`);
      return;
    }
    if (t === "iteration_started") {
      addMessage("assistant", `Let's try iteration ${event.iteration}...`);
      return;
    }
    if (t === "candidate_generated") {
      updateIDE(
        `iter ${event.iteration} · ${event.candidate_path || ""}`,
        ""
      );
      typeIntoCodePanel(event.candidate_code_preview || "", { chunkSize: 14, tickMs: 14 });
      addMessage(
        "assistant",
        `Iteration ${event.iteration}: candidate generated.\nPath: ${event.candidate_path}\n\n(Full code is in the Agent IDE pane.)\n\nAgent critique:\n${event.feedback_preview || ""}`
      );
      return;
    }
    if (t === "agent_thought") {
      addMessage("assistant", `Iteration ${event.iteration} attempt ${event.attempt} plan:\n${event.text || ""}`);
      return;
    }
    if (t === "quality_reject") {
      addMessage(
        "assistant",
        `Iteration ${event.iteration} attempt ${event.attempt} rejected by quality gate:\n- ${(event.issues || []).join("\n- ")}`
      );
      return;
    }
    if (t === "evaluation_started") {
      addMessage("assistant", `Iteration ${event.iteration}: running evaluator...`);
      updateIDE(`iter ${event.iteration} · evaluating…`, codePanel ? codePanel.textContent : "");
      return;
    }
    if (t === "evaluation_heartbeat") {
      const elapsed = Number.isFinite(event.elapsed_s) ? event.elapsed_s : 0;
      if (ideMeta) ideMeta.textContent = `iter ${event.iteration} · evaluating… ${elapsed}s`;
      return;
    }
    if (t === "evaluation_completed") {
      let metricsLine = "";
      try {
        const parsed = JSON.parse(event.stdout_preview || "{}");
        if (parsed && typeof parsed === "object" && "speedup" in parsed) {
          metricsLine =
            `\nallclose=${parsed.allclose} max_abs_diff=${parsed.max_abs_diff}` +
            `\nreference_ms=${parsed.reference_ms} candidate_ms=${parsed.candidate_ms}` +
            `\nspeedup=${parsed.speedup} score=${parsed.score}`;
        }
      } catch (_) {
        // stdout may not be JSON; ignore
      }
      addMessage(
        "assistant",
        `Iteration ${event.iteration}: score=${event.score}${metricsLine}\nstdout:\n${event.stdout_preview || ""}\nstderr:\n${event.stderr_preview || ""}`
      );
      return;
    }
    if (t === "blocked") {
      addMessage(
        "assistant",
        `Blocked at iteration ${event.iteration}.\nReason: ${event.reason || ""}\n\nHint: ${event.hint || ""}`
      );
      return;
    }
    if (t === "best_updated") {
      addMessage("assistant", `New best at iteration ${event.iteration}. score=${event.best_score}`);
      return;
    }
    if (t === "best_unchanged") {
      addMessage("assistant", `No improvement at iteration ${event.iteration}. Current best score=${event.best_score}`);
      return;
    }
    if (t === "run_completed") {
      const r = event.result || {};
      addMessage(
        "assistant",
        `Run complete.\nBest score: ${r.best_score}\nBest candidate: ${r.best_candidate_path}\nRun dir: ${r.run_dir}`
      );
      return;
    }
    if (t === "error") {
      addMessage("assistant", `Run error: ${event.detail || "Unknown error"}`);
    }
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    sendMessage(input.value);
  });

  input.addEventListener("input", function () {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 12 * 24) + "px";
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
})();
