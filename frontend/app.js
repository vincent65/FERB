(function () {
  const API_BASE = window.FERB_API_BASE || "";

  const form = document.getElementById("chatForm");
  const input = document.getElementById("input");
  const submitBtn = document.getElementById("submit");
  const messagesEl = document.getElementById("messages");

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function nl2br(text) {
    return escapeHtml(text).replace(/\n/g, "<br>");
  }

  function addMessage(role, content, options = {}) {
    const div = document.createElement("div");
    div.className = "message message--" + role + (options.loading ? " message--loading" : "");
    div.setAttribute("data-role", role);

    const bubble = document.createElement("div");
    bubble.className = "message__bubble";
    if (options.markdown) {
      bubble.innerHTML = nl2br(content);
    } else {
      bubble.innerHTML = "<p>" + nl2br(content) + "</p>";
    }
    div.appendChild(bubble);
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function setMessageContent(el, content) {
    const bubble = el.querySelector(".message__bubble");
    if (!bubble) return;
    el.classList.remove("message--loading");
    bubble.innerHTML = "<p>" + nl2br(content) + "</p>";
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendMessage(text) {
    if (!text.trim()) return;

    addMessage("user", text.trim());
    input.value = "";
    input.style.height = "auto";

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
