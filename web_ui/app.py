"""
Flask application for Donkey vs Elephant — side-by-side political text generation.

Loads two fine-tuned GPT-2 models (Democratic "donkey" and Republican "elephant")
and serves a comparison UI where users can enter prompts and see how each model
responds to the same input.

Launch:  python -m web_ui.app
"""

import os
import json
import logging
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DONKEY_MODEL_DIR = PROJECT_ROOT / "models" / "donkey"
ELEPHANT_MODEL_DIR = PROJECT_ROOT / "models" / "elephant"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("web_ui")

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading (deferred until first request)
# ---------------------------------------------------------------------------
_models: dict = {}  # {"donkey": (model, tokenizer), "elephant": (model, tokenizer)}
_models_loaded: bool = False
_load_error: str | None = None


def _load_models() -> None:
    """Attempt to load both GPT-2 models.  Sets module-level state."""
    global _models, _models_loaded, _load_error

    if _models_loaded:
        return

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        _load_error = (
            "The 'transformers' library is not installed. "
            "Run: pip install -r requirements.txt"
        )
        log.warning(_load_error)
        _models_loaded = True
        return

    for name, path in [("donkey", DONKEY_MODEL_DIR), ("elephant", ELEPHANT_MODEL_DIR)]:
        if not path.exists() or not any(path.iterdir()):
            log.info("Model directory %s missing or empty — skipping %s", path, name)
            continue
        try:
            log.info("Loading %s model from %s …", name, path)
            tokenizer = GPT2Tokenizer.from_pretrained(str(path))
            model = GPT2LMHeadModel.from_pretrained(str(path))
            model.eval()
            _models[name] = (model, tokenizer)
            log.info("Loaded %s model successfully.", name)
        except Exception as exc:
            log.warning("Failed to load %s model: %s", name, exc)

    if not _models:
        _load_error = (
            "No trained models found. Train them first with: "
            "python -m model_training.train --both"
        )
        log.warning(_load_error)

    _models_loaded = True


# ---------------------------------------------------------------------------
# Text generation helper
# ---------------------------------------------------------------------------
def _generate(name: str, prompt: str, temperature: float, max_length: int, top_k: int) -> str:
    """Generate text from the named model, or return a demo placeholder."""
    import torch

    if name not in _models:
        return _demo_response(name, prompt)

    model, tokenizer = _models[name]

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            temperature=max(temperature, 0.01),  # avoid zero
            top_k=top_k if top_k > 0 else 50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the original prompt from the front so we return only new text
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return generated.strip()


def _demo_response(name: str, prompt: str) -> str:
    """Return a placeholder when the real model is unavailable."""
    label = "Democratic (Donkey)" if name == "donkey" else "Republican (Elephant)"
    return (
        f"[Demo mode — {label} model not yet trained]\n\n"
        f'Prompt received: "{prompt}"\n\n'
        "Train the models to see real generated responses:\n"
        "  python -m model_training.train --both"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    _load_models()
    return jsonify({
        "status": "ok",
        "models": {
            "donkey": "donkey" in _models,
            "elephant": "elephant" in _models,
        },
        "demo_mode": not _models,
        "error": _load_error,
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    _load_models()

    data = request.get_json(silent=True) or {}
    system_prompt = data.get("system_prompt", "").strip()
    topic_prompt = data.get("topic_prompt", "").strip()
    user_prompt = data.get("user_prompt", "").strip()

    if not user_prompt and not topic_prompt:
        return jsonify({"error": "Please provide a prompt."}), 400

    # Build the full prompt string
    parts = [p for p in [system_prompt, topic_prompt, user_prompt] if p]
    full_prompt = "\n".join(parts)

    temperature = float(data.get("temperature", 0.9))
    max_length = int(data.get("max_length", 150))
    top_k = int(data.get("top_k", 50))

    # Clamp values
    temperature = max(0.01, min(temperature, 2.0))
    max_length = max(10, min(max_length, 512))
    top_k = max(0, min(top_k, 100))

    donkey_text = _generate("donkey", full_prompt, temperature, max_length, top_k)
    elephant_text = _generate("elephant", full_prompt, temperature, max_length, top_k)

    return jsonify({
        "donkey": donkey_text,
        "elephant": elephant_text,
        "prompt_used": full_prompt,
        "params": {
            "temperature": temperature,
            "max_length": max_length,
            "top_k": top_k,
        },
        "demo_mode": not _models,
    })


# ---------------------------------------------------------------------------
# Entry-point for `python -m web_ui.app`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting Donkey vs Elephant web UI …")
    _load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
