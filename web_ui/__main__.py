"""Allow running the web UI as a module: python -m web_ui"""
from web_ui.app import app, _load_models

if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
