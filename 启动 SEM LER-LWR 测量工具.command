#!/bin/zsh

cd -- "$(dirname "$0")"
PYTHON="/Users/yiwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"

if [[ ! -x "$PYTHON" ]]; then
  osascript -e 'display dialog "未找到测量工具所需的运行环境。请在 Codex 中重新运行该项目。" buttons {"好"} default button "好"'
  exit 1
fi

exec "$PYTHON" ler_lwr_app.py
