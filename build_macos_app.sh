#!/bin/zsh
# Build a distributable macOS app from this source folder.
set -euo pipefail

APP_DIRECTORY="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ "$(basename -- "$APP_DIRECTORY")" == "lwr:ler" ]]; then
  PROJECT_DIRECTORY="$(cd -- "$APP_DIRECTORY/.." && pwd)"
else
  PROJECT_DIRECTORY="$APP_DIRECTORY"
fi
RUNTIME="/Users/yiwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
BUILD_DIRECTORY="$PROJECT_DIRECTORY/.sem-measure-build"
BUILD_ENV="$BUILD_DIRECTORY/venv"
SOURCE_LINK="$BUILD_DIRECTORY/source"

if [[ ! -x "$RUNTIME" ]]; then
  print -u2 "未找到项目构建所需的 Python 运行环境。"
  exit 1
fi

mkdir -p "$BUILD_DIRECTORY"
ln -sfn "$APP_DIRECTORY" "$SOURCE_LINK"

if [[ ! -x "$BUILD_ENV/bin/python3" ]]; then
  "$RUNTIME" -m venv --system-site-packages "$BUILD_ENV"
  "$BUILD_ENV/bin/python3" -m pip install --upgrade pip pyinstaller
fi

"$BUILD_ENV/bin/python3" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "SEM measure" \
  --icon "$SOURCE_LINK/assets/sem-measure-icon.icns" \
  --add-data "$SOURCE_LINK/vendor:vendor" \
  --distpath "$PROJECT_DIRECTORY/release" \
  --workpath "$BUILD_DIRECTORY/pyinstaller" \
  --specpath "$BUILD_DIRECTORY" \
  "$SOURCE_LINK/ler_lwr_app.py"
