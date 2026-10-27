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
VENDOR_STAGE="$BUILD_DIRECTORY/vendor"

if [[ ! -x "$RUNTIME" ]]; then
  print -u2 "未找到项目构建所需的 Python 运行环境。"
  exit 1
fi

mkdir -p "$BUILD_DIRECTORY"
ln -sfn "$APP_DIRECTORY" "$SOURCE_LINK"

"$RUNTIME" - "$APP_DIRECTORY/vendor" "$VENDOR_STAGE" <<'PY'
from pathlib import Path
import shutil
import sys

source, target = map(Path, sys.argv[1:])
if target.exists():
    shutil.rmtree(target)
shutil.copytree(source, target, ignore=shutil.ignore_patterns(
    ".DS_Store", "__pycache__", "*.pyc", "*.baiduyun.uploading.cfg",
))
PY

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
  --add-data "$VENDOR_STAGE:vendor" \
  --distpath "$PROJECT_DIRECTORY/release" \
  --workpath "$BUILD_DIRECTORY/pyinstaller" \
  --specpath "$BUILD_DIRECTORY" \
  "$SOURCE_LINK/ler_lwr_app.py"

"$RUNTIME" - "$PROJECT_DIRECTORY/release/SEM measure.app/Contents/Info.plist" "$SOURCE_LINK/ler_lwr_app.py" <<'PY'
import ast
from pathlib import Path
import plistlib
import sys

plist_path, source_path = map(Path, sys.argv[1:])
source = ast.parse(source_path.read_text())
version = next(ast.literal_eval(node.value) for node in source.body
               if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "APP_VERSION" for target in node.targets))
info = plistlib.loads(plist_path.read_bytes())
info.update(CFBundleShortVersionString=version, CFBundleVersion=version)
plist_path.write_bytes(plistlib.dumps(info, sort_keys=False))
print(f"Built SEM measure {version}")
PY

codesign --force --deep --sign - "$PROJECT_DIRECTORY/release/SEM measure.app"
