#!/usr/bin/env bash
# 一键同步：Windows 编辑 → WSL 运行（单向，Windows 为代码基准）
# 用法（在 WSL 里跑）：cd ~/legal_agent && bash sync.sh
set -euo pipefail

SRC="/mnt/d/legal_agent/"
DST="$HOME/legal_agent/"

if [ ! -d "$SRC" ]; then
    echo "✗ 找不到源目录 $SRC（确认在 WSL 里跑，且 D 盘已挂载）" >&2
    exit 1
fi

# 注意：WSL 的 venv 叫 venv（无点），Windows 的叫 .venv（有点），两个都要排除。
# 只排除其一，另一个会被 --delete 误删（2026-08-06 事故：漏了 venv 把 WSL venv 删了）。
EXCLUDES=(--exclude venv --exclude .venv --exclude node_modules --exclude dist \
          --exclude __pycache__ --exclude .env)

echo "同步: $SRC → $DST"
echo "排除: ${EXCLUDES[*]}"
rsync -av --delete "${EXCLUDES[@]}" "$SRC" "$DST"
echo "✓ 同步完成"