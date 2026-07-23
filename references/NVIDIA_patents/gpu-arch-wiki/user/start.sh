#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🔨 构建网页..."
python3 "$SCRIPT_DIR/build_arch.py"

echo "🚀 启动本地服务器..."
echo "📍 访问地址: http://localhost:8000"
echo "⏹️  按 Ctrl+C 停止"
cd "$SCRIPT_DIR/../web" && open http://localhost:8000 && python3 -m http.server 8000
