#!/bin/bash
# ================================================
# 自动提交并推送所有 .md 修改
# 用法：
#   ./push_all.sh "你的 commit 信息"
# 如果不传参数，会使用默认时间戳信息。
# ================================================

set -e

# 获取 commit 信息
if [ -z "$1" ]; then
  msg="auto commit: $(date '+%Y-%m-%d %H:%M:%S')"
else
  msg="$1"
fi

# 显示当前分支
branch=$(git symbolic-ref --short HEAD)
echo "当前分支: $branch"

# 只添加 .md 文件
git add '*.md'

# 提交
git commit -m "$msg" || echo "没有 .md 改动需要提交"

# 推送
git push origin "$branch"

echo "已推送至远程仓库 ($branch)"
