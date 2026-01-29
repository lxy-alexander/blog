#!/usr/bin/env bash
set -e

TIME_STR=$(date "+%Y-%m-%d %H:%M:%S")
MSG="fix: format @ $TIME_STR"

echo "==> 1) Biome auto-fix..."
npx biome check --write ./src

echo "==> 2) Biome CI check..."
npx biome ci ./src

echo "==> 3) Git add..."
git add ./src

echo "==> 4) Git commit..."
git commit -m "$MSG" || {
  echo "Nothing to commit."
  exit 0
}

echo "==> 5) Git push..."
git push

echo "✅ Done: $MSG"

