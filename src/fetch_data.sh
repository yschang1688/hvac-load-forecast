#!/usr/bin/env bash
# 取得 BDG2 資料（CC BY-SA 4.0, buds-lab/building-data-genome-project-2）
# repo 用 Git LFS，直接抓 raw.githubusercontent 只會得到 133 bytes 的指標檔——
# 那是一個會靜默通過的陷阱：副檔名是 .csv、curl 回 200、檔案存在，但內容不是資料。
# 故本腳本走 LFS batch API 取真檔，並以 SHA256 對帳（oid 就是內容雜湊）。
set -euo pipefail
REPO="https://github.com/buds-lab/building-data-genome-project-2.git"
BASE="https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/master"
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/raw}"
mkdir -p "$DEST"
declare -a FILES=(
  "data/meters/raw/chilledwater.csv|chilledwater.csv"
  "data/weather/weather.csv|weather.csv"
  "data/metadata/metadata.csv|metadata.csv"
)
for entry in "${FILES[@]}"; do
  path="${entry%%|*}"; out="${entry##*|}"
  ptr="$(curl -sSL -m 60 "$BASE/$path")"
  oid="$(printf '%s' "$ptr" | sed -n 's/^oid sha256://p')"
  size="$(printf '%s' "$ptr" | sed -n 's/^size //p')"
  if [ -z "$oid" ]; then echo "!! $out 非 LFS 指標檔，直接落盤"; printf '%s' "$ptr" > "$DEST/$out"; continue; fi
  if [ -f "$DEST/$out" ] && [ "$(shasum -a 256 "$DEST/$out" | cut -d' ' -f1)" = "$oid" ]; then
    echo "== $out 已存在且雜湊相符，跳過"; continue
  fi
  href="$(curl -s -m 60 -X POST "$REPO/info/lfs/objects/batch" \
      -H "Accept: application/vnd.git-lfs+json" -H "Content-Type: application/vnd.git-lfs+json" \
      -d "{\"operation\":\"download\",\"transfers\":[\"basic\"],\"objects\":[{\"oid\":\"$oid\",\"size\":$size}]}" \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['objects'][0]['actions']['download']['href'])")"
  echo ">> $out ($(echo "$size" | awk '{printf "%.1f MB", $1/1048576}'))"
  curl -sSL -m 1800 -o "$DEST/$out" "$href"
  got="$(shasum -a 256 "$DEST/$out" | cut -d' ' -f1)"
  [ "$got" = "$oid" ] && echo "   SHA256 OK" || { echo "   !! SHA256 不符: $got != $oid"; exit 1; }
done
echo "全部完成 → $DEST"
