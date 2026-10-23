#!/bin/bash
# Bosch Budapest BP201/205 冰水機房 BMS 資料取得（P8／P10 用）
#
# 2024 全年（CC BY 4.0）：Zenodo 12590466（01–06）＋14591934（07–12），逐檔 md5 依 Zenodo API 核對。
#   用法：bash scripts/fetch_bosch.sh   → 下載到 data/raw/bosch/RBHU-2024-<月>.zip（已存在就跳過）
#
# 2025-01～07（P10 跨年基準用）：Kaggle 競賽「AI-Based Modeling for Energy-Efficient Buildings」資料頁。
#   需登入 Kaggle 並接受競賽規則後手動下載，放到
#   data/raw/bosch/kaggle2025/ai-based-modeling-for-energy-efficient-buildings.zip
#   該 zip 的 md5 為 a7a7ac1b4b0e03c5ed98de13e327d687（2026-09 下載時記錄）。
#
# 解壓只取 B205WC* 與 B106WS01*，做法見 docs/P8_PLAN.md 與 reports/P8_REPORT.md、P10_REPORT.md。
set -u
dest="$(cd "$(dirname "$0")/.." && pwd)/data/raw/bosch"
mkdir -p "$dest" && cd "$dest"
for pair in "12590466 01 02 03 04 05" "14591934 07 08 09 10 11 12"; do
  set -- $pair; rec=$1; shift
  for m in "$@"; do f="RBHU-2024-$m.zip"; [[ -s $f ]] && continue
    curl -sL --retry 3 -o "$f" "https://zenodo.org/records/$rec/files/$f?download=1"
    want=$(curl -s "https://zenodo.org/api/records/$rec" | python3 -c "import json,sys;print({x['key']:x['checksum'] for x in json.load(sys.stdin)['files']}['$f'].split(':')[1])")
    got=$(md5 -q "$f"); [[ $want == $got ]] && echo "OK $f" || { echo "MD5 FAIL $f"; mv "$f" "$f.bad"; }
  done
done
echo ALLDONE
