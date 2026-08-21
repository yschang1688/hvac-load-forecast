#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "=== [1/3] P3 第二輪 lagged_only ==="
$PY - <<'PYEOF'
import json; from pathlib import Path
p=Path("config/pipeline.json"); c=json.loads(p.read_text(encoding="utf-8"))
c["modeling"]["weather_mode"]="lagged_only"
p.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding="utf-8"); print("weather_mode=lagged_only")
PYEOF
$PY src/run_p3.py
echo "=== [2/3] 還原 perfect_forecast 設定 ==="
$PY - <<'PYEOF'
import json; from pathlib import Path
p=Path("config/pipeline.json"); c=json.loads(p.read_text(encoding="utf-8"))
c["modeling"]["weather_mode"]="perfect_forecast"
p.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding="utf-8"); print("restored")
PYEOF
echo "=== [3/3] P4 漂移監控與重訓 ==="
$PY src/run_p4.py
echo "=== 全部完成 ==="
