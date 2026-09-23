# hvac-load-forecast

建築冷卻負荷（chilled water）的多步時序預測，與它的驗收流程。

> **In brief** — A multi-building chilled-water load forecasting pipeline whose real subject is
> *whether the reported accuracy can be trusted*. One pipeline and one config file serve 465
> buildings with zero per-building code. Headline finding: the study's main conclusion flips on
> 2 of 45 evaluation windows, so it is reported as **not claimable**. Also quantifies why PSI
> drift monitoring fails in this domain (624 of 624 weeks alerted).

## 想看什麼，去哪裡

| 你想看 | 去這裡 |
|---|---|
| 為什麼一條「合理」的品質規則會製造 70% 誤報 | [P1 報告](reports/P1_REPORT.md) |
| 時序洩漏的第二種，靜態檢查抓不到的那種 | [P2 報告](reports/P2_REPORT.md) · [`src/features.py`](src/features.py) |
| 主結論取決於 45 個視窗中的 2 個 | [P3 報告](reports/P3_REPORT.md) |
| PSI 在非平穩場域為何失效，以及該用什麼替代 | [P4 報告](reports/P4_REPORT.md) |
| 多廠牌欄位方言怎麼歸一，schema 違規怎麼擋 | [P5 報告](reports/P5_REPORT.md) · [`src/contracts.py`](src/contracts.py) |
| 非顯而易見的設計決定與踩過的坑 | [設計筆記](docs/DESIGN_NOTES.md) |
| 三個結論的下一步：哪些做、哪些資料說不該做 | [P6 路線圖](docs/ROADMAP_P6.md) · [氣象預報 MCP 契約](docs/WEATHER_MCP_CONTRACT.md) · [特徵提案裁定](docs/DECISION_P6_FEATURES.md) |

資料源：[Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2)（CC BY-SA 4.0），
1,600+ 棟建築的逐時計量資料，本專案使用其中的冰水表、氣象與建物中繼資料。

## 這個專案在做什麼

不是做一個預測模型，是做**一套能跑在很多棟建築上的預測管線，以及判斷它的準確率能不能信的流程**。

- 一套 pipeline、一份設定檔（`config/pipeline.json`），N 棟建築各自產出模型與驗收報告
- 新增一棟建築不改程式碼

## 進度

- [x] **P1 資料契約與品質閘** — 6 條規則、三級判定、465 棟實跑（[報告](reports/P1_REPORT.md)）
- [x] **P2 防洩漏特徵管線** — 兩型時序洩漏各有守衛、rolling-origin 切分（[報告](reports/P2_REPORT.md)）
- [x] **P3 多步預測策略與架構對照** — 3 策略 × 4 架構 × 12 棟 × 4 folds，兩種天氣模式（[報告](reports/P3_REPORT.md)）
- [x] **P4 漂移監控與 retrain 自動化** — 624 個監控週、注入演練、對照組（[報告](reports/P4_REPORT.md)）
- [x] **P5 服務層與 API contract** — FastAPI＋SQLAlchemy＋PostgreSQL、多廠牌方言歸一（[報告](reports/P5_REPORT.md)）
- [ ] **P6 誤差監控與氣象預報契約** — 主訊號改誤差、PSI 降級；`forecast`／`noisy_forecast` 兩種新天氣模式；部署模擬重跑未做（[路線圖](docs/ROADMAP_P6.md)）

## 幾個結論

- **主結論取決於 2 個視窗**：45 個評估視窗中排除 2 個「冰機實質關機」的，
  模型對基準線就從 p=0.590 不可宣稱翻成 p<0.0001 可宣稱。因門檻未事前宣告，本專案採**不可宣稱**。
- **遞迴式多步預測比什麼都不做更差**（p=0.003）：誤差沿步長放大 2.25×，
  省下 23 個模型的維運成本買到的是劣於「抄上週同時刻」的結果。
- **PSI 在這個場域沒有鑑別力**：624 個監控週判了 624 次 alert，換三種參考期都一樣。
  該監控的是模型的誤差，不是模型的輸入。P6 已把誤差監控做成函式並以同一批 624 週離線重算：
  它有鑑別力（誤報 28–48%、偵測 51–67%，依基準期選法），但**固定早期基準會被季節性咬**。
- **完美氣象預報的價值量不出來**：perfect vs lagged 全部落在雜訊帶內。

## 快速開始

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
src/fetch_data.sh                      # 取資料（走 Git LFS API，附 SHA256 對帳）
.venv/bin/python src/run_quality.py    # P1：品質閘跑全部案場
.venv/bin/python src/run_p3.py         # P3：策略 × 架構對照
.venv/bin/python src/run_p4.py         # P4：部署模擬與漂移監控
.venv/bin/python src/analyze_p4_error.py   # P6：誤差監控 vs PSI 離線重算（不重跑模擬，秒級）
docker compose up -d                   # P5 需要 PostgreSQL
.venv/bin/uvicorn --app-dir src service:app   # P5：模型服務
.venv/bin/python -m pytest tests/ -q   # 61 則（14 則需 PostgreSQL，未啟動則 skip）
```

> 取資料為什麼不是 `curl raw.githubusercontent.com`：該 repo 用 Git LFS，直接抓只會得到
> 133 bytes 的指標檔——副檔名是 `.csv`、HTTP 回 200、檔案存在，但內容不是資料。
> `fetch_data.sh` 走 LFS batch API 取真檔並以 SHA256（即 LFS oid）逐檔對帳。

## 設計原則

1. **原始值不刪不改**：閘門只產生判定與標記，修補另存欄位並記錄補值方法。
2. **三級判定而非二分法**：Pass／Conditional／Reject。
3. **規則要能說出物理理由**：每條規則附 `rationale`，被質疑時答得出來。
4. **因果性分層**：`streaming_safe` 的規則只看當下與過去，生產可逐筆跑；其餘僅限批次剖析。
5. **閘門必須自證**：真實資料全過不是有效的證據，逐型植入缺陷才是。
