# hvac-load-forecast

建築冷卻負荷（chilled water）的多步時序預測，與它的驗收流程。

> **In brief** — A multi-building chilled-water load forecasting pipeline whose real subject is
> *whether the reported accuracy can be trusted*. One pipeline and one config file serve 465
> buildings with zero per-building code. Headline finding: the study's main conclusion flips on
> 2 of 45 evaluation windows, so it is reported as **not claimable**. Also quantifies why PSI
> drift monitoring fails in this domain (624 of 624 weeks alerted), replaces it with error-based
> monitoring computed in SQL, and shows in an end-to-end replay on 6 real buildings that a
> same-period-last-year baseline cuts false alarms from 23% to 16% versus a rolling baseline.

## 想看什麼，去哪裡

| 你想看 | 去這裡 |
|---|---|
| 為什麼一條「合理」的品質規則會製造 70% 誤報 | [P1 報告](reports/P1_REPORT.md) |
| 時序洩漏的第二種，靜態檢查抓不到的那種 | [P2 報告](reports/P2_REPORT.md) · [`src/features.py`](src/features.py) |
| 主結論取決於 45 個視窗中的 2 個，排除關機視窗也救不回來 | [P3 報告](reports/P3_REPORT.md)（附錄二） · [`src/operating_state.py`](src/operating_state.py) |
| PSI 在非平穩場域為何失效，以及該用什麼替代 | [P4 報告](reports/P4_REPORT.md) |
| 多廠牌欄位方言怎麼歸一，schema 違規怎麼擋 | [P5 報告](reports/P5_REPORT.md) · [`src/contracts.py`](src/contracts.py) |
| 預測怎麼跟實際值對帳、每週 skill 怎麼用 SQL 算、基準為何用往年同期 | [P7 報告](reports/P7_REPORT.md) · [`src/db.py`](src/db.py) |
| 非顯而易見的設計決定與踩過的坑 | [設計筆記](docs/DESIGN_NOTES.md) |
| 商用系統的輸入與功能，對照本專案做了什麼、沒做什麼 | [領域參考](docs/DOMAIN_REFERENCE.md) |
| 誤差監控取代 PSI、氣象預報怎麼接進來 | [P6 路線圖](docs/ROADMAP_P6.md) · [氣象預報 MCP 契約](docs/WEATHER_MCP_CONTRACT.md) |
| 熱慣性、假日、季節性補值為什麼不收 | [特徵提案裁定](docs/DECISION_P6_FEATURES.md) |

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
- [x] **P6 誤差監控與氣象預報契約** — 主訊號改誤差、PSI 降級、重訓未改善不晉升；`forecast`／`noisy_forecast` 兩種新天氣模式（[路線圖](docs/ROADMAP_P6.md)）
- [x] **P7 實際值回填與 SQL 週 skill** — 端到端回放 6 棟真實建築；多年同期基準誤報 16% vs 滾動 23%；冰機關機近似規則（[報告](reports/P7_REPORT.md)）

### 還沒做的（依優先序）

1. **用凍結的關機規則開封測試期**（2017-07 起，至今未開封）——這是「排除關機視窗後模型勝過基準線」唯一能變成事前宣告檢定的路徑
2. **誤差觸發重訓的部署模擬**——P6／P7 量的是監控訊號品質，重訓後誤差改善多少仍未量
3. **同期基準與滾動基準並用**——同期抓持續偏移、滾動抓第一週突變（P7 回放兩者各有勝場）
4. **Seq2Seq 給足算力重跑**（序列 168、兩層、60 epochs）——P3 的 NN 呈欠擬合，架構差異尚未真正比過
5. **真實案場**：冰機啟停點位、工作時程表、真實氣象預報封存——三者 BDG2 都沒有，見[領域參考](docs/DOMAIN_REFERENCE.md)

## 幾個結論

- **主結論取決於 2 個視窗**：45 個評估視窗中排除 2 個冬季低載視窗，
  模型對基準線就從 p=0.590 不可宣稱翻成 p<0.0001 可宣稱。因門檻未事前宣告，本專案採**不可宣稱**。
  P7 補上視窗層級的關機規則後重新聚合：主規則下 p=0.33，9 組門檻中 4 組可宣稱、5 組不可，
  **結論取決於門檻，仍不可宣稱**。翻轉結論的那個視窗是間歇低載而非關機，零值段規則在定義上抓不到。
- **遞迴式多步預測比什麼都不做更差**（p=0.003）：誤差沿步長放大 2.25×，
  省下 23 個模型的維運成本買到的是劣於「抄上週同時刻」的結果。
- **PSI 在這個場域沒有鑑別力**：624 個監控週判了 624 次 alert，換三種參考期都一樣。
  該監控的是模型的誤差，不是模型的輸入。誤差監控有鑑別力，但**固定早期基準會被季節性咬**。
- **完美氣象預報的價值量不出來**：perfect vs lagged 全部落在雜訊帶內。
- **誤差監控的基準要用往年同期**：6 棟真實建築端到端回放（預測落庫 → 回填 → SQL 週 skill → 判定），
  同期基準誤報 16%、滾動基準 23%（p=0.001），偵測率 40% vs 23%，逐棟 5 勝 1 平。
- **外部常見建議的特徵在本資料上沒有增益**：熱慣性（3／6 小時累積氣溫）p=0.26、國定假日 p=0.90。
  7 個 site 中 5 個負荷比氣溫早達峰，負荷由開機排程主導，不是外牆吸熱。

## 快速開始

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
src/fetch_data.sh                      # 取資料（走 Git LFS API，附 SHA256 對帳）
.venv/bin/python src/run_quality.py    # P1：品質閘跑全部案場
.venv/bin/python src/run_p3.py         # P3：策略 × 架構對照
.venv/bin/python src/run_p4.py         # P4：部署模擬與漂移監控
.venv/bin/python src/analyze_p4_error.py    # P6：誤差監控 vs PSI 離線重算（秒級）
.venv/bin/python src/ablate_features.py     # P6：熱慣性／假日特徵消融（約 12 分鐘）
.venv/bin/python src/analyze_p3_shutdown.py # P7：關機規則重新聚合 P3（不重訓）

docker compose up -d                        # 以下需要 PostgreSQL（OrbStack／Docker）
.venv/bin/uvicorn --app-dir src service:app # P5：模型服務
.venv/bin/python src/replay_p7.py           # P7：端到端回放（約 25 分鐘，會重建資料表）
.venv/bin/python -m pytest tests/ -q        # 78 則（25 則需 PostgreSQL，未啟動則 skip）
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
6. **判定規則先寫死再看結果**：宣稱門檻、關機規則、監控門檻都在看到資料前寫進程式或設定檔；看過結果才寫的規則，只能當敏感度分析。
7. **一個指標只有一個定義處**：週 skill 只定義在 SQL view，Python 只讀不重算。

## 範圍與誠實邊界

- 全部是**公開資料上的方法論驗證**，不是任何真實場域的導入成果。
- 本專案止於**負荷預測與其可信度驗收**，不含控制決策，所以沒有任何節能率數字。
- 服務層是可運行原型：單機、無認證授權、無告警通道，不是生產環境。
