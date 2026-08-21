# HANDOFF — hvac-load-forecast（給接手的 session）

> 產生時間：2026-08-21 · 前一 session：Claude Code Fable 5
> **目標**：把這個 repo 做完，讓 John 能以它投遞台灣愛淨節能 **91c8c 資深機器學習工程師**（80–100K）。
> 求職面的脈絡在 `~/resume & CV/job-applications/ecofirst-ml-engineer/`
> （`JD_Match_*.md` 有 JD 逐條匹配與公司情報；`Portfolio_Plan_HVAC_Forecast.md` 是本專案的規格）。

## 為什麼做這個專案（不要弄丟這條線）

91c8c 把四件事列為**必備條件**，而 John 目前都沒有：
`multi-step 時序預測落地`、`完整 production ML`、`模型標準化設計（多案場）`、`MLOps（pipeline／監控／retrain 自動化）`。
空手投遞會被必備條件直接刷掉，所以**先做完專案再投**。

因此本專案的主軸是**公版化**，不是「做一個準的模型」：

> 一套 pipeline、一份設定檔，N 棟建築各自產出模型與驗收報告；新增一棟不改程式碼。

風格延續 John 既有八個 repo：**不打「模型多準」，打「這個準確率能不能信」**。
負面結果照樣報（前例：surface-defect 的超參搜尋輸給預設配方，那個 repo 照樣寫進去）。

## 現況

| Phase | 狀態 |
|---|---|
| P1 資料契約與品質閘 | ✅ 完成（`reports/P1_REPORT.md`，465 棟實跑） |
| P2 防洩漏特徵與切分 | ✅ 完成（`reports/P2_REPORT.md`） |
| P3 策略 × 架構對照 | ✅ 第一輪完成（`reports/P3_REPORT.md`）；🟡 第二輪 `lagged_only` 與 P4 背景跑中 |
| P4 漂移監控與 retrain 自動化 | 🟡 程式完成（`src/drift.py`／`src/run_p4.py`），背景跑中 |
| P5 FastAPI＋SQLAlchemy 服務層 | ⬜ 未開始 |

測試 29 則全綠：`.venv/bin/python -m pytest tests/ -q`

### 背景工作的當下狀態

`nohup ./run_rest.sh > reports/run_rest.log 2>&1 &` 依序跑三件事：
1. P3 第二輪 `weather_mode=lagged_only`（單棟約 340 秒 × 12，約 68 分鐘）
2. 把設定還原成 `perfect_forecast`
3. P4 漂移監控（6 棟 × 注入/未注入兩組）

**進度**：`tail reports/run_rest.log`。逐棟落盤，中斷不會全損。
**中斷後如何續跑**：直接重跑 `./run_rest.sh` 即可（會覆蓋重算，沒有增量機制）。

跑完後要做的：
- `.venv/bin/python src/analyze_p3.py lagged_only` → 與 perfect_forecast 對照，
  兩者差距＝**氣象預報對這個模型值多少**。這是 P2 報告寫下的承諾，只報一輪是片面的。
- P4 結果在 `reports/p4_drift.csv`，需要寫 `reports/P4_REPORT.md`（分析腳本尚未寫）

### P3 第一輪的關鍵結論（寫報告時別弄丟）

**主結論取決於 45 個視窗中的 2 個**：排除兩個「冰機實質關機」的視窗
（零值 85–93%、平均負荷 0.2–0.4、skill 7.07 與 4.47），
lgbm_direct 對基準線就從 p=0.590「不可宣稱」翻成 p<0.0001「可宣稱」。
因為視窗層級的關機門檻**沒有事前宣告**，報告立場採**不可宣稱**，
排除後的版本只列為敏感度分析。**這個立場不要為了讓數字好看而改掉。**

其餘：遞迴式誤差累積可宣稱且比免費基準線更差；三個架構彼此分不出高下；
兩個 NN 呈欠擬合樣態（逐步長誤差幾乎是平的），已標註為算力取捨而非架構結論。

## 接手時務必知道的五個坑（都踩過了，別再踩一次）

1. **OpenMP 雙向 segfault**：lightgbm 與 torch 各帶一份 OpenMP runtime，任一方先載入、
   另一方後使用都會 SIGSEGV，**沒有 Python 例外**，看起來像程式被殺掉。
   `KMP_DUPLICATE_LIB_OK` 無效。解法是把 OMP 執行緒數壓到 1 且在載入前設定
   （`src/models.py` 最頂端與 `tests/conftest.py`）。**別動那幾行的位置。**
2. **標的含 NaN 會靜默吃掉整個架構**：loss 變 nan → `nan < best` 恆為 False →
   val loss 停在 inf → 上層跳過該模型且不報錯，症狀是「結果表裡就是沒有那幾列」。
   已修並有兩則守衛測試。
3. **Git LFS 指標檔**：BDG2 用 LFS，`curl raw.githubusercontent` 只會拿到 133 bytes 的
   指標檔（副檔名 `.csv`、HTTP 200、檔案存在）。用 `src/fetch_data.sh`，它走 LFS batch API
   並以 SHA256 對帳。
4. **零膨脹**：冰機關機時讀數為 0，有案場零值佔比 77–80%。任何以整體分布估的尺度
   （IQR、median|diff|、平坦段）都會退化成「把所有非零值判成異常」。P1 為此踩了三次，
   三條規則全部改在**非零子集**上估。寫新規則時先問一句：這個統計量遇到大量 0 會怎樣？
5. **別多加 `X.notna().all(axis=1)`**：LightGBM 原生支援缺值，而 `cloudCoverage` 在部分
   site 缺值 52%，多加那個條件會讓可用列從 100% 掉到 29%。

## P4 要做什麼（規格）

對映 91c8c 必備條件「建立並維護公版 ML pipeline：部署、監控、retrain 自動化」。

- 2016 訓練、2017 部署模擬，**逐週 PSI** 監控特徵分布漂移
- PSI 超門檻**自動觸發重訓**，重訓前後用**同一套 backtest 協定**驗收——重訓不是信仰，要驗收
- **概念漂移注入演練**：人工改變某棟建築的負荷型態（模擬使用行為改變），
  量測多久捕捉到、重訓救不救得回，並找出**哪類漂移重訓救不回**（那種要換特徵）
- MLflow 記錄實驗與模型版本
- 公版化第二層：一份設定檔驅動 N 棟建築的訓練→驗收→註冊，每棟一份驗收報告

## P5 要做什麼（規格）

對映 91c8c 必備條件「Python(FastAPI / PostgreSQL / SQLAlchemy) 開發模型服務與排程任務」
與「跟軟體工程師共同定義資料 schema 與 API contract」。

- **FastAPI＋Pydantic**：兩套「廠牌方言」欄位（模擬大金 vs 開利命名）→ canonical schema 的
  ingestion 對映；schema 違規在閘口擋下並回結構化錯誤（不是讓髒資料流進模型）
- **PostgreSQL＋SQLAlchemy ORM**（**必須真的用 ORM**，這是履歷的關鍵字缺口）：
  預測結果、品質閘攔截記錄、模型版本與驗收指標落庫
- 排程任務：每日預測、每週重訓檢查
- 儀表板：預測 vs 實際、PSI 狀態、品質閘攔截記錄**同頁**

## 完工後要回填的四個地方（少一處這件事對機器就不存在）

1. `~/resume & CV/job-applications/resume-master/metrics.md` — 新增本專案的數字列與佐證出處
2. `CV_Master.md` **技能區** — 加 ATS 關鍵字（HTML 註解不會進 PDF，ATS 掃不到）
3. `CV_Master.md` **誠實邊界註解** — 現有那句「影像 CNN，無 RNN／序列模型、無自行訓練
   Transformer」**做完 P3 就過期了**，必須改
4. `Portfolio_Plan_HVAC_Forecast.md` — 勾銷

然後才重做四件組投 91c8c。**送出前一定要先給 John 過目**（他明確要求過）。

## 宣稱紀律（寫進 metrics.md 前先看這段）

- 可寫：「公開建築能耗資料上的多步時序預測**方法論驗證**」＋實測數字；
  「一套 pipeline 跨 N 棟建築、新增建築不改程式碼」
- **不得寫**：「HVAC 節能系統開發經驗」「生產環境時序預測」「為案場建模」「節能率 X%」
- **不得寫**「模型調得更準」除非實測 p<0.05 且差距大於雜訊帶
- 公開前走 `public-repo-prep`：掃 `求職|作品集|履歷|面試|JD`——**本檔就是必須被掃掉的那種內容**
