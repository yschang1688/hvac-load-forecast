# 氣象預報 MCP：API contract（P6 草案）

> 狀態：**契約已定、介面已接（`weather_mode="forecast"`）、資料源未接**。
> 本檔是給 ML 端與供給端共同遵守的契約，不是實作說明。

## 為什麼要用 MCP 而不是直接呼叫氣象 API

模型端要的不是「今天的預報」，是**「在時點 t 能拿到的、對 t+1…t+24 的預報」**。
這兩件事在離線評估時差很多：直接呼叫氣象 API 只會給最新一版，回放歷史時就會拿到
事後修正過的值——那是 P2 第二型洩漏（用實測未來天氣）的變形，一樣不會被 shift 檢查抓到。

把預報供給封成 MCP server 的意義是把這條因果規則寫進契約，讓任何呼叫端
（訓練管線、線上推論、回放評估、agent）都拿到同一種語意的資料：
**每筆預報帶 `issued_at` 與 `target_ts`，呼叫端只能用 `issued_at <= t` 的最新一版。**
這和 P5 的 `predictions` 表把 `issued_at` 與 `target_ts` 分開記是同一條紀律。

## Tools

### `get_forecast`

在指定發布時點之前，取某案場對一段目標時段的最新預報。

```json
{
  "name": "get_forecast",
  "input": {
    "site_key": "DAIKIN_SITE",
    "as_of": "2017-03-01T06:00:00Z",
    "target_start": "2017-03-01T07:00:00Z",
    "target_end": "2017-03-02T06:00:00Z",
    "variables": ["airTemperature", "dewTemperature"],
    "provider": "cwa"
  },
  "output": {
    "site_key": "DAIKIN_SITE",
    "provider": "cwa",
    "model_run": "2017-03-01T00:00:00Z",
    "issued_at": "2017-03-01T03:20:00Z",
    "rows": [
      {"target_ts": "2017-03-01T07:00:00Z", "lead_hours": 1, "airTemperature": 18.4, "dewTemperature": 14.1},
      {"target_ts": "2017-03-01T08:00:00Z", "lead_hours": 2, "airTemperature": 19.0, "dewTemperature": 14.3}
    ],
    "units": {"airTemperature": "degC", "dewTemperature": "degC"},
    "missing": []
  }
}
```

契約條款：
1. **`issued_at <= as_of`**，且回傳的是滿足此條件的**最新一次**發布；server 不得回傳更晚的修正版。
2. `target_ts` 為整點、UTC；`lead_hours = target_ts − issued_at` 的整數小時數（供 ML 端建 σ(h) 用）。
3. 單位固定 SI（°C、m/s、%），換算在 server 端做——與 P5 廠牌方言歸一同一原則：方言不進模型。
4. 缺值就缺（`missing` 列出 target_ts），**不插補**；插補是 ML 端依 `weather_interp_limit` 的決定。
5. 無預報可用時回空 `rows` 而非錯誤——ML 端據此退到 `lagged_only`（下界），並記一筆事件。

### `get_forecast_archive`

批次取一段期間內每個發布時點的預報，供回放評估與 `noisy_forecast` 雜訊尺度校準。
輸入 `site_key, issued_from, issued_to, variables, provider`；輸出為長表
`(issued_at, target_ts, lead_hours, 變數…)`，即 `features.forecast_block_from_frame` 的輸入格式。

### `get_forecast_skill`

回傳該 provider 在該案場的**歷史驗證誤差** `{variable: {lead_hours: mae}}`。
用途兩個：(a) `noisy_forecast` 的 σ(h) 必須從這裡來，不得手填；
(b) 線上監控時把「預報本身錯了」和「模型錯了」分開歸因——沒有這一項，P6 的誤差監控
會把一次爛預報算成模型漂移而觸發不必要的重訓。

## Resources

- `weather://{site_key}/latest` — 最新一版預報（線上推論用）
- `weather://{site_key}/providers` — 該案場可用的 provider 與各自的驗證誤差摘要

## ML 端的對應

| 契約條款 | 程式落點 |
|---|---|
| issued_at ≤ t 取最新 | `features.forecast_block_from_frame`（守門測試：事後修正版不得被用到） |
| 無預報退 lagged_only | `weather_mode` 切換；事件寫 `quality_events`（待接） |
| σ(h) 來自 `get_forecast_skill` | `config.features.forecast_noise_sigma`（目前為佔位值，**未經驗證**） |
| 預報誤差 vs 模型誤差歸因 | P6 誤差監控的下一步；需 `predictions` 表加 `forecast_issued_at` 欄 |

## 資料源候選與各自的坑

| Provider | 涵蓋 | 坑 |
|---|---|---|
| 中央氣象署開放資料（台灣案場） | 鄉鎮 3 小時／1 週預報 | 官方**不提供歷史預報封存**，要自己每日抓存才有回放資料——上線第一天起就得開始存 |
| Open-Meteo Previous Runs / Historical Forecast | 全球、逐時 | 封存起點約 2021–2022，**BDGP2 的 2016–2017 沒有**；本專案的公開資料集無法用真實預報回放 |
| ECMWF / NOAA 封存 | 全球、有歷史 | 取用門檻高（格式、配額、授權），先確認回報值得 |

**這表的結論**：在本專案的公開資料集上，「真實預報」這條線**做不出來**，只能做 `noisy_forecast` 的中間態；
真實預報要等真實案場上線後從第一天開始封存。這是誠實邊界，不是待辦。
