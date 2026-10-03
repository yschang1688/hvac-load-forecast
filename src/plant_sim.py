"""P11｜假設的冰水機房耗電模型（**不是從資料辨識的**）。

Bosch 資料沒有主機電表、供水設定點全年恆為 7.0 °C，效率曲線無法從資料學。
這裡的參數模型只用來讓控制邏輯有東西可以算；每個參數都標來源或標「假設」，見 docs/P11_PLAN.md。
**由它算出的 kW 與百分比是模擬值，不是節能量。**

所有函式都接受純量或可廣播的 numpy 陣列。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

RHO_CP = 4.186 * 1000 / 3600        # kW/(m³/h·K)，同 bosch_plant.RHO_CP
KW_PER_TON = 3.517
GPM_PER_M3H = 4.4029


@dataclass(frozen=True)
class PlantConfig:
    # 主機：3 台相同，額定 COP 取台灣 MEPS（水冷離心式 ≥1,055 kW）；設計點 7／30 °C 為假設
    n_chillers: int = 3
    q_unit_kw: float = 1100.0
    cop_nom: float = 6.10
    chws_design_c: float = 7.0
    cw_design_c: float = 30.0
    stage_plr: float = 0.9              # 台數規則：每台負載率超過 0.9 就加開（假設）
    variable_speed: bool = True
    # 溫度修正（每 K 的 COP 增益）：量級取自台電 2014 簡報與綠基會手冊的經驗值
    gain_chws_vs: float = 0.03
    gain_cw_vs: float = 0.045
    gain_chws_cs: float = 0.02
    gain_cw_cs: float = 0.02
    # 部分負載修正：1 − b·(PLR − p*)²，形狀為假設
    plr_peak_vs: float = 0.55
    plr_curv_vs: float = 0.4
    plr_peak_cs: float = 0.80
    plr_curv_cs: float = 0.6
    # 冰水泵：變頻；額定 22 W/gpm（ASHRAE 90.1 附錄 G，轉引自 Trane 2021）
    coil_dt_design_k: float = 5.5
    coil_dt_slope: float = 0.5          # 供水溫每升 1 K，盤管溫差縮 0.5 K（假設）
    pump_w_per_gpm: float = 22.0
    flow_min_frac: float = 0.6
    flow_max_frac: float = 1.1
    # 冷卻塔與冷卻水泵（Trane 2021 引 ASHRAE GreenGuide）
    fan_kw_per_ton: float = 0.036
    cw_pump_kw_per_ton: float = 0.019
    approach_design_k: float = 4.0
    fan_min_frac: float = 0.2
    # 硬限制
    chws_min_c: float = 6.0
    chws_max_c: float = 10.0
    cw_min_c: float = 18.0
    cw_max_c: float = 32.0

    @property
    def flow_unit_m3h(self) -> float:
        return self.q_unit_kw / (RHO_CP * self.coil_dt_design_k)

    @property
    def hr_nom_kw(self) -> float:
        """全廠額定排熱量：冷量加壓縮機耗電。"""
        return self.n_chillers * self.q_unit_kw * (1 + 1 / self.cop_nom)


def wet_bulb_stull(t_c, rh_pct):
    """Stull (2011) 濕球溫度經驗式。適用 RH 5–99%、−20–50 °C；範圍外回 NaN。"""
    t = np.asarray(t_c, dtype=float)
    rh = np.asarray(rh_pct, dtype=float)
    tw = (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh)
          - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)
    bad = (rh < 5) | (rh > 99) | (t < -20) | (t > 50)
    return np.where(bad, np.nan, tw)


def n_on(q_kw, cfg: PlantConfig):
    q = np.asarray(q_kw, dtype=float)
    n = np.ceil(q / (cfg.stage_plr * cfg.q_unit_kw))
    return np.where(q > 0, np.clip(n, 1, cfg.n_chillers), 0)


def coil_dt(chws_c, cfg: PlantConfig):
    return cfg.coil_dt_design_k - cfg.coil_dt_slope * (np.asarray(chws_c, dtype=float) - cfg.chws_design_c)


def flow_required(q_kw, chws_c, cfg: PlantConfig):
    return np.asarray(q_kw, dtype=float) / (RHO_CP * coil_dt(chws_c, cfg))


def flow_max(q_kw, cfg: PlantConfig):
    return cfg.flow_max_frac * n_on(q_kw, cfg) * cfg.flow_unit_m3h


def max_feasible_chws(q_kw: float, cfg: PlantConfig) -> float:
    """當下負載不超過流量上限的最高供水溫（連續值，未截到硬限制）。負載為 0 時回上限。"""
    fmax = float(flow_max(q_kw, cfg))
    if fmax <= 0:
        return cfg.chws_max_c
    dt_needed = q_kw / (RHO_CP * fmax)
    return cfg.chws_design_c + (cfg.coil_dt_design_k - dt_needed) / cfg.coil_dt_slope


def cop(q_kw, chws_c, cw_c, cfg: PlantConfig):
    q = np.asarray(q_kw, dtype=float)
    n = n_on(q, cfg)
    plr = np.divide(q, n * cfg.q_unit_kw, out=np.zeros(np.broadcast(q, n).shape), where=n > 0)
    if cfg.variable_speed:
        g1, g2, pk, cv = cfg.gain_chws_vs, cfg.gain_cw_vs, cfg.plr_peak_vs, cfg.plr_curv_vs
    else:
        g1, g2, pk, cv = cfg.gain_chws_cs, cfg.gain_cw_cs, cfg.plr_peak_cs, cfg.plr_curv_cs
    f_lift = 1 + g1 * (np.asarray(chws_c, dtype=float) - cfg.chws_design_c) \
               + g2 * (cfg.cw_design_c - np.asarray(cw_c, dtype=float))
    f_plr = 1 - cv * (plr - pk) ** 2
    return cfg.cop_nom * np.maximum(f_lift, 0.5) * np.maximum(f_plr, 0.3)


def approach_required(q_kw, chws_c, cw_c, cfg: PlantConfig):
    """塔扇全開時能達到的最小趨近溫度：設計趨近 × 排熱比。"""
    q = np.asarray(q_kw, dtype=float)
    hr = q * (1 + 1 / cop(q, chws_c, cw_c, cfg))
    return cfg.approach_design_k * hr / cfg.hr_nom_kw


def power(q_kw, twb_c, chws_c, cw_c, cfg: PlantConfig) -> dict:
    """各設備耗電（kW）與可行性。`flow_over`＝供冷不足；`tower_short`＝塔扇全開也到不了這個冷卻水溫。"""
    q = np.asarray(q_kw, dtype=float)
    chws = np.asarray(chws_c, dtype=float)
    cw = np.asarray(cw_c, dtype=float)
    twb = np.asarray(twb_c, dtype=float)
    n = n_on(q, cfg)
    on = n > 0
    p_ch = np.where(on, q / cop(q, chws, cw, cfg), 0.0)

    f_req = flow_required(q, chws, cfg)
    f_nom = n * cfg.flow_unit_m3h
    f_app = np.maximum(f_req, cfg.flow_min_frac * f_nom)
    ratio = np.divide(f_app, f_nom, out=np.zeros(np.broadcast(f_app, f_nom).shape), where=f_nom > 0)
    p_pump_unit = cfg.pump_w_per_gpm * cfg.flow_unit_m3h * GPM_PER_M3H / 1000.0
    p_pump = n * p_pump_unit * ratio ** 3
    flow_over = on & (f_req > cfg.flow_max_frac * f_nom + 1e-9)

    hr = q + p_ch
    gap = cw - twb
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(gap > 0, cfg.approach_design_k * (hr / cfg.hr_nom_kw) / gap, np.inf)
    tower_short = on & (frac > 1 + 1e-9)
    fan_nom = cfg.fan_kw_per_ton * cfg.n_chillers * cfg.q_unit_kw / KW_PER_TON
    p_fan = np.where(on, fan_nom * np.clip(frac, cfg.fan_min_frac, 1.0) ** 3, 0.0)
    p_cwp = n * cfg.cw_pump_kw_per_ton * cfg.q_unit_kw / KW_PER_TON

    return {"chiller": p_ch, "chw_pump": p_pump, "fan": p_fan, "cw_pump": p_cwp,
            "total": p_ch + p_pump + p_fan + p_cwp,
            "flow_over": flow_over, "tower_short": tower_short, "feasible": ~(flow_over | tower_short)}
