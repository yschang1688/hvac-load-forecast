"""P11｜控制層骨架：滾動視窗最佳化、獨立安全層、降級。

三個部分刻意分開：
- `solve`          最佳化器。往後看 H 步，在設定點格點上用動態規劃找總耗電最低的路徑，只回第一步。
- `safety_filter`  安全層。**不依賴最佳化器**：任何指令（含 NaN、超界）進來，輸出一定在硬限制內。
- `control_step`   把兩者接起來，並處理輸入缺值時的降級。

設計見 docs/P11_PLAN.md。機房耗電模型是假設的（plant_sim.py），算出的 kW 是模擬值。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.ndimage import minimum_filter

import plant_sim as ps

CHWS_STEP, CW_STEP = 0.25, 0.5          # 設定點格點（K）
SLEW_CHWS, SLEW_CW = 2, 2               # 每步最多移動的格數：0.5 K 與 1.0 K（假設值）
HORIZON = 8                             # 15 分鐘 × 8 ＝ 2 小時
BIG = 1e6                               # 不可行狀態的懲罰（kW）
RULE_CHWS, RULE_CW = 7.0, 30.0          # 規則設定點：降級時用


def grids(cfg: ps.PlantConfig) -> tuple[np.ndarray, np.ndarray]:
    chws = np.arange(cfg.chws_min_c, cfg.chws_max_c + 1e-9, CHWS_STEP)
    cw = np.arange(cfg.cw_min_c, cfg.cw_max_c + 1e-9, CW_STEP)
    return chws, cw


def step_cost(q_kw, twb_c, cfg: ps.PlantConfig) -> np.ndarray:
    """每個（供水溫, 冷卻水溫）格點的全廠耗電；不可行的格點加懲罰。q、twb 可為長度 H 的陣列 → (H, n_chws, n_cw)。"""
    chws, cw = grids(cfg)
    q = np.atleast_1d(np.asarray(q_kw, dtype=float))[:, None, None]
    twb = np.atleast_1d(np.asarray(twb_c, dtype=float))[:, None, None]
    p = ps.power(q, twb, chws[None, :, None], cw[None, None, :], cfg)
    return np.where(p["feasible"], p["total"], p["total"] + BIG)


def solve(prev: tuple[float, float], q_now: float, twb_now: float,
          q_fc, twb_fc, cfg: ps.PlantConfig) -> tuple[float, float]:
    """回傳第一步的設定點。`q_fc`、`twb_fc` 是第 1..H−1 步的預測（可為空）。"""
    chws, cw = grids(cfg)
    q = np.concatenate([[q_now], np.asarray(q_fc, dtype=float)])
    twb = np.concatenate([[twb_now], np.asarray(twb_fc, dtype=float)])
    cost = step_cost(q, twb, cfg)
    size = (2 * SLEW_CHWS + 1, 2 * SLEW_CW + 1)
    v_next = np.zeros(cost.shape[1:])
    for h in range(len(q) - 1, 0, -1):
        v_next = cost[h] + minimum_filter(v_next, size=size, mode="constant", cval=np.inf)
    j0 = cost[0] + minimum_filter(v_next, size=size, mode="constant", cval=np.inf)
    # 第 0 步只能從上一步的設定點移動變率限制以內
    i0 = int(np.clip(round((prev[0] - chws[0]) / CHWS_STEP), 0, len(chws) - 1))
    k0 = int(np.clip(round((prev[1] - cw[0]) / CW_STEP), 0, len(cw) - 1))
    mask = np.full(j0.shape, np.inf)
    mask[max(i0 - SLEW_CHWS, 0):i0 + SLEW_CHWS + 1, max(k0 - SLEW_CW, 0):k0 + SLEW_CW + 1] = 0.0
    i, k = np.unravel_index(np.argmin(j0 + mask), j0.shape)
    return float(chws[i]), float(cw[k])


def safety_filter(chws_cmd: float, cw_cmd: float, q_now: float, twb_now: float,
                  cfg: ps.PlantConfig) -> tuple[float, float, list[str]]:
    """獨立於最佳化器的硬限制。回傳（供水溫, 冷卻水溫, 介入原因）。

    供水溫只會被往下修（往安全側），冷卻水溫只會被往上修；結果對齊格點。
    q_now 或 twb_now 缺值時，跳過需要它的那一條檢查，只做範圍截斷。
    """
    reasons: list[str] = []
    if not (_finite(chws_cmd) and _finite(cw_cmd)):
        chws_cmd, cw_cmd = RULE_CHWS, RULE_CW
        reasons.append("invalid_command")

    chws = min(max(chws_cmd, cfg.chws_min_c), cfg.chws_max_c)
    cw = min(max(cw_cmd, cfg.cw_min_c), cfg.cw_max_c)
    if chws != chws_cmd or cw != cw_cmd:
        reasons.append("out_of_range")

    if _finite(q_now) and q_now > 0:
        cap = ps.max_feasible_chws(q_now, cfg)
        if chws > cap + 1e-9:
            chws = max(cap, cfg.chws_min_c)
            reasons.append("flow_limit" if cap >= cfg.chws_min_c else "over_capacity")
    chws = max(cfg.chws_min_c, math.floor((chws - cfg.chws_min_c) / CHWS_STEP + 1e-9) * CHWS_STEP + cfg.chws_min_c)

    cw = _ceil_cw(cw, cfg)
    if _finite(q_now) and q_now > 0 and _finite(twb_now):
        # 冷卻水溫往上修會讓主機效率變差、排熱變多，所需趨近溫度跟著變大：修到可行為止，不是修一次
        for _ in range(int((cfg.cw_max_c - cfg.cw_min_c) / CW_STEP) + 1):
            need = twb_now + float(ps.approach_required(q_now, chws, cw, cfg))
            if cw >= need - 1e-9:
                break
            if "tower_limit" not in reasons:
                reasons.append("tower_limit")
            if cw >= cfg.cw_max_c:
                reasons[reasons.index("tower_limit")] = "tower_over_capacity"
                break
            cw = _ceil_cw(max(need, cw + CW_STEP), cfg)
    return chws, cw, reasons


def _ceil_cw(cw: float, cfg: ps.PlantConfig) -> float:
    return min(cfg.cw_max_c, math.ceil((cw - cfg.cw_min_c) / CW_STEP - 1e-9) * CW_STEP + cfg.cw_min_c)


@dataclass
class Decision:
    chws_c: float
    cw_c: float
    mode: str                       # "mpc" | "rule" | "fallback"
    reasons: list[str] = field(default_factory=list)
    slew_broken: bool = False       # 安全層為了可行性打破變率限制


def control_step(prev: tuple[float, float], q_now: float, twb_now: float,
                 q_fc, twb_fc, cfg: ps.PlantConfig, optimize: bool = True) -> Decision:
    """一個控制週期。`optimize=False` 是固定規則設定點（對照組），一樣經過安全層。"""
    reasons: list[str] = []
    if not (_finite(q_now) and _finite(twb_now)):
        chws, cw, r = safety_filter(RULE_CHWS, RULE_CW, q_now, twb_now, cfg)
        return Decision(chws, cw, "fallback", ["missing_measurement"] + r)

    if optimize:
        q_fc = np.asarray(q_fc, dtype=float)
        twb_fc = np.asarray(twb_fc, dtype=float)
        if np.isnan(q_fc).any() or np.isnan(twb_fc).any():
            q_fc = np.where(np.isnan(q_fc), q_now, q_fc)
            twb_fc = np.where(np.isnan(twb_fc), twb_now, twb_fc)
            reasons.append("forecast_filled")
        cmd = solve(prev, q_now, twb_now, q_fc, twb_fc, cfg)
    else:
        cmd = (RULE_CHWS, RULE_CW)

    chws, cw, r = safety_filter(cmd[0], cmd[1], q_now, twb_now, cfg)
    broken = optimize and (abs(chws - prev[0]) > SLEW_CHWS * CHWS_STEP + 1e-9
                           or abs(cw - prev[1]) > SLEW_CW * CW_STEP + 1e-9)
    return Decision(chws, cw, "mpc" if optimize else "rule", reasons + r, broken)


def within_hard_limits(chws: float, cw: float, cfg: ps.PlantConfig) -> bool:
    return (cfg.chws_min_c - 1e-9 <= chws <= cfg.chws_max_c + 1e-9
            and cfg.cw_min_c - 1e-9 <= cw <= cfg.cw_max_c + 1e-9)


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
