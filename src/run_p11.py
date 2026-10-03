"""P11｜控制層回放：真實負載與氣象 × 假設的機房模型 × 四種策略 × 兩種主機參數。

設計與事前預期見 docs/P11_PLAN.md。**kWh 與百分比都是假設機房上的模擬值，不是節能量。**

前置：python src/bosch_plant.py（P8 格點資料）；濕度點位直接讀 data/interim/bosch。
    python src/run_p11.py
輸出：reports/p11_results.csv、reports/p11_summary.txt
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_p8  # noqa: E402  （先載入 lightgbm）
import bosch_plant as bp  # noqa: E402
import plant_sim as ps  # noqa: E402
import mpc  # noqa: E402

START = pd.Timestamp("2024-01-15", tz="UTC")            # 濕度點位自此有值
STEPS_PER_DAY = 96
STRATEGIES = ["fixed", "mpc_persist", "mpc_yesterday", "mpc_perfect"]


def load_inputs() -> pd.DataFrame:
    y, t, _ = run_p8.load_inputs()
    rh = bp.hold_grid(bp.load_point(bp.WEATHER["rh_pct"], str(ROOT / bp.ROOT)), "15min", "2h").reindex(y.index)
    d = pd.DataFrame({"q": y, "t": t, "rh": rh})
    d["twb"] = ps.wet_bulb_stull(d.t, d.rh)
    return d[d.index >= START]


def forecast(q: np.ndarray, i: int, kind: str) -> np.ndarray:
    """第 1..H−1 步的負荷預測。超出資料尾端的步補 NaN（control_step 會以當下值補）。"""
    idx = np.arange(i + 1, i + mpc.HORIZON)
    if kind == "mpc_persist":
        return np.full(len(idx), q[i])
    src = idx if kind == "mpc_perfect" else idx - STEPS_PER_DAY
    out = np.full(len(idx), np.nan)
    ok = (src >= 0) & (src < len(q))
    out[ok] = q[src[ok]]
    return out


def replay(d: pd.DataFrame, cfg: ps.PlantConfig, strategy: str) -> dict:
    q, twb = d.q.to_numpy(), d.twb.to_numpy()
    n = len(q)
    prev = (mpc.RULE_CHWS, mpc.RULE_CW)
    chws, cw = np.empty(n), np.empty(n)
    broken = np.zeros(n, bool)
    filt = np.zeros(n, bool)
    fallback = np.zeros(n, bool)
    for i in range(n):
        if strategy == "fixed":
            dec = mpc.control_step(prev, q[i], twb[i], [], [], cfg, optimize=False)
        else:
            idx = np.arange(i + 1, i + mpc.HORIZON)
            twb_fc = np.where(idx < n, twb[np.minimum(idx, n - 1)], np.nan)      # 氣象預報視為完美
            dec = mpc.control_step(prev, q[i], twb[i], forecast(q, i, strategy), twb_fc, cfg)
        chws[i], cw[i] = dec.chws_c, dec.cw_c
        broken[i] = dec.slew_broken
        filt[i] = any(r in ("flow_limit", "over_capacity", "tower_limit", "tower_over_capacity", "out_of_range",
                            "invalid_command") for r in dec.reasons)
        fallback[i] = dec.mode == "fallback"
        prev = (dec.chws_c, dec.cw_c)

    valid = np.isfinite(q) & np.isfinite(twb)
    p = ps.power(np.where(valid, q, 0.0), np.where(valid, twb, 0.0), chws, cw, cfg)
    kwh = {k: float(np.sum(p[k][valid]) / 4) for k in ("chiller", "chw_pump", "fan", "cw_pump", "total")}
    hard = sum(not mpc.within_hard_limits(a, b, cfg) for a, b in zip(chws, cw))
    running = valid & (q > 0)
    return {"strategy": strategy, "variable_speed": cfg.variable_speed, "steps": n, "scored_steps": int(valid.sum()),
            **{f"{k}_mwh": v / 1000 for k, v in kwh.items()},
            "hard_limit_violations": int(hard),
            "infeasible_steps": int((~p["feasible"] & valid).sum()),
            "safety_interventions": int((filt & valid).sum()),
            "slew_broken_steps": int((broken & valid).sum()),
            "fallback_steps": int(fallback.sum()),
            "mean_chws_c": float(chws[running].mean()), "mean_cw_c": float(cw[running].mean())}


def main():
    d = load_inputs()
    lines = [f"P11 摘要：{d.index[0].date()}～{d.index[-1].date()}，{len(d):,} 步（15 分鐘）；"
             f"負載與氣象為 Bosch 實測，機房耗電模型為假設參數",
             "**以下 MWh 與百分比都是假設機房上的模擬值，不是節能量。**", ""]
    lines.append(f"  有效步（負載與濕球溫皆有值）{int((d.q.notna() & d.twb.notna()).sum()):,}；"
                 f"濕球溫 {d.twb.min():.1f}～{d.twb.max():.1f} °C；負載中位 {d.q.median():.0f} kW、最大 {d.q.max():.0f} kW")
    rows = []
    for vs in (True, False):
        cfg = ps.PlantConfig(variable_speed=vs)
        lines += ["", f"## {'變速' if vs else '定速'}主機參數"]
        res = {}
        for s in STRATEGIES:
            t0 = time.time()
            r = replay(d, cfg, s)
            r["seconds"] = round(time.time() - t0, 1)
            res[s] = r
            rows.append(r)
        base, best = res["fixed"]["total_mwh"], res["mpc_perfect"]["total_mwh"]
        for s in STRATEGIES:
            r = res[s]
            lines.append(
                f"  {s:14s} 總計 {r['total_mwh']:8.1f} MWh（主機 {r['chiller_mwh']:.1f}／冰水泵 {r['chw_pump_mwh']:.1f}／"
                f"塔扇 {r['fan_mwh']:.1f}／冷卻水泵 {r['cw_pump_mwh']:.1f}）｜對 fixed {r['total_mwh'] / base - 1:+.1%}｜"
                f"對 perfect {r['total_mwh'] / best - 1:+.2%}")
            lines.append(
                f"  {'':14s} 超出硬限制 {r['hard_limit_violations']}｜不可行步 {r['infeasible_steps']}｜"
                f"安全層介入 {r['safety_interventions']}｜打破變率 {r['slew_broken_steps']}｜降級 {r['fallback_steps']}｜"
                f"平均供水溫 {r['mean_chws_c']:.2f} °C、冷卻水溫 {r['mean_cw_c']:.2f} °C｜{r['seconds']} 秒")
    pd.DataFrame(rows).to_csv(ROOT / "reports/p11_results.csv", index=False)
    txt = "\n".join(lines)
    (ROOT / "reports/p11_summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
