"""P11｜控制層的 API 契約（獨立於 P5 的 service.py，不需要資料庫）。

POST /optimize  當下量測＋未來 H−1 步預測 → 經安全層的設定點
回應永遠帶 `mode` 與 `reasons`：呼叫端看得到這一步是最佳化、降級，還是被安全層改過。
機房耗電模型是假設的（plant_sim.py），`predicted_total_kw` 是模擬值。
"""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

import mpc
import plant_sim as ps

app = FastAPI(title="HVAC setpoint optimizer (P11 skeleton)")


class OptimizeRequest(BaseModel):
    site_key: str
    variable_speed_chiller: bool = True
    prev_chws_c: float = Field(mpc.RULE_CHWS, description="上一個週期套用的冰水供水溫設定點")
    prev_cw_c: float = Field(mpc.RULE_CW, description="上一個週期套用的冷卻水進水溫設定點")
    load_kw: float | None = Field(None, description="當下量到的冷量；缺值就降級")
    wet_bulb_c: float | None = Field(None, description="當下濕球溫；缺值就降級")
    forecast_load_kw: list[float | None] = Field(default_factory=list, description="第 1..H−1 步")
    forecast_wet_bulb_c: list[float | None] = Field(default_factory=list, description="第 1..H−1 步")

    @model_validator(mode="after")
    def _same_length(self):
        n, m = len(self.forecast_load_kw), len(self.forecast_wet_bulb_c)
        if n != m:
            raise ValueError(f"forecast_load_kw（{n}）與 forecast_wet_bulb_c（{m}）長度必須相同")
        if n > mpc.HORIZON - 1:
            raise ValueError(f"預測最多 {mpc.HORIZON - 1} 步，收到 {n}")
        return self


class OptimizeResponse(BaseModel):
    site_key: str
    chws_setpoint_c: float
    cw_setpoint_c: float
    mode: str
    reasons: list[str]
    slew_broken: bool
    predicted_total_kw: float | None


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    cfg = ps.PlantConfig(variable_speed=req.variable_speed_chiller)
    q, twb = _nan(req.load_kw), _nan(req.wet_bulb_c)
    d = mpc.control_step((req.prev_chws_c, req.prev_cw_c), q, twb,
                         [_nan(v) for v in req.forecast_load_kw],
                         [_nan(v) for v in req.forecast_wet_bulb_c], cfg)
    total = None
    if d.mode == "mpc":
        total = round(float(ps.power(q, twb, d.chws_c, d.cw_c, cfg)["total"]), 2)
    return OptimizeResponse(site_key=req.site_key, chws_setpoint_c=d.chws_c, cw_setpoint_c=d.cw_c,
                            mode=d.mode, reasons=d.reasons, slew_broken=d.slew_broken,
                            predicted_total_kw=total)
