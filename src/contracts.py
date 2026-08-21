"""P5｜資料介面與 API contract：多廠牌方言 → canonical schema。

**這一層存在的理由**：從「一案場一客製」走向平台時，最先崩的不是模型，是欄位。
大金叫 `chwSupplyTemp`、開利叫 `CHW_SUPPLY_TEMPERATURE`、特靈用另一套單位，
如果讓每個案場的方言直接流進模型，每接一個新案場就要改一次模型端程式碼——
那就不是平台，只是把客製化搬到另一個檔案裡。

**設計原則**
1. **對映寫在設定，不寫在程式**：新增廠牌＝加一個 `VendorDialect`，不改 ingestion 邏輯。
2. **在閘口擋下，不要往下傳**：Pydantic 驗證失敗即回結構化錯誤（哪個欄位、什麼值、
   違反哪條規則），而不是塞 None 讓模型吃到怪東西。
3. **單位歸一是契約的一部分**：°F→°C、gal/min→m³/h 在此處完成，
   否則同一個欄位名在不同案場代表不同東西，是最難查的一種錯。
"""
from __future__ import annotations
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CanonicalReading(BaseModel):
    """所有案場、所有廠牌，進了系統就長這個樣子。"""
    site_key: str
    ts: datetime
    chw_load_kw: float = Field(description="冰水側瞬時熱負荷，kW")
    oat_c: float | None = Field(default=None, description="室外乾球溫度，°C")
    dew_c: float | None = Field(default=None, description="露點溫度，°C")

    @field_validator("chw_load_kw")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        # 與 P1 的 R001 同一條物理規則，在閘口先擋一次
        if v < 0:
            raise ValueError("chw_load_kw 不可為負（冰水側熱負荷無負值的物理意義）")
        return v

    @field_validator("oat_c", "dew_c")
    @classmethod
    def _plausible_temp(cls, v: float | None) -> float | None:
        if v is not None and not (-60.0 <= v <= 70.0):
            raise ValueError(f"溫度 {v}°C 超出地表可能範圍，多半是單位或位元組序錯誤")
        return v

    @model_validator(mode="after")
    def _dew_not_above_oat(self):
        # 物理不等式：露點不可能高於乾球溫度。這種錯不會被單欄位驗證抓到。
        if self.oat_c is not None and self.dew_c is not None and self.dew_c > self.oat_c + 0.5:
            raise ValueError(f"露點 {self.dew_c}°C 高於乾球 {self.oat_c}°C，感測器或對映有誤")
        return self


def f_to_c(v: float) -> float:
    return (v - 32.0) * 5.0 / 9.0


def rt_to_kw(v: float) -> float:
    """冷凍噸 → kW（1 RT = 3.51685 kW）。"""
    return v * 3.51685


class VendorDialect(BaseModel):
    """一個廠牌的欄位對映與單位換算。新增廠牌只加一筆，不改 ingestion 程式。"""
    name: str
    ts_field: str
    load_field: str
    load_unit: Literal["kW", "RT"] = "kW"
    oat_field: str | None = None
    oat_unit: Literal["C", "F"] = "C"
    dew_field: str | None = None
    dew_unit: Literal["C", "F"] = "C"


DIALECTS: dict[str, VendorDialect] = {
    # 兩套刻意不同的方言：欄位名、大小寫、單位、時戳鍵全部不一樣
    "vendor_a": VendorDialect(
        name="vendor_a", ts_field="timestamp", load_field="chwLoadKw",
        load_unit="kW", oat_field="oaTempC", oat_unit="C", dew_field="dewPointC", dew_unit="C"),
    "vendor_b": VendorDialect(
        name="vendor_b", ts_field="RECORD_TIME", load_field="CHW_LOAD_RT",
        load_unit="RT", oat_field="OA_TEMP_F", oat_unit="F", dew_field="DEW_POINT_F", dew_unit="F"),
}


class ContractError(Exception):
    """在閘口就擋下，並帶著足以定位問題的結構化資訊。"""

    def __init__(self, field: str, value, reason: str):
        self.field, self.value, self.reason = field, value, reason
        super().__init__(f"[{field}] {value!r}: {reason}")

    def as_dict(self) -> dict:
        return {"field": self.field, "value": str(self.value), "reason": self.reason}


def check_capacity(r: CanonicalReading, capacity_kw: float | None, k: float = 1.5) -> None:
    """對照銘牌額定容量。**沒有銘牌就不猜**——寧可不判，也不要用資料自身估一個假上限
    （P1 踩過：零膨脹下用分位數估上限會讓每個非零讀數都變成違規）。"""
    if capacity_kw is None or capacity_kw <= 0:
        return
    if r.chw_load_kw > k * capacity_kw:
        raise ContractError("chw_load_kw", r.chw_load_kw,
                            f"超過銘牌額定 {capacity_kw} kW 的 {k} 倍，多為計量器溢位或單位錯誤")


def to_canonical(site_key: str, vendor: str, raw: dict,
                 capacity_kw: float | None = None) -> CanonicalReading:
    d = DIALECTS.get(vendor)
    if d is None:
        raise ContractError("vendor", vendor, f"未登錄的廠牌方言（已知：{sorted(DIALECTS)}）")
    for f in (d.ts_field, d.load_field):
        if f not in raw:
            raise ContractError(f, None, "必要欄位缺失")

    load = float(raw[d.load_field])
    if d.load_unit == "RT":
        load = rt_to_kw(load)

    def temp(field, unit):
        if not field or raw.get(field) is None:
            return None
        v = float(raw[field])
        return f_to_c(v) if unit == "F" else v

    try:
        r = CanonicalReading(site_key=site_key, ts=raw[d.ts_field], chw_load_kw=load,
                             oat_c=temp(d.oat_field, d.oat_unit),
                             dew_c=temp(d.dew_field, d.dew_unit))
    except ValueError as e:
        first = str(e).splitlines()
        msg = next((x.strip() for x in first if "Value" in x or "不可" in x or "超出" in x), str(e))
        raise ContractError("payload", raw, msg[:200]) from e
    check_capacity(r, capacity_kw)
    return r
