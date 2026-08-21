"""pytest 必須在載入任何 OpenMP 使用者（lightgbm / torch / numpy BLAS）之前壓住執行緒數。

見 src/models.py 頂端註解：兩者各帶一份 OpenMP runtime，共存時任一方向都會 segfault，
而 segfault 沒有 Python 例外——測試會顯示成「崩潰」而不是「失敗」，很容易誤判成環境壞掉。
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
