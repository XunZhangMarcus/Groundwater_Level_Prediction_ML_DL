"""
地下水水位序列
读取 → 清洗 → 截取有效区间(≤2019-10-01) → 线性/样条插值 → 双-subplot 折线图 → Excel
"""

# --------------------------------------------------------------------
# 0. 环境与全局配置
# --------------------------------------------------------------------
import warnings, re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ▸ 插值方式：'linear' 或 'spline'
INTERP_METHOD = "linear"

# ▸ 文件路径
DATA_PATH = "./database/ZoupingCounty_gwl_data.xlsx"
OUT_EXCEL = "./database/ZoupingCounty_gwl_filled.xlsx"
PLOT_DIR  = Path("./database/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ▸ 中文字体（如无 SimHei 请改成已安装字体或注掉）
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# ▸ 静默常见警告
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"Glyph .* missing from current font")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"Could not infer format")

# --------------------------------------------------------------------
# 1. 读取原始 Excel
# --------------------------------------------------------------------
df_raw = pd.read_excel(DATA_PATH, header=None)

# --------------------------------------------------------------------
# 2. 生成列名：<中文名称>-井<n>  (n 从 1 开始按列顺序)
# --------------------------------------------------------------------
names_row = df_raw.iloc[1]                  # “名称”行
rename_map, used = {}, set()

for idx, col in enumerate(df_raw.columns[1:], start=1):   # 跳过日期列
    name_cn = str(names_row[col]).strip() if pd.notna(names_row[col]) else ""
    base    = f"{name_cn}-井{idx}" if name_cn else f"井{idx}"
    new     = base
    k = 2
    while new in used:                                    # 保证唯一
        new = f"{base}_{k}"
        k += 1
    rename_map[col] = new
    used.add(new)

# --------------------------------------------------------------------
# 3. 清洗 & 日期解析
# --------------------------------------------------------------------
df = df_raw.iloc[3:].copy()
df.rename(columns={df.columns[0]: "日期", **rename_map}, inplace=True)

# 两段解析：含时分秒 / 仅日期
dt = pd.to_datetime(df["日期"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
mask = dt.isna()
if mask.any():
    dt2 = pd.to_datetime(df.loc[mask, "日期"], format="%Y-%m-%d", errors="coerce")
    dt[mask] = dt2
df["日期"] = dt
df = df[df["日期"].notna()].copy()
df.set_index("日期", inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")

# --------------------------------------------------------------------
# 4. 截取有效区间（首条有效观测 ~ 2019-10-01）
# --------------------------------------------------------------------
END_DATE = pd.Timestamp("2019-10-01")
valid_mask = df.notna().any(axis=1)
start_date = valid_mask.idxmax()
df = df.loc[start_date:END_DATE]            # 直接裁剪到 2019-10-01（含）

# --------------------------------------------------------------------
# 5. 插值
# --------------------------------------------------------------------
def interpolate_filled(data: pd.DataFrame, method: str):
    if method == "linear":
        return data.interpolate("linear", limit_direction="both")
    if method == "spline":
        try:
            import scipy  # noqa: F401
            return data.interpolate("spline", order=3, limit_direction="both")
        except ImportError:
            warnings.warn("未装 SciPy，样条降级为线性")
            return data.interpolate("linear", limit_direction="both")
    raise ValueError("method 仅支持 'linear' 或 'spline'")

df_filled = interpolate_filled(df, INTERP_METHOD)

# --------------------------------------------------------------------
# 6. 双-subplot 折线图
# --------------------------------------------------------------------
def plot_before_after(before: pd.DataFrame, after: pd.DataFrame,
                      tag: str, out_dir: Path):
    for col in before.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        # 原始
        axes[0].plot(before.index, before[col], lw=1, color="#1f77b4")
        axes[0].set_title("原始数据")
        axes[0].set_xlabel("日期"); axes[0].set_ylabel("地下水位")
        axes[0].grid(ls="--", alpha=.3)
        # 插值后
        axes[1].plot(after.index,  after[col], lw=1, color="#ff7f0e")
        axes[1].set_title(f"处理后（{tag}）")
        axes[1].set_xlabel("日期")
        axes[1].grid(ls="--", alpha=.3)
        fig.suptitle(col)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_dir / f"{col}.png", dpi=150)
        plt.close(fig)

plot_before_after(
    df, df_filled,
    tag="线性插值" if INTERP_METHOD == "linear" else "样条插值",
    out_dir=PLOT_DIR,
)
print("✅ 折线图已保存到:", PLOT_DIR.resolve())

# --------------------------------------------------------------------
# 7. 保存结果 Excel
# --------------------------------------------------------------------
with pd.ExcelWriter(OUT_EXCEL) as writer:
    df_filled.to_excel(writer, sheet_name=f"{INTERP_METHOD}插值")

print("✅ 插值数据已保存:", OUT_EXCEL)
