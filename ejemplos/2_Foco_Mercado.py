import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from reports.mercado_foco_pdf_report import MercadoFocoPdfReport, MercadoFocoPdfInputs


# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Mercado (Categorías y Marcas)",
    layout="wide",
    page_icon="https://chiper.cl/wp-content/uploads/2023/06/cropped-favicon-192x192.png",
)
st.title("Análisis de Mercado 2025")

DATA_CSV_PATH = os.getenv("FOCO_MERCADO_CSV_PATH", "/srv/streamlit/myapp/data/raw_utt_2.csv")
REQUIRED = {"fecha", "categoria", "marca", "venta"}

# Colores: UNIFICADOS entre tabla y mapa foco
FOCO_COLORS = {
    "Oportunidad": "#2CA02C",  # verde (plotly clásico)
    "Defender": "#D62728",     # rojo
    "Monitorear": "#7F7F7F",   # gris
}

# Rojo "Chiper" para resaltar el mes seleccionado en el gráfico de mercado
CHIPER_RED = "#E4002B"

# Colores para rol (semaforización auxiliar)
ROLE_COLORS = {
    "At Risk": "#D62728",
    "Seasonal": "#1F77B4",
    "Driver": "#FF7F0E",
    "Builder": "#2CA02C",
    "Monitoreo": "#7F7F7F",
}

# ======================================================
# HELPERS: IO + schema
# ======================================================
def normalize_colname(c: str) -> str:
    if c is None:
        return ""
    c = str(c).replace("\ufeff", "")  # BOM
    c = c.strip().lower()
    c = c.replace(" ", "_").replace("-", "_")
    c = c.replace("\r", "").replace("\n", "")
    return c


@st.cache_data(show_spinner=False)
def read_flexible_csv_from_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("No existe el archivo de mercado en el servidor.")
    try:
        return pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        pass
    for sep in [";", ","]:
        try:
            return pd.read_csv(path, sep=sep, dtype=str)
        except Exception:
            continue
    raise ValueError("No se pudo leer el CSV. Revisa separador/encoding/estructura.")


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    alias = {
        "fecha": "fecha", "mes": "fecha", "month": "fecha", "periodo": "fecha", "period": "fecha",
        "categoria": "categoria", "category": "categoria", "cat": "categoria",
        "marca": "marca", "brand": "marca",
        "venta": "venta", "ventas": "venta", "sales": "venta", "revenue": "venta", "monto": "venta",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    missing = REQUIRED - set(df.columns)
    if missing:
        st.error(f"Faltan columnas requeridas: {sorted(missing)}")
        st.write("Columnas detectadas:", list(df.columns))
        raise ValueError("CSV no cumple esquema mínimo.")
    return df


def to_month_start(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    dt = pd.to_datetime(s + "-01", errors="coerce")
    return dt.dt.to_period("M").dt.to_timestamp()


# ======================================================
# HELPERS: métricas y formato
# ======================================================
def safe_pct_change(cur, prev):
    if prev is None or pd.isna(prev) or prev == 0:
        return np.nan
    return (cur / prev) - 1.0


def fmt_money(x, currency="$"):
    if pd.isna(x):
        return "—"
    try:
        return f"{currency}{float(x):,.0f}"
    except Exception:
        return "—"


def fmt_pct(x, digits=4):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "—"


def fmt_pp(x, digits=2):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x) * 100:.{digits}f} pp"
    except Exception:
        return "—"


def month_label(dt):
    return pd.to_datetime(dt).strftime("%Y-%m")


def zscore_last(s: pd.Series):
    s = s.dropna()
    if len(s) < 4:
        return np.nan
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return np.nan
    return (s.iloc[-1] - mu) / sd


# ======================================================
# HELPERS: styling (colores consistentes con mapa foco)
# ======================================================
def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgba_bg(hex_color: str, alpha: float = 0.16):
    r, g, b = _hex_to_rgb(hex_color)
    return f"background-color: rgba({r},{g},{b},{alpha});"


def style_foco(val):
    if val in FOCO_COLORS:
        return _rgba_bg(FOCO_COLORS[val], 0.18)
    return ""


def style_rol(val):
    if not isinstance(val, str):
        return ""
    if val.startswith("At Risk"):
        key = "At Risk"
    elif val.startswith("Seasonal"):
        key = "Seasonal"
    elif val.startswith("Driver"):
        key = "Driver"
    elif val.startswith("Builder"):
        key = "Builder"
    else:
        key = "Monitoreo"
    return _rgba_bg(ROLE_COLORS.get(key, "#7F7F7F"), 0.12)


def style_delta_pct(val):
    if pd.isna(val):
        return ""
    if val >= 0.10:
        return _rgba_bg(FOCO_COLORS["Oportunidad"], 0.12)
    if val <= -0.05:
        return _rgba_bg(FOCO_COLORS["Defender"], 0.12)
    return ""


def style_delta_share(val):
    if pd.isna(val):
        return ""
    if val >= 0.003:
        return _rgba_bg(FOCO_COLORS["Oportunidad"], 0.12)
    if val <= -0.003:
        return _rgba_bg(FOCO_COLORS["Defender"], 0.12)
    return ""


def style_alert_text(val):
    if isinstance(val, str) and val.strip():
        return _rgba_bg(FOCO_COLORS["Defender"], 0.10)
    return ""


# ======================================================
# HELPERS: auto-detección de columnas a formatear
# ======================================================
def _infer_money_cols(cols):
    """
    Columnas a mostrar como $ (pesos).
    Regla: nombres típicos de venta/monto/total y deltas absolutos.
    """
    out = []
    for c in cols:
        lc = str(c).lower()

        # Nunca dinero si es share o cambio porcentual explícito
        if "share" in lc:
            continue
        if lc in {"delta_pct"}:
            continue
        if any(t in lc for t in ["pct", "percent", "porc", "porcentaje", "mom", "yoy", "wow", "growth", "rate"]):
            continue

        if lc in {"cat_total", "delta_venta"}:
            out.append(c)
            continue

        if any(t in lc for t in ["venta", "ventas", "revenue", "monto", "ingreso", "total", "gmv"]):
            out.append(c)

    # únicos manteniendo orden
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _infer_share_cols(cols):
    """
    Shares siempre van como % con 4 decimales.
    """
    out = [c for c in cols if "share" in str(c).lower()]
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _infer_pct_cols(cols):
    """
    Cambios porcentuales (NO shares) como % con 4 decimales.
    """
    out = []
    for c in cols:
        lc = str(c).lower()
        if "share" in lc:
            continue
        if lc == "delta_pct":
            out.append(c)
            continue
        if any(t in lc for t in ["pct", "percent", "porc", "porcentaje", "mom", "yoy", "wow", "growth", "chg", "change", "variacion", "cambio"]):
            out.append(c)

    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def dataframe_styled(df_show: pd.DataFrame, style_kind: str = "plain"):
    """
    Renderiza CUALQUIER tabla con:
    - $ para ventas/montos/delta_venta/etc.
    - % con 4 decimales para shares y cambios porcentuales (delta_pct, mom, etc.)
    - estilos semáforo cuando aplica
    """
    df_show = df_show.copy()

    # ---- detectar columnas
    money_cols = _infer_money_cols(df_show.columns)
    share_cols = _infer_share_cols(df_show.columns)
    pct_cols = _infer_pct_cols(df_show.columns)

    # ---- forzar numéricos (clave para que Streamlit aplique formato)
    for c in set(money_cols + share_cols + pct_cols + ["season_si", "cv12"]):
        if c in df_show.columns:
            df_show[c] = pd.to_numeric(df_show[c], errors="coerce")

    sty = df_show.style

    # ---- estilos semáforo
    if style_kind == "foco":
        if "foco" in df_show.columns:
            sty = sty.map(style_foco, subset=["foco"])
        if "rol" in df_show.columns:
            sty = sty.map(style_rol, subset=["rol"])
        if "delta_pct" in df_show.columns:
            sty = sty.map(style_delta_pct, subset=["delta_pct"])
        if "delta_share" in df_show.columns:
            sty = sty.map(style_delta_share, subset=["delta_share"])

    if style_kind == "brands":
        if "delta_pct" in df_show.columns:
            sty = sty.map(style_delta_pct, subset=["delta_pct"])

    if style_kind == "alerts":
        if "alerta" in df_show.columns:
            sty = sty.map(style_alert_text, subset=["alerta"])

    # ---- formatos: usar format strings (más estable en Streamlit)
    fmt_map = {}

    for c in money_cols:
        fmt_map[c] = "${:,.0f}"

    for c in share_cols:
        fmt_map[c] = "{:.4%}"

    for c in pct_cols:
        fmt_map[c] = "{:.4%}"

    # season_si y cv12 como ratios
    if "season_si" in df_show.columns:
        fmt_map["season_si"] = "{:.2f}"
    if "cv12" in df_show.columns:
        fmt_map["cv12"] = "{:.2f}"

    if fmt_map:
        sty = sty.format(fmt_map, na_rep="—")

    sty = sty.set_table_styles([
        {"selector": "th", "props": [("font-weight", "600")]},
        {"selector": "td", "props": [("padding", "6px 10px")]},
    ])

    st.dataframe(sty, use_container_width=True, hide_index=True)


# ======================================================
# LOAD + CLEAN
# ======================================================
st.sidebar.header("Parámetros")

try:
    raw = read_flexible_csv_from_path(DATA_CSV_PATH)
    raw = coerce_schema(raw)
except Exception as e:
    st.error(str(e))
    st.stop()

df = raw.copy()
df["categoria"] = df["categoria"].astype(str).str.strip()
df["marca"] = df["marca"].astype(str).str.strip()
df["fecha"] = to_month_start(df["fecha"])
df["venta"] = pd.to_numeric(df["venta"], errors="coerce")

df = df.dropna(subset=["fecha", "categoria", "marca", "venta"])
df = df[df["venta"] >= 0]

if df.empty:
    st.error("No hay datos válidos tras limpieza. Revisa fecha YYYY-MM y venta numérica.")
    st.stop()

df = df.groupby(["fecha", "categoria", "marca"], as_index=False)["venta"].sum()

months = sorted(df["fecha"].unique())
if not months:
    st.error("No hay meses detectados.")
    st.stop()

# ======================================================
# SIDEBAR (sin secciones desplegables)
# ======================================================
sel_month = st.sidebar.selectbox("Mes de análisis", months, index=len(months) - 1, format_func=month_label)

# (2) NUEVAS OPCIONES DE COMPARACIÓN
compare_mode = st.sidebar.selectbox(
    "Comparar contra",
    ["Mes anterior", "2 meses", "3 meses", "6 meses", "1 año"],
    index=0,
)

if compare_mode == "Mes anterior":
    cmp_month = (pd.to_datetime(sel_month) - pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
elif compare_mode == "2 meses":
    cmp_month = (pd.to_datetime(sel_month) - pd.offsets.MonthBegin(2)).to_period("M").to_timestamp()
elif compare_mode == "3 meses":
    cmp_month = (pd.to_datetime(sel_month) - pd.offsets.MonthBegin(3)).to_period("M").to_timestamp()
elif compare_mode == "6 meses":
    cmp_month = (pd.to_datetime(sel_month) - pd.offsets.MonthBegin(6)).to_period("M").to_timestamp()
else:  # "1 año"
    cmp_month = (pd.to_datetime(sel_month) - pd.DateOffset(years=1)).to_period("M").to_timestamp()

cmp_exists = cmp_month in set(months)

df_sel_all = df[df["fecha"] == sel_month].copy()
cats_in_month = sorted(df_sel_all["categoria"].unique())
brands_in_month = sorted(df_sel_all["marca"].unique())

st.sidebar.markdown("---")
sel_cats = st.sidebar.multiselect(
    "Categorías (opcional)",
    cats_in_month,
    default=[],
)
sel_brands = st.sidebar.multiselect("Marcas (opcional)", brands_in_month, default=[])

st.sidebar.markdown("---")
top_n = st.sidebar.slider("Top N", min_value=5, max_value=50, value=15, step=5)
only_top_categories = st.sidebar.toggle("Optimizar: ver solo Top 20 categorías del mes", value=False)

# (3) NUEVO: número de categorías para el mix "Qué compra el cliente"
st.sidebar.markdown("---")
st.sidebar.subheader("Mix Cliente")
mix_top_cats = st.sidebar.number_input(
    "Número de categorías en mix",
    min_value=1,
    max_value=max(1, len(cats_in_month)),
    value=min(8, max(1, len(cats_in_month))),
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Alertas (Hit List)")
alert_neg_streak = st.sidebar.selectbox("Racha negativa (meses)", [2, 3, 4], index=1)
alert_share_loss_pp = st.sidebar.slider("Pérdida de share (pp)", 0.1, 5.0, 0.5, 0.1) / 100.0
alert_shock_z = st.sidebar.slider("Cambio brusco (|z|)", 1.5, 4.0, 2.5, 0.1)

# (4) NUEVO: toggle hit list + selector de ranking
st.sidebar.markdown("---")
st.sidebar.subheader("Hit List: configuración")
show_hitlist = st.sidebar.toggle("Mostrar Hit List", value=True)

hit_rank_label = st.sidebar.selectbox(
    "Ordenar ganadores/perdedores por",
    ["Δ venta", "Venta", "Share", "Δ share"],
    index=0,
)
hit_rank_map = {
    "Δ venta": "delta_venta",
    "Venta": "venta_cur",
    "Share": "share_cur",
    "Δ share": "delta_share",
}
hit_rank_col = hit_rank_map[hit_rank_label]

df_f = df[df["categoria"].isin(sel_cats)].copy() if sel_cats else df.copy()
if sel_brands:
    df_f = df_f[df_f["marca"].isin(sel_brands)]

# ======================================================
# CÁLCULOS PRINCIPALES
# ======================================================
def month_view(d: pd.DataFrame, month):
    return d[d["fecha"] == month].copy()

cur = month_view(df_f, sel_month)
prev = month_view(df_f, cmp_month) if cmp_exists else None

cur_total = cur["venta"].sum()
prev_total = prev["venta"].sum() if prev is not None else np.nan
market_delta = safe_pct_change(cur_total, prev_total)

# ---- Categorías
cur_cat = cur.groupby("categoria", as_index=False)["venta"].sum().rename(columns={"venta": "venta_cur"})
cur_cat["share_cur"] = np.where(cur_total > 0, cur_cat["venta_cur"] / cur_total, 0.0)

if prev is not None:
    prev_cat = prev.groupby("categoria", as_index=False)["venta"].sum().rename(columns={"venta": "venta_prev"})
    prev_total2 = float(prev_total) if pd.notna(prev_total) else 0.0
    prev_cat["share_prev"] = np.where(prev_total2 > 0, prev_cat["venta_prev"] / prev_total2, 0.0)

    cat = cur_cat.merge(prev_cat, on="categoria", how="left")
    cat["venta_prev"] = cat["venta_prev"].fillna(0.0)
    cat["share_prev"] = cat["share_prev"].fillna(0.0)
    cat["delta_venta"] = cat["venta_cur"] - cat["venta_prev"]
    cat["delta_pct"] = np.where(cat["venta_prev"] > 0, (cat["venta_cur"] / cat["venta_prev"]) - 1.0, np.nan)
    cat["delta_share"] = cat["share_cur"] - cat["share_prev"]
else:
    cat = cur_cat.copy()
    cat["venta_prev"] = np.nan
    cat["share_prev"] = np.nan
    cat["delta_venta"] = np.nan
    cat["delta_pct"] = np.nan
    cat["delta_share"] = np.nan

cat = cat.sort_values("venta_cur", ascending=False)
cat_focus_base = cat.head(20).copy() if only_top_categories else cat.copy()

# ---- Categoría-marca
cur_cb = cur.groupby(["categoria", "marca"], as_index=False)["venta"].sum().rename(columns={"venta": "venta_cur"})
cur_cb = cur_cb.merge(
    cur_cat[["categoria", "venta_cur"]].rename(columns={"venta_cur": "cat_total"}),
    on="categoria",
    how="left",
)
cur_cb["share_in_cat"] = np.where(cur_cb["cat_total"] > 0, cur_cb["venta_cur"] / cur_cb["cat_total"], 0.0)

if prev is not None:
    prev_cb = prev.groupby(["categoria", "marca"], as_index=False)["venta"].sum().rename(columns={"venta": "venta_prev"})
    cur_cb = cur_cb.merge(prev_cb, on=["categoria", "marca"], how="left")
    cur_cb["venta_prev"] = cur_cb["venta_prev"].fillna(0.0)
    cur_cb["delta_venta"] = cur_cb["venta_cur"] - cur_cb["venta_prev"]
    cur_cb["delta_pct"] = np.where(cur_cb["venta_prev"] > 0, (cur_cb["venta_cur"] / cur_cb["venta_prev"]) - 1.0, np.nan)
else:
    cur_cb["venta_prev"] = np.nan
    cur_cb["delta_venta"] = np.nan
    cur_cb["delta_pct"] = np.nan

# ---- Basket
def build_basket(cat_df, cb_df, top_c=5, top_b=5):
    top_cats = cat_df.sort_values("venta_cur", ascending=False).head(top_c)["categoria"].tolist()
    out = []
    for c in top_cats:
        sub = cb_df[cb_df["categoria"] == c].sort_values("venta_cur", ascending=False).head(top_b).copy()
        out.append((c, sub))
    return out

basket = build_basket(cat, cur_cb, top_c=int(mix_top_cats), top_b=5)

# ---- Foco cualitativo
df_ts = df_f.groupby(["fecha", "categoria"], as_index=False)["venta"].sum().sort_values(["categoria", "fecha"])

def cat_cv12(c):
    s = df_ts[df_ts["categoria"] == c].sort_values("fecha")["venta"]
    tail = s.tail(min(12, len(s)))
    mu = tail.mean()
    sd = tail.std(ddof=0)
    if mu == 0 or pd.isna(mu):
        return np.nan
    return float(sd / mu)

cat_focus = cat_focus_base.copy()
cat_focus["cv12"] = cat_focus["categoria"].map(cat_cv12)

size_thr = cat_focus["venta_cur"].quantile(0.65) if len(cat_focus) >= 5 else cat_focus["venta_cur"].median()
oppty_thr = 0.10
share_gain_thr = 0.003
share_loss_thr = -0.003

def foco_label(row):
    big = row["venta_cur"] >= size_thr
    if prev is None:
        return "Oportunidad" if big else "Monitorear"
    dp = row["delta_pct"]
    ds = row["delta_share"]
    if (pd.notna(dp) and dp >= oppty_thr and big) or (pd.notna(ds) and ds >= share_gain_thr and big):
        return "Oportunidad"
    if big and ((pd.notna(dp) and dp <= -0.05) or (pd.notna(ds) and ds <= share_loss_thr)):
        return "Defender"
    return "Monitorear"

def foco_reason(row):
    if prev is None:
        return "Prioridad por tamaño del mes."
    reasons = []
    if pd.notna(row["delta_share"]) and abs(row["delta_share"]) >= 0.002:
        reasons.append(f"Δshare {fmt_pp(row['delta_share'])}")
    if pd.notna(row["delta_pct"]):
        reasons.append(f"Δventa {fmt_pct(row['delta_pct'], 1)}")
    q70 = cat_focus["cv12"].quantile(0.7) if cat_focus["cv12"].notna().any() else np.nan
    if pd.notna(q70) and pd.notna(row["cv12"]) and row["cv12"] >= q70:
        reasons.append("Alta variabilidad")
    return " · ".join(reasons) if reasons else "Cambio menor / estable."

cat_focus["foco"] = cat_focus.apply(foco_label, axis=1)
cat_focus["razon"] = cat_focus.apply(foco_reason, axis=1)

# ---- Seasonality index
df_ts2 = df_f.groupby(["fecha", "categoria"], as_index=False)["venta"].sum()
df_ts2["m"] = df_ts2["fecha"].dt.month
month_num = pd.to_datetime(sel_month).month

def seasonal_index(c, m):
    sub = df_ts2[df_ts2["categoria"] == c]
    if sub.empty:
        return np.nan
    mean_all = sub["venta"].mean()
    if mean_all == 0 or pd.isna(mean_all):
        return np.nan
    mean_m = sub[sub["m"] == m]["venta"].mean()
    if pd.isna(mean_m):
        return np.nan
    return float(mean_m / mean_all)

cat_focus["season_si"] = cat_focus["categoria"].map(lambda c: seasonal_index(c, month_num))

share_hi = cat_focus["share_cur"].quantile(0.75) if len(cat_focus) >= 4 else cat_focus["share_cur"].max()

def plan_role(row):
    if row["foco"] == "Defender":
        return "At Risk (corregir)"
    if pd.notna(row["season_si"]) and row["season_si"] >= 1.15:
        return "Seasonal (calendarizar)"
    if row["share_cur"] >= share_hi and row["venta_cur"] >= size_thr:
        return "Driver (defender)"
    if row["foco"] == "Oportunidad":
        return "Builder (capturar)"
    return "Monitoreo"

def plan_action(row):
    rol = row["rol"]
    if rol.startswith("At Risk"):
        return "Asegurar disponibilidad, ajustar condiciones/promos, revisar competitividad y surtido."
    if rol.startswith("Seasonal"):
        return "Preparar plan de activación: abastecimiento + ejecución antes del pico estacional."
    if rol.startswith("Driver"):
        return "Proteger continuidad: stock/visibilidad, evitar disrupciones, cuidar ejecución."
    if rol.startswith("Builder"):
        return "Aprovechar tracción: empujar ejecución y evaluar captura de valor con condiciones."
    return "Monitoreo regular. Intervenir solo ante alertas o cambio de share."

cat_focus["rol"] = cat_focus.apply(plan_role, axis=1)
cat_focus["accion"] = cat_focus.apply(plan_action, axis=1)

# ---- Hit list
alerts = []
df_alerts = pd.DataFrame(columns=["categoria", "venta_mes", "share_mes", "alerta"])

if prev is not None:
    ts = df_f.groupby(["fecha", "categoria"], as_index=False)["venta"].sum().sort_values(["categoria", "fecha"])
    market_ts = df_f.groupby("fecha", as_index=False)["venta"].sum().rename(columns={"venta": "venta_m"})
    ts = ts.merge(market_ts, on="fecha", how="left")
    ts["share"] = np.where(ts["venta_m"] > 0, ts["venta"] / ts["venta_m"], 0.0)
    ts["mom"] = ts.groupby("categoria")["venta"].pct_change()

    for c in cat["categoria"].tolist():
        sub = ts[ts["categoria"] == c].sort_values("fecha")

        tail_mom = sub["mom"].dropna().tail(alert_neg_streak)
        neg_streak = (len(tail_mom) == alert_neg_streak) and (tail_mom < 0).all()

        s_cur = float(sub[sub["fecha"] == sel_month]["share"].iloc[0]) if not sub[sub["fecha"] == sel_month].empty else np.nan
        s_cmp = float(sub[sub["fecha"] == cmp_month]["share"].iloc[0]) if not sub[sub["fecha"] == cmp_month].empty else 0.0
        ds = s_cur - s_cmp

        tail = sub.tail(min(12, len(sub)))
        zl = zscore_last(tail["mom"])
        shock = (abs(zl) >= alert_shock_z) if pd.notna(zl) else False

        if neg_streak or (ds < -alert_share_loss_pp) or shock:
            reasons = []
            if neg_streak:
                reasons.append(f"Racha negativa {alert_neg_streak}m")
            if ds < -alert_share_loss_pp:
                reasons.append(f"Pérdida share {fmt_pp(ds)}")
            if shock:
                reasons.append("Cambio fuera de lo normal")
            v_cur = float(sub[sub["fecha"] == sel_month]["venta"].iloc[0]) if not sub[sub["fecha"] == sel_month].empty else 0.0
            alerts.append({"categoria": c, "venta_mes": v_cur, "share_mes": s_cur, "alerta": " · ".join(reasons)})

    df_alerts = (
        pd.DataFrame(alerts).sort_values("venta_mes", ascending=False)
        if alerts else
        pd.DataFrame(columns=["categoria", "venta_mes", "share_mes", "alerta"])
    )

# (5) Hit list ranking flexible (solo cambia el ORDEN, no el resto)
def _can_rank(col: str, prev_ok: bool) -> bool:
    if col in {"delta_venta", "delta_share"}:
        return prev_ok
    return True

prev_ok = prev is not None

if not _can_rank(hit_rank_col, prev_ok):
    winners = cat.sort_values("venta_cur", ascending=False).head(top_n) if len(cat) else cat.head(0)
    losers = cat.sort_values("venta_cur", ascending=True).head(top_n) if len(cat) else cat.head(0)
else:
    winners = cat.sort_values(hit_rank_col, ascending=False).head(top_n) if len(cat) else cat.head(0)
    losers = cat.sort_values(hit_rank_col, ascending=True).head(top_n) if len(cat) else cat.head(0)

# ======================================================
# UI
# ======================================================
st.markdown("---")
ctx = st.columns([1, 1, 2])
with ctx[0]:
    st.metric("Mes analizado", month_label(sel_month))
with ctx[1]:
    st.metric("Comparación", month_label(cmp_month) if cmp_exists else "No disponible")
with ctx[2]:
    st.write(f"Categorías: {len(sel_cats) if sel_cats else 'todas'} | Marcas: {'todas' if not sel_brands else len(sel_brands)}")

# A) Mercado
st.markdown("---")
st.subheader("Información de Mercado")

c1, c2 = st.columns(2)
c1.metric("Venta total mes", fmt_money(cur_total))
c2.metric("Δ vs comparación", fmt_pct(market_delta, 4) if prev is not None else "—")

if prev is not None and len(cat) >= 1:
    drivers = cat.sort_values("delta_venta", ascending=False).head(3)["categoria"].tolist()
    brakes = cat.sort_values("delta_venta", ascending=True).head(3)["categoria"].tolist()
else:
    drivers, brakes = [], []

if drivers or brakes:
    l, r = st.columns(2)
    with l:
        if drivers:
            st.markdown("**Empujan**")
            st.write("\n".join([f"- {x}" for x in drivers]))
    with r:
        if brakes:
            st.markdown("**Frenan**")
            st.write("\n".join([f"- {x}" for x in brakes]))

market_line = df_f.groupby("fecha", as_index=False)["venta"].sum().sort_values("fecha")
market_line = market_line.tail(min(24, len(market_line)))
fig_m = px.line(market_line, x="fecha", y="venta", markers=True, title="Venta mercado (últimos meses)")

# (6) Resaltar el punto del mes seleccionado con rojo Chiper
sel_point = market_line[market_line["fecha"] == sel_month]
if not sel_point.empty:
    fig_m.add_trace(
        go.Scatter(
            x=sel_point["fecha"],
            y=sel_point["venta"],
            mode="markers",
            marker=dict(color=CHIPER_RED, size=12, line=dict(width=2, color=CHIPER_RED)),
            name="Mes seleccionado",
            showlegend=False,
            hovertemplate="Mes seleccionado<br>%{x|%Y-%m}<br>Venta: %{y}<extra></extra>",
        )
    )

fig_m.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig_m, use_container_width=True)

top_cat = (
    cat
    .sort_values("venta_cur", ascending=False)  # mayor → menor
    .head(min(10, len(cat)))
    .copy()
)

fig_cat = px.bar(
    top_cat,
    x="venta_cur",
    y="categoria",
    orientation="h",
    title="Qué vende en mes X: Top categorías",
)

fig_cat.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=50, b=10),
    yaxis=dict(autorange="reversed"),  # mayor arriba
)

st.plotly_chart(fig_cat, use_container_width=True)


# B) Qué compra el cliente
st.markdown("---")
st.subheader("Qué compra el cliente en esta fecha (mix categoría → marca)")

# Render en filas de 5 columnas (para soportar cualquier N sin romper layout)
if basket:
    chunk = 4
    for start in range(0, len(basket), chunk):
        row = basket[start:start + chunk]
        cols = st.columns(len(row))
        for i, (c, sub) in enumerate(row):
            with cols[i]:
                row_cat = cur_cat[cur_cat["categoria"] == c]
                total_c = float(row_cat["venta_cur"].iloc[0]) if not row_cat.empty else 0.0

                # Share de mercado de la categoría (dentro del mercado filtrado del mes)
                share_c = float(row_cat["share_cur"].iloc[0]) if (
                            "share_cur" in row_cat.columns and not row_cat.empty) else np.nan

                st.markdown(
                    f"**{c}**  \n"
                    f"Venta: {fmt_money(total_c)}  \n"
                    f"Share mercado: {fmt_pct(share_c, 4)}"
                )

                sub_tbl = sub[["marca", "venta_cur", "share_in_cat"]].copy()
                dataframe_styled(sub_tbl, style_kind="plain")

# C) Marcas relevantes por categoría
st.markdown("---")
st.subheader("Marcas relevantes por categoría")

left, right = st.columns([1, 2])
with left:
    cat_sel = st.selectbox("Categoría", cat["categoria"].tolist() if len(cat) else cats_in_month)

with right:
    sub = cur_cb[cur_cb["categoria"] == cat_sel].sort_values("venta_cur", ascending=False).head(top_n).copy()

    if prev is not None:
        cols_show = ["marca", "venta_cur", "share_in_cat", "delta_venta", "delta_pct"]
    else:
        cols_show = ["marca", "venta_cur", "share_in_cat"]

    dataframe_styled(sub[cols_show].copy(), style_kind="brands")

    top_b = (
        sub.sort_values("venta_cur", ascending=False)
        .head(min(12, len(sub)))
        .copy()
    )

    fig_b = px.bar(
        top_b,
        x="venta_cur",
        y="marca",
        orientation="h",
        title=f"Top marcas en {cat_sel} — {month_label(sel_month)}",
    )

    fig_b.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(autorange="reversed"),  # mayor arriba
    )

    st.plotly_chart(fig_b, use_container_width=True)

# D) Plan de Pricing + Categorías Foco
st.markdown("---")
st.subheader("Plan de Pricing + Categorías Foco (directo y accionable)")

order_map = {"Defender": 0, "Oportunidad": 1, "Monitorear": 2}
foco_tbl = cat_focus.copy()
foco_tbl["__ord"] = foco_tbl["foco"].map(order_map).fillna(9).astype(int)
foco_tbl = foco_tbl.sort_values(["__ord", "venta_cur"], ascending=[True, False]).drop(columns="__ord")

show_cols = ["categoria", "foco", "rol", "accion", "venta_cur", "share_cur", "delta_pct", "delta_share", "season_si", "razon"]
dataframe_styled(foco_tbl[show_cols].copy(), style_kind="foco")

# ======================================================
# DEFINICIONES ACCIONABLES
# ======================================================
with st.expander("Definiciones de accionables", expanded=False):
    st.markdown("""
## Interpretación de las acciones de pricing (lógica práctica)

La tabla **Plan de Pricing + Categorías Foco** traduce señales de mercado (cambios de venta y share) a una recomendación operativa:

### 1) Defender → **At Risk (corregir)**
**Señal típica:** categoría grande con caída relevante en venta o pérdida de share.  
**Interpretación:** se está perdiendo tracción contra el resto del mercado (o contra sustitutos).  
**Qué significa en pricing/ejecución:** priorizar continuidad y competitividad:  
- Asegurar disponibilidad (quiebres destruyen share).  
- Revisar precio/condiciones/promos (si el share cae, el cliente está eligiendo alternativas).  
- Ajustar surtido o execution (visibilidad, facing, etc.).

### 2) Seasonal → **Seasonal (calendarizar)**
**Señal típica:** $SI \\ge 1.15$ (mes fuerte recurrente).  
**Interpretación:** la demanda sube por calendario, no necesariamente por acción competitiva.  
**Qué significa:** planificar con anticipación:  
- Abastecimiento + ejecución antes del pico.  
- Promos/condiciones calibradas para capturar volumen sin destruir margen.

### 3) Driver → **Driver (defender)**
**Señal típica:** alta participación y tamaño (pilares del mix).  
**Interpretación:** estas categorías “sostienen” el resultado; el riesgo es la disrupción.  
**Qué significa:** proteger estabilidad:  
- Evitar quiebres y rupturas de precio (cambios bruscos generan fuga).  
- Mantener estrategia de pricing consistente (no “experimentar” en el core sin control).

### 4) Oportunidad → **Builder (capturar)**
**Señal típica:** crecimiento fuerte en venta o ganancia de share (especialmente en categorías grandes).  
**Interpretación:** hay tracción; el mercado “te está dando permiso” para capturar valor o acelerar.  
**Qué significa:** empujar ejecución:  
- Evaluar captura (mejorar margen/condiciones sin frenar demasiado).  
- Escalar la estrategia que está funcionando (surtido, promos, exhibición).

### 5) Monitorear → **Monitoreo**
**Señal típica:** cambios menores o categoría pequeña con señales débiles.  
**Interpretación:** no justifica intervención inmediata.  
**Qué significa:** observar y actuar solo si aparece alerta (racha negativa, pérdida de share, shock).

**Regla mental clave:**  
- Si **share de mercado baja** (delta_share < 0), aunque la venta no caiga, suele indicar pérdida relativa frente al mercado → revisar competitividad/ejecución.  
- Si **share sube**, indica ganancia relativa → oportunidad para consolidar (o capturar valor con cuidado).
""")

# Mapa foco (con colores fijos)
if prev is not None:
    mat = cat_focus.copy()
    mat["bubble"] = mat["share_cur"].fillna(0.0).clip(lower=0.0) + 1e-6

    # --- auto-ajuste eje Y (delta_pct)
    y = pd.to_numeric(mat["delta_pct"], errors="coerce").dropna()
    if len(y) >= 4:
        q05, q95 = np.nanpercentile(y, [5, 95])
        span = max(q95 - q05, 0.06)          # rango mínimo para que "respire"
        pad = span * 0.35                    # padding
        y_min = q05 - pad
        y_max = q95 + pad
    elif len(y) >= 1:
        mu = float(y.mean())
        y_min, y_max = mu - 0.15, mu + 0.15  # fallback simple
    else:
        y_min, y_max = -0.2, 0.2

    fig_mat = px.scatter(
        mat,
        x="venta_cur",
        y="delta_pct",
        size="bubble",
        color="foco",
        color_discrete_map=FOCO_COLORS,
        hover_name="categoria",
        title="Mapa foco: tamaño del mes vs % cambio",
        size_max=35,
    )

    fig_mat.update_yaxes(range=[y_min, y_max], tickformat=".1%")  # <-- rango dinámico + formato %
    fig_mat.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_mat, use_container_width=True)
else:
    st.info("No hay mes de comparación; el mapa foco requiere comparación para Δ%.")

# ======================================================
# PDF descargable (Plan + Definiciones + Hitlist + Alertas)
# ======================================================

def _build_definiciones_html() -> str:
    # Texto “equivalente” al expander, en HTML simple compatible con ReportLab Paragraph.
    # (Evito Markdown/LaTeX para que el render sea limpio.)
    return """
    La tabla <b>Plan de Pricing + Categorías Foco</b> traduce señales (venta/share) a acciones.
    <br/><br/>
    <b>1) Defender - At Risk (corregir)</b><br/>
    Categoria grande con caida relevante en venta o perdida de share. Accion: continuidad y competitividad (stock, precio/condiciones, promos, surtido, ejecucion).
    <br/><br/>
    <b>2) Seasonal - Seasonal (calendarizar)</b><br/>
    SI >= 1.15 sugiere mes fuerte recurrente. Accion: planificar abastecimiento y activacion antes del pico, promos calibradas.
    <br/><br/>
    <b>3) Driver - Driver (defender)</b><br/>
    Alta participacion y tamaño. Accion: proteger estabilidad, evitar quiebres y cambios bruscos en el core.
    <br/><br/>
    <b>4) Oportunidad - Builder (capturar)</b><br/>
    Crecimiento fuerte en venta o ganancia de share. Accion: empujar ejecucion y evaluar captura de valor con cuidado.
    <br/><br/>
    <b>5) Monitorear - Monitoreo</b><br/>
    Cambios menores o categoria pequeña. Accion: observar y actuar solo ante alertas.
    <br/><br/>
    <b>Regla mental:</b> si el share baja, aunque la venta no caiga, suele indicar perdida relativa; si el share sube, indica ganancia relativa.
    """

def _month_label(dt):
    return pd.to_datetime(dt).strftime("%Y-%m")

# Arma parámetros visibles en el PDF
pdf_params = {
    "Mes analizado": _month_label(sel_month),
    "Comparacion": _month_label(cmp_month) if cmp_exists else "No disponible",
    "Comparar contra": str(compare_mode),
    "Categorias": f"{len(sel_cats)} seleccionadas" if sel_cats else "Todas",
    "Marcas": f"{len(sel_brands)} seleccionadas" if sel_brands else "Todas",
    "Top N": str(top_n),
    "Optimizacion Top20": "Si" if only_top_categories else "No",
}

# Selecciona columnas exactas del plan (por seguridad)
plan_cols = ["categoria", "foco", "rol", "accion", "venta_cur", "share_cur", "delta_pct", "delta_share", "season_si", "razon"]
foco_pdf_tbl = foco_tbl[plan_cols].copy()

# Hitlist: asegurar columnas estables
hit_cols = ["categoria", "venta_cur", "venta_prev", "delta_venta", "delta_pct", "delta_share"]
w_pdf = winners[hit_cols].copy() if not winners.empty else winners.copy()
l_pdf = losers[hit_cols].copy() if not losers.empty else losers.copy()

# Alertas: asegurar columnas estables
al_cols = ["categoria", "venta_mes", "share_mes", "alerta"]
a_pdf = df_alerts[al_cols].copy() if (df_alerts is not None and not df_alerts.empty) else df_alerts

inputs = MercadoFocoPdfInputs(
    titulo="Reporte Mercado (Plan pricing, Hit List y Alertas)",
    params=pdf_params,
    foco_tbl=foco_pdf_tbl,
    winners=w_pdf,
    losers=l_pdf,
    df_alerts=a_pdf,
    definiciones_html=_build_definiciones_html(),
)

pdf_bytes = MercadoFocoPdfReport.build(inputs=inputs)

st.download_button(
    label="Descargar PDF (Plan pricing + Hit List + Alertas)",
    data=pdf_bytes,
    file_name=f"reporte_mercado_{_month_label(sel_month)}.pdf",
    mime="application/pdf",
    use_container_width=False,
)


# E) Hit list
st.markdown("---")
st.subheader("Hit List")

if not show_hitlist:
    st.info("Hit List desactivado en el sidebar.")
else:
    if prev is None and hit_rank_col in {"delta_venta", "delta_share"}:
        st.info("Comparación no disponible para este mes. Para Hit List por Δ, selecciona un mes con comparación o cambia el ranking a Venta/Share.")
    else:
        a, b = st.columns(2)
        with a:
            st.markdown(f"**Ganadores (Top por {hit_rank_label})**")
            ww = winners[["categoria", "venta_cur", "venta_prev", "delta_venta", "delta_pct", "delta_share"]].copy()
            dataframe_styled(ww, style_kind="foco")

        with b:
            st.markdown(f"**Perdedores (Top por {hit_rank_label})**")
            ll = losers[["categoria", "venta_cur", "venta_prev", "delta_venta", "delta_pct", "delta_share"]].copy()
            dataframe_styled(ll, style_kind="foco")

        st.markdown("### Alertas")
        if prev is None:
            st.info("Comparación no disponible para el mes seleccionado.")
        else:
            if df_alerts.empty:
                st.success("Sin alertas bajo umbrales actuales.")
            else:
                dataframe_styled(df_alerts[["categoria", "venta_mes", "share_mes", "alerta"]].copy(), style_kind="alerts")

# ======================================================
# DEFINICIONES / GLOSARIO
# ======================================================
st.markdown("---")
with st.expander("Definiciones de indicadores (glosario)", expanded=False):
    st.markdown("""
### Variables base

**venta_cur**: venta total de la categoría en el mes seleccionado.  
$$
venta_{cur}(c)=\\sum_{b} venta(m,c,b)
$$

**venta_prev**: venta total de la categoría en el mes de comparación.  
$$
venta_{prev}(c)=\\sum_{b} venta(m_0,c,b)
$$

**delta_venta**: cambio absoluto en venta.  
$$
\\Delta venta(c)=venta_{cur}(c)-venta_{prev}(c)
$$

**delta_pct**: crecimiento porcentual (solo si $venta_{prev}>0$).  
$$
\\Delta\\%(c)=\\frac{venta_{cur}(c)}{venta_{prev}(c)}-1
$$

### Shares

**share_cur**: participación de la categoría dentro del mercado filtrado del mes.  
$$
share_{cur}(c)=\\frac{venta_{cur}(c)}{\\sum_{c} venta_{cur}(c)}
$$

**delta_share**: cambio de share.  
$$
\\Delta share(c)=share_{cur}(c)-share_{prev}(c)
$$

### Marcas dentro de categoría

**share_in_cat**: participación de una marca dentro de una categoría en el mes.  
$$
share_{in\\_cat}(c,b)=\\frac{venta(m,c,b)}{venta_{cur}(c)}
$$

### Variabilidad

**cv12**: coeficiente de variación.  
$$
CV=\\frac{\\sigma(venta)}{\\mu(venta)}
$$

### Estacionalidad

**season_si**: índice estacional (promedio del mes calendario / promedio total).  
$$
SI(c,m)=\\frac{E[venta(c)\\mid mes=m]}{E[venta(c)]}
$$

Regla: si $SI\\ge 1.15$ sugiere mes estacional.
""")

st.caption(
    "Nota: el “mercado” aquí es el universo filtrado (categorías/marcas). Con precio/margen se puede convertir en plan cuantitativo."
)
