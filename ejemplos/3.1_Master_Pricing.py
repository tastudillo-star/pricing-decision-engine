from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.data_layer import (
    build_master_dataset,
    default_focus_month,
    load_foco_cat_table,
)

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuracion de pagina
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Posicionamiento + Foco (UTT → Chiper)",
    layout="wide",
    page_icon="https://chiper.cl/wp-content/uploads/2023/06/cropped-favicon-192x192.png",
)
st.title("Posicionamiento + Foco (UTT → Chiper)")
st.caption("Cruza posicionamiento (SKU/competidor) con foco de mercado (categorías UTT) usando tabla de mapeo categoria_utt.")


# ---------------------------------------------------------------------------
# Sidebar: controles y cache
# ---------------------------------------------------------------------------
st.sidebar.subheader("Recargar data")
if st.sidebar.button("Limpiar caché (data layer)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.pop("_df_master", None)
    st.session_state.pop("_df_master_meta", None)
    st.session_state.pop("_last_sig", None)
    st.session_state.pop("_has_run", None)
    st.rerun()

st.sidebar.subheader("Parámetros — Posicionamiento")
COMPETIDORES = {1: "Central Mayorista", 2: "Alvi", 3: "La Oferta", 4: "Mejor precio (min entre 1–3)"}
id_competidor = st.sidebar.selectbox(
    "Competidor",
    options=list(COMPETIDORES.keys()),
    format_func=lambda x: f"{x} – {COMPETIDORES.get(x, 'Competidor')}",
    index=3,
)

def last_sunday(d: pd.Timestamp) -> pd.Timestamp:
    days_since_sun = (d.weekday() - 6) % 7
    return d - pd.Timedelta(days=days_since_sun)

fecha_base = st.sidebar.date_input("Fecha base (END ventanas)", value=last_sunday(pd.Timestamp.today()))
VENTANA_PRESETS = {
    "Diario": 1,
    "Última semana (7 días)": 7,
    "Últimas 2 semanas (14 días)": 14,
    "Último mes (30 días)": 30,
    "Últimos 3 meses (90 días)": 90,
    "Personalizado": None,
}

def _pick_window(label: str, default: str) -> int:
    preset = VENTANA_PRESETS[label]
    if preset is None:
        return int(st.sidebar.number_input("Días ventana", min_value=1, max_value=365, value=30, step=1))
    return int(preset)

ventana_chiper = _pick_window(
    st.sidebar.selectbox("Ventana Chiper", options=list(VENTANA_PRESETS.keys()), index=1),
    "Última semana (7 días)",
)
ventana_comp = _pick_window(
    st.sidebar.selectbox("Ventana Competidor", options=list(VENTANA_PRESETS.keys()), index=1),
    "Última semana (7 días)",
)
aplicar_filtro_pos = st.sidebar.checkbox("Aplicar filtro posicionamiento (0.5 – 2.0)", value=True)
excluir_dias_sin_venta_chiper = st.sidebar.checkbox("Excluir días sin venta en Chiper", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros — Foco de Mercado")
import os
DATA_CSV_PATH_DEFAULT = os.getenv("FOCO_MERCADO_CSV_PATH", "/srv/streamlit/myapp/data/raw_utt_2.csv")
#DATA_CSV_PATH_DEFAULT = os.getenv("FOCO_MERCADO_CSV_PATH", r"C:\Users\tgast\PycharmProjects\StreamlitChiperPricingBI\data\raw_utt_2.csv")
csv_path = st.sidebar.text_input("CSV mercado (UTT)", value=DATA_CSV_PATH_DEFAULT)

# Carga ligera para poblar selector de meses (cacheada)
try:
    df_foco_preview = load_foco_cat_table(csv_path)
    foco_months = sorted(df_foco_preview["fecha"].dropna().unique())
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"No se pudieron leer meses del CSV: {exc}")
    foco_months = []

# Default: mes actual del año anterior + 1 mes
_default_month = default_focus_month()
if foco_months:
    target = pd.to_datetime(_default_month)
    meses_ts = [pd.to_datetime(x) for x in foco_months]
    if target in meses_ts:
        idx_default = meses_ts.index(target)
    else:
        idx_default = max(0, min(len(meses_ts) - 1, np.searchsorted(meses_ts, target)))
    sel_month = st.sidebar.selectbox(
        "Mes foco (YYYY-MM)",
        options=foco_months,
        index=idx_default,
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m"),
    )
else:
    sel_month = None
    st.sidebar.warning("No se pudieron detectar meses en el CSV (revisar columnas/fecha).")

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros — Recomendación de Precio")
delta_max_precio = st.sidebar.slider("Δ precio máx (±%)", min_value=0.0, max_value=0.20, value=0.06, step=0.01, format="%.2f")
eps_low = st.sidebar.number_input("Elasticidad baja", value=-0.15, step=0.05, format="%.2f")
eps_mid = st.sidebar.number_input("Elasticidad media", value=-0.25, step=0.05, format="%.2f")
eps_high = st.sidebar.number_input("Elasticidad alta", value=-0.40, step=0.05, format="%.2f")
lambda_mkt = st.sidebar.slider("λ (impacto mercado)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
g_mkt_cap = st.sidebar.slider("Clamp g_mkt (|Δ venta|)", min_value=0.0, max_value=1.0, value=0.50, step=0.05, format="%.2f")
momentum_cap = st.sidebar.slider("Clamp momentum (Δshare/estac)", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.2f")

compare_mode = st.sidebar.selectbox(
    "Comparación foco",
    options=["MOM (mes anterior)", "YOY (año anterior)", "SIN comparación"],
    index=0,
)
run_btn = st.sidebar.button("Ejecutar")


# ---------------------------------------------------------------------------
# Ejecucion controlada con session_state
# ---------------------------------------------------------------------------
def _params_signature() -> str:
    return "|".join(
        [
            f"id_comp={int(id_competidor)}",
            f"fecha={pd.to_datetime(fecha_base).strftime('%Y-%m-%d')}",
            f"vch={int(ventana_chiper)}",
            f"vco={int(ventana_comp)}",
            f"fpos={int(bool(aplicar_filtro_pos))}",
            f"excl={int(bool(excluir_dias_sin_venta_chiper))}",
            f"csv={csv_path}",
            f"sel={pd.to_datetime(sel_month).strftime('%Y-%m') if sel_month is not None else 'None'}",
            f"cmp={compare_mode}",
            f"dmax={delta_max_precio:.4f}",
            f"eps={eps_low:.3f},{eps_mid:.3f},{eps_high:.3f}",
            f"lambda={lambda_mkt:.3f}",
            f"gcap={g_mkt_cap:.3f}",
            f"mcap={momentum_cap:.3f}",
        ]
    )

if "_has_run" not in st.session_state:
    st.session_state["_has_run"] = False
if "_last_sig" not in st.session_state:
    st.session_state["_last_sig"] = None

sig = _params_signature()
if run_btn:
    st.session_state["_has_run"] = True
    st.session_state["_last_sig"] = sig
    st.session_state.pop("_df_master", None)
    st.session_state.pop("_df_master_meta", None)

if st.session_state["_has_run"] and st.session_state.get("_last_sig") != sig:
    st.info("Cambiaste parámetros, vuelve a ejecutar.")
    st.stop()

if not st.session_state["_has_run"]:
    st.info("Configura parámetros y presiona **Ejecutar**.")
    st.stop()

if sel_month is None:
    st.error("No hay mes foco seleccionable. Revisa el CSV.")
    st.stop()


# ---------------------------------------------------------------------------
# Capa de datos: llamada unica a data_layer (cacheada)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=600)
def _run_master(sig_key: str) -> tuple[pd.DataFrame, dict]:
    df, meta = build_master_dataset(
        id_competidor=id_competidor,
        fecha_base=pd.to_datetime(fecha_base),
        ventana_chiper=ventana_chiper,
        ventana_comp=ventana_comp,
        aplicar_filtro_pos=aplicar_filtro_pos,
        excluir_dias_sin_venta_chiper=excluir_dias_sin_venta_chiper,
        csv_path=csv_path,
        sel_month=pd.to_datetime(sel_month),
        compare_mode=compare_mode,
        delta_max_precio=float(delta_max_precio),
        eps_low=float(eps_low),
        eps_mid=float(eps_mid),
        eps_high=float(eps_high),
        lambda_mkt=float(lambda_mkt),
        g_mkt_cap=float(g_mkt_cap),
        momentum_cap=float(momentum_cap),
    )
    return df, meta

with st.spinner("Calculando dataset maestro (cacheado)…"):
    try:
        df_show, meta = _run_master(sig)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error al preparar datos: {exc}")
        st.stop()

# guardar en session para descargas inmediatas
st.session_state["_df_master"] = df_show
st.session_state["_df_master_meta"] = meta

st.caption(f"Mapeo foco: {meta.get('n_utt_mapped', 0)}/{meta.get('n_utt_total', 0)} categorías UTT con id_categoria_chiper.")


# ---------------------------------------------------------------------------
# Umbrales dinamicos para formatos
# ---------------------------------------------------------------------------
def calculate_dynamic_thresholds(df: pd.DataFrame, col_name: str, center: float | None = None) -> dict:
    if col_name not in df.columns:
        return {"min": 0, "center": center or 0, "max": 1}
    s = pd.to_numeric(df[col_name], errors="coerce").dropna()
    if s.empty:
        return {"min": 0, "center": center or 0, "max": 1}
    if center is None:
        return {"min": float(s.quantile(0.05)), "center": float(s.median()), "max": float(s.quantile(0.95))}
    min_val = float(s.min())
    max_val = float(s.max())
    dist_min = center - float(s.quantile(0.05))
    dist_max = float(s.quantile(0.95)) - center
    max_dist = max(dist_min, dist_max) * 1.1 if max(dist_min, dist_max) > 0 else max(abs(min_val - center), abs(max_val - center)) or 1
    return {"min": center - max_dist, "center": center, "max": center + max_dist}

thr_pos = calculate_dynamic_thresholds(df_show, "posicionamiento", center=1.0)
thr_delta_pct = calculate_dynamic_thresholds(df_show, "delta_pct", center=0.0)
thr_delta_share = calculate_dynamic_thresholds(df_show, "delta_share", center=0.0)


# Colores/cuadrantes para scatter y roles
QUADRANT_COLOR = {
    1: "#2F80ED",  # Contribuyente
    2: "#9B51E0",  # Poderosa
    3: "#F2994A",  # Magnetica
    4: "#27AE60",  # Oportunista
}
QUADRANT_LABEL = {
    1: "Contribuyente",
    2: "Poderosa",
    3: "Magnética",
    4: "Oportunista",
}

# ---------------------------------------------------------------------------
# Scatter: Posicionamiento vs Margen (mismo estilo que página de posicionamiento)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### Posicionamiento vs Margen (Análisis por Segmento)")

scatter_mode = st.radio("Vista", options=["Macro", "Categoría", "Proveedor"], index=0, horizontal=True)
macro_sel = None
cat_sel = None

opts = df_show.copy()
opts["macro_categoria"] = opts["macro_categoria"].fillna("Sin macro")
opts["categoria"] = opts["categoria"].fillna("Sin categoría")
opts["proveedor"] = opts.get("proveedor", pd.Series(["Sin proveedor"] * len(opts))).fillna("Sin proveedor")
macros_opt = sorted(opts["macro_categoria"].dropna().unique().tolist())

if scatter_mode == "Categoría":
    macro_sel = st.selectbox("Macro a detallar", options=macros_opt or ["Sin macro"], index=0)
elif scatter_mode == "Proveedor":
    macro_sel = st.selectbox("Macro", options=macros_opt or ["Sin macro"], index=0)
    cats_opt = sorted(opts[opts["macro_categoria"] == macro_sel]["categoria"].dropna().unique().tolist()) if macro_sel else []
    cat_sel = st.selectbox("Categoría", options=cats_opt or ["Sin categoría"], index=0)


def _scope(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return df_in
    d = df_in.copy()
    if scatter_mode == "Macro":
        return d
    if scatter_mode == "Categoría":
        return d[d["macro_categoria"] == macro_sel]
    if scatter_mode == "Proveedor":
        if macro_sel:
            d = d[d["macro_categoria"] == macro_sel]
        if cat_sel:
            d = d[d["categoria"] == cat_sel]
        return d
    return d


def _agg_weighted(grp: pd.DataFrame) -> pd.Series:
    vt = float(grp["venta_neta"].sum(skipna=True) or 0.0)
    pos = float((grp["posicionamiento"] * grp["venta_neta"]).sum(skipna=True) / vt) if vt else np.nan
    mg = float((grp["margen"] * grp["venta_neta"]).sum(skipna=True) / vt) if vt else np.nan
    return pd.Series({"venta_neta_level": vt, "posicionamiento_pond": pos, "margen_pond": mg})

sc_df = _scope(opts)
if scatter_mode == "Macro":
    group_col = "macro_categoria"
elif scatter_mode == "Categoría":
    group_col = "categoria"
else:
    group_col = "proveedor"

agg_sc = sc_df.groupby(group_col, dropna=False).apply(_agg_weighted).reset_index().rename(columns={group_col: "entity"})
agg_sc = agg_sc.dropna(subset=["posicionamiento_pond", "margen_pond"])
agg_sc = agg_sc[agg_sc["venta_neta_level"] > 0]

if agg_sc.empty:
    st.info("No hay datos suficientes para el scatter.")
else:
    d_roles = sc_df.rename(columns={group_col: "entity"}).groupby("entity", dropna=False).agg({"rol_rep": lambda x: x.mode()[0] if len(x.mode()) else "N/A"}).reset_index()
    agg_sc = agg_sc.merge(d_roles, on="entity", how="left")

    def _rol_to_quadrant(rol: str) -> int | None:
        r = str(rol).lower()
        if "contribuyente" in r:
            return 1
        if "poderosa" in r:
            return 2
        if "magn" in r:
            return 3
        if "oportun" in r:
            return 4
        return None

    agg_sc["quadrant"] = agg_sc["rol_rep"].map(_rol_to_quadrant).astype("Int64")
    agg_sc["quadrant_label"] = agg_sc["quadrant"].map(QUADRANT_LABEL)
    agg_sc["color"] = agg_sc["quadrant"].map(QUADRANT_COLOR).fillna("#7F7F7F")
    agg_sc["pos_pct"] = agg_sc["posicionamiento_pond"] * 100.0
    agg_sc["margen_pct"] = agg_sc["margen_pond"] * 100.0

    x_ref = 100.0
    y_ref = 17.72

    x_min_data = float(agg_sc["pos_pct"].min())
    x_max_data = float(agg_sc["pos_pct"].max())
    y_min_data = float(agg_sc["margen_pct"].min())
    y_max_data = float(agg_sc["margen_pct"].max())
    x_range = max(10.0, x_max_data - x_min_data)
    y_range = max(5.0, y_max_data - y_min_data)
    x_margin = 0.20 * x_range
    y_margin = 0.20 * y_range
    xmin = x_min_data - x_margin
    xmax = x_max_data + x_margin
    ymin = y_min_data - y_margin
    ymax = y_max_data + y_margin

    fig_sc = go.Figure()
    INF = 1e9

    def add_quad(x0, x1, y0, y1, q):
        fig_sc.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=QUADRANT_COLOR[q], opacity=0.12, line=dict(width=0), layer="below")

    add_quad(x_ref, INF, y_ref, INF, 1)
    add_quad(-INF, x_ref, y_ref, INF, 2)
    add_quad(-INF, x_ref, -INF, y_ref, 3)
    add_quad(x_ref, INF, -INF, y_ref, 4)

    size_vals = np.sqrt(np.clip(agg_sc["venta_neta_level"].astype(float).values, 0, None))
    size_vals = 10 + 45 * (size_vals / np.nanmax(size_vals)) if np.nanmax(size_vals) > 0 else np.full_like(size_vals, 18.0)

    fig_sc.add_trace(
        go.Scatter(
            x=agg_sc["pos_pct"],
            y=agg_sc["margen_pct"],
            mode="markers+text",
            text=agg_sc["entity"],
            textposition="top center",
            marker=dict(size=size_vals, color=agg_sc["color"], opacity=0.8, line=dict(width=0.8, color="black")),
            customdata=np.stack(
                [
                    agg_sc["entity"].astype(str).values,
                    agg_sc["rol_rep"].astype(str).values,
                    agg_sc["quadrant_label"].astype(str).fillna("").values,
                    agg_sc["venta_neta_level"].astype(float).values,
                    agg_sc["pos_pct"].astype(float).values,
                    agg_sc["margen_pct"].astype(float).values,
                ],
                axis=-1,
            ),
            hovertemplate=(
                f"<b>{group_col}:</b> %{{customdata[0]}}<br>"
                "<b>Rol:</b> %{customdata[1]}<br>"
                "<b>Cuadrante:</b> %{customdata[2]}<br>"
                "Posicionamiento: <b>%{customdata[4]:.2f}%</b><br>"
                "Margen: <b>%{customdata[5]:.2f}%</b><br>"
                "Venta: <b>$%{customdata[3]:,.0f}</b><br>"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig_sc.add_vline(x=x_ref, line_width=1, line_dash="dash", opacity=0.8)
    fig_sc.add_hline(y=y_ref, line_width=1, line_dash="dash", opacity=0.8)

    fig_sc.add_annotation(x=(x_ref + xmax) / 2, y=(y_ref + ymax) / 2, text="Contribuyente", showarrow=False, opacity=1)
    fig_sc.add_annotation(x=(xmin + x_ref) / 2, y=(y_ref + ymax) / 2, text="Poderosa", showarrow=False, opacity=1)
    fig_sc.add_annotation(x=(xmin + x_ref) / 2, y=(ymin + y_ref) / 2, text="Magnética", showarrow=False, opacity=1)
    fig_sc.add_annotation(x=(x_ref + xmax) / 2, y=(ymin + y_ref) / 2, text="Oportunista", showarrow=False, opacity=1)

    fig_sc.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=600, xaxis_title="Posicionamiento (%)", yaxis_title="Margen (%)", xaxis=dict(range=[xmin, xmax]), yaxis=dict(range=[ymin, ymax]))
    st.plotly_chart(fig_sc, use_container_width=True)


# ---------------------------------------------------------------------------
# Tabla pivote: Macro > Categoría > SKU
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Tabla pivote — Posicionamiento + Foco (SKU)")
st.caption("Métricas calculadas en data_layer (SQL + limpieza + agregación correcta de ratios). Solo render UI aquí.")

if "_df_master" not in st.session_state or st.session_state["_df_master"].empty:
    st.warning("Sin datos para mostrar.")
    st.stop()

df_show = st.session_state["_df_master"].copy()

sum_cols = {"venta_neta", "peso_venta", "v0", "v1_low", "v1_mid", "v1_high", "delta_v_low", "delta_v_mid", "delta_v_high", "q0"}
wavg_cols = {"posicionamiento", "posicionamiento_rol", "margen", "margen_rol", "delta_pct", "delta_share", "season_si", "precio_cur", "precio_comp", "precio_rec", "delta_precio_pct", "delta_v_pct_low", "delta_v_pct_mid", "delta_v_pct_high"}
total_row = {}
vt = pd.to_numeric(df_show["venta_neta"], errors="coerce").sum()
for col in df_show.columns:
    if col in sum_cols:
        total_row[col] = pd.to_numeric(df_show[col], errors="coerce").sum()
    elif col in wavg_cols:
        total_row[col] = (pd.to_numeric(df_show[col], errors="coerce") * pd.to_numeric(df_show["venta_neta"], errors="coerce")).sum() / vt if vt else np.nan
    else:
        total_row[col] = "TOTAL"
# share_cur no es agregable globalmente
total_row["share_cur"] = None

with_totals = pd.concat([df_show, pd.DataFrame([total_row])], ignore_index=True)
df_grid = with_totals.where(pd.notna(with_totals), None)

if not AGGRID_AVAILABLE:
    st.dataframe(df_grid, use_container_width=True, height=650)
else:
    # Comentarios breves: agregadores personalizados para sumas/ponderados y estilos de semaforos.
    agg_sum = JsCode("""
    function(params) {
        if (!params.rowNode || !params.rowNode.allLeafChildren) return null;
        var s = 0;
        params.rowNode.allLeafChildren.forEach(function(c) {
            var v = c.data ? c.data[params.column.getColId()] : null;
            if (v != null && isFinite(Number(v))) s += Number(v);
        });
        return s;
    }
    """)
    agg_wavg = JsCode("""
    function(params) {
        if (!params.rowNode || !params.rowNode.allLeafChildren) return null;
        var sum = 0, wsum = 0;
        params.rowNode.allLeafChildren.forEach(function(c) {
            if (!c.data) return;
            var v = c.data[params.column.getColId()];
            var w = c.data['venta_neta'];
            if (v != null && w != null && isFinite(Number(v)) && isFinite(Number(w))) {
                sum += Number(v) * Number(w);
                wsum += Number(w);
            }
        });
        return wsum > 0 ? sum / wsum : null;
    }
    """)
    agg_share = JsCode("""
    function(params) {
        if (!params.rowNode) return null;
        var children = params.rowNode.childrenAfterGroup;
        if (children && children.length) {
            var total = 0;
            children.forEach(function(child){
                var val = null;
                if (child.aggData && child.aggData.hasOwnProperty(params.column.getColId())) {
                    val = child.aggData[params.column.getColId()];
                } else if (child.data) {
                    val = child.data[params.column.getColId()];
                }
                if (val != null && isFinite(Number(val))) total += Number(val);
            });
            return total;
        }
        if (!params.rowNode.allLeafChildren) return null;
        for (var i = 0; i < params.rowNode.allLeafChildren.length; i++) {
            var c = params.rowNode.allLeafChildren[i];
            var v = c.data ? c.data[params.column.getColId()] : null;
            if (v != null && isFinite(Number(v))) return Number(v);
        }
        return null;
    }
    """)

    style_pos = JsCode(f"""
    function(params) {{
        var v = params.value; if (v == null) return {{}}; v = Number(v);
        var minVal = {thr_pos['min']}; var midVal = {thr_pos['center']}; var maxVal = {thr_pos['max']};
        function clamp(x,a,b){{return Math.max(a, Math.min(b, x));}}
        function lerp(a,b,t){{return a + (b - a) * t;}}
        var red = [248,105,107], green = [99,190,123], white=[255,255,255], c;
        if (v < midVal) {{ var t = clamp((midVal - v)/(midVal - minVal),0,1); c=[lerp(white[0],green[0],t),lerp(white[1],green[1],t),lerp(white[2],green[2],t)]; }}
        else if (v > midVal) {{ var t = clamp((v - midVal)/(maxVal - midVal),0,1); c=[lerp(white[0],red[0],t),lerp(white[1],red[1],t),lerp(white[2],red[2],t)]; }}
        else c=white;
        return {{'backgroundColor': 'rgb(' + c.map(Math.round).join(',') + ')', 'color': 'black'}};
    }}
    """)

    style_delta = JsCode(f"""
    function(params) {{
        var v = params.value; if (v == null) return {{}}; v = Number(v);
        var minVal = {thr_delta_pct['min']}; var midVal = {thr_delta_pct['center']}; var maxVal = {thr_delta_pct['max']};
        function clamp(x,a,b){{return Math.max(a, Math.min(b, x));}}
        function lerp(a,b,t){{return a + (b - a) * t;}}
        var red = [248,105,107], green = [99,190,123], white=[255,255,255], c;
        if (v < midVal) {{ var t = clamp((midVal - v)/(midVal - minVal),0,1); c=[lerp(white[0],red[0],t),lerp(white[1],red[1],t),lerp(white[2],red[2],t)]; }}
        else if (v > midVal) {{ var t = clamp((v - midVal)/(maxVal - midVal),0,1); c=[lerp(white[0],green[0],t),lerp(white[1],green[1],t),lerp(white[2],green[2],t)]; }}
        else c=white;
        return {{'backgroundColor': 'rgb(' + c.map(Math.round).join(',') + ')', 'color': 'black'}};
    }}
    """)
    style_delta_share = JsCode(f"""
    function(params) {{
        var v = params.value; if (v == null) return {{}}; v = Number(v);
        var minVal = {thr_delta_share['min']}; var midVal = {thr_delta_share['center']}; var maxVal = {thr_delta_share['max']};
        function clamp(x,a,b){{return Math.max(a, Math.min(b, x));}}
        function lerp(a,b,t){{return a + (b - a) * t;}}
        var red=[248,105,107], green=[99,190,123], white=[255,255,255], c;
        if (v < midVal) {{ var t = clamp((midVal - v)/(midVal - minVal),0,1); c=[lerp(white[0],red[0],t),lerp(white[1],red[1],t),lerp(white[2],red[2],t)]; }}
        else if (v > midVal) {{ var t = clamp((v - midVal)/(maxVal - midVal),0,1); c=[lerp(white[0],green[0],t),lerp(white[1],green[1],t),lerp(white[2],green[2],t)]; }}
        else c=white;
        return {{'backgroundColor': 'rgb(' + c.map(Math.round).join(',') + ')', 'color': 'black'}};
    }}
    """)

    fmt_pct = "value == null ? '' : (Number(value) * 100).toFixed(1) + '%'"
    fmt_pct2 = "value == null ? '' : (Number(value) * 100).toFixed(2) + '%'"
    fmt_money = "value == null ? '' : '$' + Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0})"
    fmt_money0 = "value == null ? '' : '$' + Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0})"
    fmt_units = "value == null ? '' : Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0})"
    fmt_season = "value == null ? '' : ((Number(value) - 1) * 100).toFixed(0) + '%'"

    gb = GridOptionsBuilder.from_dataframe(df_grid)
    gb.configure_default_column(resizable=True, filter=True, sortable=True, autoSizeColumns=True)
    gb.configure_column("macro_categoria", headerName="Macro", rowGroup=True, hide=True)
    gb.configure_column("categoria", headerName="Categoría", rowGroup=True, hide=True)
    gb.configure_column("nombre_sku", headerName="SKU (Nombre)", rowGroup=True, hide=True)

    gb.configure_column("rol_rep", headerName="Rol", aggFunc="first", minWidth=100)
    gb.configure_column("bucket_cat", headerName="Segmento", aggFunc="first", minWidth=110)
    gb.configure_column("venta_neta", headerName="Venta Chiper", type=["numericColumn"], aggFunc=agg_sum, valueFormatter=fmt_money, minWidth=140)
    gb.configure_column("peso_venta", headerName="Peso Venta", type=["numericColumn"], aggFunc=agg_sum, valueFormatter=fmt_pct2, minWidth=120)
    gb.configure_column("posicionamiento", headerName="Posición", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, cellStyle=style_pos, minWidth=120)
    gb.configure_column("posicionamiento_rol", headerName="Pos. Objetivo", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, cellStyle=style_pos, minWidth=140)
    gb.configure_column("margen", headerName="Margen", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, minWidth=110)
    gb.configure_column("margen_rol", headerName="Margen Obj.", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, minWidth=140)
    gb.configure_column("precio_cur", headerName="Precio Actual", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_money0, minWidth=130)
    gb.configure_column("precio_comp", headerName="Precio Comp.", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_money0, minWidth=130)
    gb.configure_column("precio_rec", headerName="Precio Recomendado", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_money0, minWidth=150)
    gb.configure_column("delta_precio_pct", headerName="Δ Precio", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct2, minWidth=110)
    gb.configure_column("share_cur", headerName="Share Actual", type=["numericColumn"], aggFunc=agg_share, valueFormatter=fmt_pct2, minWidth=140)
    gb.configure_column("delta_share", headerName="Δ Share", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct2, cellStyle=style_delta_share, minWidth=120)
    gb.configure_column("delta_pct", headerName="Δ Venta", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, cellStyle=style_delta, minWidth=120)
    gb.configure_column("season_si", headerName="Estacionalidad", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_season, minWidth=100)
    gb.configure_column("q0", headerName="Unidades Base", type=["numericColumn"], aggFunc=agg_sum, valueFormatter=fmt_units, minWidth=120)
    gb.configure_column("v0", headerName="Venta Base", type=["numericColumn"], aggFunc=agg_sum, valueFormatter=fmt_money, minWidth=120)
    gb.configure_column("v1_mid", headerName="Venta Proy. (mid)", type=["numericColumn"], aggFunc=agg_sum, valueFormatter=fmt_money, minWidth=150)
    gb.configure_column("delta_v_mid", headerName="Δ Venta (mid)", type=["numericColumn"], aggFunc=agg_sum, valueFormatter=fmt_money, minWidth=140)
    gb.configure_column("delta_v_pct_mid", headerName="Δ Venta % (mid)", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct2, minWidth=140)
    gb.configure_column("proveedor", headerName="Proveedor", minWidth=110)
    gb.configure_column("sku", headerName="SKU (ID)", aggFunc="first", minWidth=110)

    gb.configure_grid_options(groupIncludeFooter=True, groupIncludeTotalFooter=True, animateRows=True, suppressAggFuncInHeader=True, autoSizePadding=5)
    grid_opts = gb.build()
    grid_opts["groupDefaultExpanded"] = 0
    grid_opts["autoGroupColumnDef"] = {"headerName": "Macro → Categoría → SKU", "minWidth": 320, "cellRendererParams": {"suppressCount": True}, "pinned": "left"}

    AgGrid(
        df_grid,
        gridOptions=grid_opts,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        update_mode="NO_UPDATE",
        height=520,
        key="master_pricing_grid_new",
    )

st.download_button(
    "Descargar CSV (tabla unificada)",
    data=df_show.to_csv(index=False).encode("utf-8"),
    file_name="posicionamiento_foco_unificado.csv",
    mime="text/csv",
)


# ---------------------------------------------------------------------------
# Sección de oportunidades (misma lógica, optimizada)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### Detección de oportunidades — lectura rápida y accionables")

ctrl_opp_1, ctrl_opp_2, ctrl_opp_3 = st.columns(3)
with ctrl_opp_1:
    agresividad = st.slider("Agresividad (%)", min_value=0, max_value=100, value=60, step=5)
with ctrl_opp_2:
    modo_pesos = st.selectbox("Config. Pesos", options=["Manual", "Preset: Balanceado", "Preset: Crecimiento", "Preset: Rentabilidad", "Preset: Defensivo"], index=0)
with ctrl_opp_3:
    filtro_prioridad = st.multiselect("Filtrar prioridad", options=["P0", "P1", "P2"], default=["P0", "P1", "P2"])

with st.expander("Parámetros de scoring (pesos y umbrales)", expanded=False):
    presets_pesos = {
        "Manual": (0.30, 0.40, 0.30),
        "Preset: Balanceado": (0.30, 0.40, 0.30),
        "Preset: Crecimiento": (0.50, 0.25, 0.25),
        "Preset: Rentabilidad": (0.20, 0.60, 0.20),
        "Preset: Defensivo": (0.30, 0.50, 0.20),
    }
    preset_vals = presets_pesos.get(modo_pesos, (0.30, 0.40, 0.30))
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        w_market_raw = st.slider("Peso mercado", 0.0, 1.0, float(preset_vals[0]), 0.05)
    with col_w2:
        w_gap_raw = st.slider("Peso gap", 0.0, 1.0, float(preset_vals[1]), 0.05)
    with col_w3:
        w_impact_raw = st.slider("Peso impacto", 0.0, 1.0, float(preset_vals[2]), 0.05)
    total_w = w_market_raw + w_gap_raw + w_impact_raw
    w_market = w_market_raw / total_w if total_w else 0.33
    w_gap = w_gap_raw / total_w if total_w else 0.33
    w_impact = w_impact_raw / total_w if total_w else 0.34

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        tau_pos = st.number_input("τ posición", value=0.06, step=0.01, format="%.3f")
        tau_margen = st.number_input("τ margen (pp)", value=0.015, step=0.005, format="%.3f")
    with col_t2:
        tau_delta_share = st.number_input("τ Δshare (pp)", value=0.002, step=0.001, format="%.4f")
        tau_delta_pct = st.number_input("τ Δventa", value=0.03, step=0.01, format="%.3f")
    with col_t3:
        tau_peso_venta = st.number_input("τ peso (BIG)", value=0.01, step=0.005, format="%.3f")
        tau_season = st.number_input("τ estacionalidad", value=0.10, step=0.05, format="%.2f")
    with col_t4:
        conf_min = st.number_input("Confianza mín", value=0.55, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        penal_low = st.number_input("Penaliz. low conf", value=0.20, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")

    allow_qualitative = st.checkbox("Permitir palancas cualitativas (P2/P5)", value=True)

solo_big_weight = st.checkbox("Solo BIG_WEIGHT (peso >= umbral)", value=False)

# ---------------------------
# Construcción de oportunidades a nivel SKU (luego se agregan en grilla)
# ---------------------------

def _agg_level(df_base: pd.DataFrame, level: str) -> pd.DataFrame:
    d = df_base.copy()
    num_cols = ["venta_neta", "peso_venta", "posicionamiento", "posicionamiento_rol", "margen", "margen_rol", "share_cur", "delta_share", "delta_pct", "season_si"]
    for col in num_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    if level == "SKU":
        return d
    if level == "Categoría":
        rows = []
        for (macro, cat), grp in d.groupby(["macro_categoria", "categoria"], dropna=False):
            vt = grp["venta_neta"].sum()
            if vt > 0:
                pos_pond = (grp["posicionamiento"] * grp["venta_neta"]).sum() / vt
                pos_rol_pond = (grp["posicionamiento_rol"] * grp["venta_neta"]).sum() / vt
                mg_pond = (grp["margen"] * grp["venta_neta"]).sum() / vt
                mg_rol_pond = (grp["margen_rol"] * grp["venta_neta"]).sum() / vt
            else:
                pos_pond = pos_rol_pond = mg_pond = mg_rol_pond = np.nan
            rows.append({
                "macro_categoria": macro,
                "categoria": cat,
                "nombre_sku": "—",
                "proveedor": "—",
                "rol_rep": grp["rol_rep"].mode()[0] if len(grp["rol_rep"].mode()) else grp["rol_rep"].iloc[0],
                "bucket_cat": grp["bucket_cat"].mode()[0] if len(grp["bucket_cat"].mode()) else grp["bucket_cat"].iloc[0],
                "venta_neta": vt,
                "peso_venta": grp["peso_venta"].sum(),
                "posicionamiento": pos_pond,
                "posicionamiento_rol": pos_rol_pond,
                "margen": mg_pond,
                "margen_rol": mg_rol_pond,
                "share_cur": grp["share_cur"].iloc[0] if "share_cur" in grp.columns else np.nan,
                "delta_share": grp["delta_share"].iloc[0] if "delta_share" in grp.columns else np.nan,
                "delta_pct": grp["delta_pct"].iloc[0] if "delta_pct" in grp.columns else np.nan,
                "season_si": grp["season_si"].iloc[0] if "season_si" in grp.columns else np.nan,
            })
        return pd.DataFrame(rows)
    if level == "Macro":
        rows = []
        for macro, grp in d.groupby("macro_categoria", dropna=False):
            vt = grp["venta_neta"].sum()
            if vt > 0:
                pos_pond = (grp["posicionamiento"] * grp["venta_neta"]).sum() / vt
                pos_rol_pond = (grp["posicionamiento_rol"] * grp["venta_neta"]).sum() / vt
                mg_pond = (grp["margen"] * grp["venta_neta"]).sum() / vt
                mg_rol_pond = (grp["margen_rol"] * grp["venta_neta"]).sum() / vt
            else:
                pos_pond = pos_rol_pond = mg_pond = mg_rol_pond = np.nan
            share_total = grp.groupby("categoria", dropna=False)["share_cur"].max().sum() if "share_cur" in grp.columns else np.nan
            delta_s = (grp["delta_share"] * grp["share_cur"]).sum() / grp["share_cur"].sum() if "share_cur" in grp.columns and grp["share_cur"].sum() > 0 else grp.get("delta_share", pd.Series()).mean()
            delta_p = (grp["delta_pct"] * grp["share_cur"]).sum() / grp["share_cur"].sum() if "share_cur" in grp.columns and grp["share_cur"].sum() > 0 else grp.get("delta_pct", pd.Series()).mean()
            season_w = (grp["season_si"] * grp["share_cur"]).sum() / grp["share_cur"].sum() if "share_cur" in grp.columns and grp["share_cur"].sum() > 0 else grp.get("season_si", pd.Series()).mean()
            rows.append({
                "macro_categoria": macro,
                "categoria": "—",
                "nombre_sku": "—",
                "proveedor": "—",
                "rol_rep": grp["rol_rep"].mode()[0] if len(grp["rol_rep"].mode()) else grp["rol_rep"].iloc[0],
                "bucket_cat": grp["bucket_cat"].mode()[0] if len(grp["bucket_cat"].mode()) else grp["bucket_cat"].iloc[0],
                "venta_neta": vt,
                "peso_venta": grp["peso_venta"].sum(),
                "posicionamiento": pos_pond,
                "posicionamiento_rol": pos_rol_pond,
                "margen": mg_pond,
                "margen_rol": mg_rol_pond,
                "share_cur": share_total,
                "delta_share": delta_s,
                "delta_pct": delta_p,
                "season_si": season_w,
            })
        return pd.DataFrame(rows)
    return d


# ---------------------------
# Construcción de oportunidades a nivel SKU (luego se agregan en grilla)
# ---------------------------

def _score_level(df_agg: pd.DataFrame) -> pd.DataFrame:
    if df_agg.empty:
        return df_agg

    df_agg = df_agg.copy()
    required_cols = [
        "share_cur",
        "delta_share",
        "delta_pct",
        "posicionamiento",
        "posicionamiento_rol",
        "margen",
        "margen_rol",
        "season_si",
        "venta_neta",
        "peso_venta",
    ]
    for col in required_cols:
        if col not in df_agg.columns:
            df_agg[col] = np.nan

    # A) Normalización por columna (ratio vs %): decide por mediana/p95, no por fila.
    def _norm_col(series: pd.Series, is_ratio: bool) -> pd.Series:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(arr)
        if not ok.any():
            return pd.Series(arr, index=series.index)
        med = np.nanmedian(arr)
        p95 = np.nanpercentile(arr, 95)
        if is_ratio:
            return pd.Series(arr / 100.0, index=series.index) if (med > 3 or p95 > 3) else pd.Series(arr, index=series.index)
        return pd.Series(arr / 100.0, index=series.index) if (med > 1 or p95 > 1) else pd.Series(arr, index=series.index)

    if "posicionamiento" in df_agg.columns:
        df_agg["posicionamiento"] = _norm_col(df_agg["posicionamiento"], is_ratio=True)
    if "posicionamiento_rol" in df_agg.columns:
        df_agg["posicionamiento_rol"] = _norm_col(df_agg["posicionamiento_rol"], is_ratio=True)
    if "margen" in df_agg.columns:
        df_agg["margen"] = _norm_col(df_agg["margen"], is_ratio=False)
    if "margen_rol" in df_agg.columns:
        df_agg["margen_rol"] = _norm_col(df_agg["margen_rol"], is_ratio=False)

    df_agg["gap_pos"] = df_agg["posicionamiento"] - df_agg["posicionamiento_rol"]
    df_agg["gap_margen"] = df_agg["margen"] - df_agg["margen_rol"]

    # B) Confianza y gating duro: faltantes bajan score, conf<conf_min => P2 + MONITOREAR salvo venta muy alta.
    missing_market = df_agg[["share_cur", "delta_share", "delta_pct"]].isna().sum(axis=1)
    missing_gap = df_agg[["posicionamiento", "posicionamiento_rol", "margen", "margen_rol"]].isna().sum(axis=1)
    conf = np.ones(len(df_agg))
    conf *= np.where(missing_market > 0, 0.45, 1.0)
    conf *= np.where(missing_gap > 0, 0.55, 1.0)
    conf *= np.where(pd.to_numeric(df_agg["venta_neta"], errors="coerce") <= 0, 0.0, 1.0)
    df_agg["conf"] = np.clip(conf, 0, 1)

    # Señales booleanas
    df_agg["BIG_WEIGHT"] = df_agg["peso_venta"] >= tau_peso_venta
    df_agg["POS_TOO_EXP"] = df_agg["gap_pos"] >= tau_pos
    df_agg["POS_TOO_CHEAP"] = df_agg["gap_pos"] <= -tau_pos
    df_agg["MARGIN_LOW"] = df_agg["gap_margen"] <= -tau_margen
    df_agg["MARGIN_HIGH"] = df_agg["gap_margen"] >= tau_margen
    df_agg["MKT_GROW"] = df_agg["delta_pct"] >= tau_delta_pct
    df_agg["MKT_DECLINE"] = df_agg["delta_pct"] <= -tau_delta_pct
    df_agg["MKT_MIX_UP"] = df_agg["delta_share"] >= tau_delta_share
    df_agg["MKT_MIX_DOWN"] = df_agg["delta_share"] <= -tau_delta_share
    df_agg["SEASON_HIGH"] = df_agg["season_si"] >= (1.0 + tau_season)
    df_agg["SEASON_LOW"] = df_agg["season_si"] <= (1.0 - tau_season)

    eps = 1e-6
    p90_venta = df_agg["venta_neta"].quantile(0.90) if df_agg["venta_neta"].notna().any() else 0.0
    high_sale = p90_venta * 1.2 if p90_venta > 0 else 0.0

    # C) Estrategia tamaño: market solo momentum (delta_share, delta_pct, season); impacto usa venta_neta (sin share_cur/peso doble).
    delta_share_z = np.tanh(df_agg["delta_share"].fillna(0) / (tau_delta_share + eps))
    delta_pct_z = np.tanh(df_agg["delta_pct"].fillna(0) / (tau_delta_pct + eps))
    season_term = np.clip((df_agg["season_si"].fillna(1.0) - 1.0) / 0.5, -1, 1)
    market_raw = (0.45 * delta_share_z + 0.45 * delta_pct_z + 0.10 * season_term) * 100
    market_raw *= np.where(missing_market > 0, 0.35, 1.0)  # gating en score_market

    # D) Gap asimétrico: más peso si pos caro con mercado favorable, o pos barato con margen alto/mercado favorable.
    gpos_scaled = np.clip(df_agg["gap_pos"] / (tau_pos + eps), -3, 3)
    gmg_scaled = np.clip(df_agg["gap_margen"] / (tau_margen + eps), -3, 3)
    favor_exp = df_agg["POS_TOO_EXP"] & (df_agg["MKT_GROW"] | df_agg["MKT_MIX_UP"])
    favor_cheap = df_agg["POS_TOO_CHEAP"] & (df_agg["MARGIN_HIGH"] | df_agg["MKT_GROW"] | df_agg["MKT_MIX_UP"])
    asym_factor = 1.0 + 0.25 * favor_exp.astype(float) + 0.20 * favor_cheap.astype(float)
    gap_term = (0.6 * np.abs(gpos_scaled) * asym_factor + 0.4 * np.abs(gmg_scaled)) * 100
    gap_term *= np.where(missing_gap > 0, 0.40, 1.0)  # gating en score_gap

    impact_term = np.sqrt(np.clip(df_agg["venta_neta"] / (p90_venta + eps), 0, 4)) * 100

    df_agg["score_market"] = market_raw.clip(0, 100)
    df_agg["score_gap"] = gap_term.clip(0, 120)
    df_agg["score_impact"] = impact_term.clip(0, 140)

    df_agg["score_raw"] = w_market * df_agg["score_market"] + w_gap * df_agg["score_gap"] + w_impact * df_agg["score_impact"]
    df_agg["score_raw"] *= (0.85 + 0.30 * (agresividad / 100.0))
    df_agg["score_final_pre"] = df_agg["score_raw"] * (0.55 + 0.45 * df_agg["conf"])

    # E) Palancas (P3 siempre HOLD): margen bajo + pos barato => TERMS/PROMO; sin P3 con UP/DOWN.
    def assign_palanca(row):
        pid, dirn = "P1", "HOLD"
        if row["MARGIN_LOW"] and row["POS_TOO_CHEAP"]:
            pid = "P3" if abs(row["gap_margen"]) >= 1.5 * tau_margen else "P2"
            dirn = "HOLD"
        elif row["MARGIN_LOW"] and row["POS_TOO_EXP"]:
            pid = "P3"; dirn = "HOLD"
        elif row["POS_TOO_EXP"]:
            pid = "P1"; dirn = "DOWN"
        elif row["POS_TOO_CHEAP"]:
            pid = "P1"; dirn = "UP"
        elif row["MKT_DECLINE"] and allow_qualitative:
            pid = "P2"; dirn = "HOLD"
        elif row["MKT_MIX_DOWN"] and allow_qualitative:
            pid = "P5"; dirn = "HOLD"
        return pid, dirn

    df_agg[["palanca_id", "accion_direccion"]] = df_agg.apply(lambda r: pd.Series(assign_palanca(r)), axis=1)
    palanca_map = {"P1": "PRICE", "P2": "PROMO", "P3": "TERMS", "P4": "AVAIL", "P5": "ACTIVATION"}
    df_agg["accion_palanca"] = df_agg["palanca_id"].map(palanca_map)
    df_agg["palanca_confianza"] = np.clip(df_agg["conf"], 0, 1)

    # F) Score final interpretable 0–100 y penalización low_conf
    df_agg["score_final"] = (df_agg["score_final_pre"] * 0.7 + df_agg["score_final_pre"] * df_agg["palanca_confianza"] * 0.3)
    df_agg["score_final"] = df_agg["score_final"].clip(0, 100)

    def classify_opp(row):
        if row["conf"] < conf_min and row["venta_neta"] < high_sale:
            return "MONITOREAR"
        pid = row["palanca_id"]
        if pid == "P3" and row["MARGIN_LOW"]:
            return "RECUPERAR MARGEN"
        if pid == "P1" and row["accion_direccion"] == "DOWN" and (row["MKT_GROW"] or row["MKT_MIX_UP"]):
            return "CAPTURAR CRECIMIENTO"
        if pid == "P1" and row["accion_direccion"] == "DOWN" and (row["MKT_DECLINE"] or row["MKT_MIX_DOWN"]):
            return "DEFENDER"
        if pid == "P1" and row["accion_direccion"] == "UP" and (row["MARGIN_HIGH"] or row["MKT_GROW"] or row["MKT_MIX_UP"]):
            return "CAPTURAR VALOR"
        if pid == "P1" and abs(row["gap_pos"]) >= tau_pos * 1.5:
            return "CORREGIR POSICIONAMIENTO"
        if pid == "P4":
            return "CORREGIR DISPONIBILIDAD"
        if pid == "P5":
            return "ACTIVAR VISIBILIDAD"
        return "OPTIMIZAR"

    df_agg["tipo_oportunidad"] = df_agg.apply(classify_opp, axis=1)

    def get_rationale(row):
        reasons = []
        if row["MKT_GROW"] or row["MKT_MIX_UP"]:
            reasons.append("Mkt↑")
        elif row["MKT_DECLINE"] or row["MKT_MIX_DOWN"]:
            reasons.append("Mkt↓")
        if row["POS_TOO_EXP"]:
            reasons.append("GapPos↑")
        elif row["POS_TOO_CHEAP"]:
            reasons.append("GapPos↓")
        if row["MARGIN_LOW"]:
            reasons.append("GapMargen↓")
        if row["MARGIN_HIGH"]:
            reasons.append("Margen↑")
        if row["BIG_WEIGHT"]:
            reasons.append("Peso alto")
        return " + ".join(reasons[:4]) if reasons else "—"

    df_agg["accion_rationale"] = df_agg.apply(get_rationale, axis=1)

    def get_signals(row):
        signals = []
        for sig in ["MKT_MIX_UP", "MKT_MIX_DOWN", "MKT_GROW", "MKT_DECLINE", "POS_TOO_EXP", "POS_TOO_CHEAP", "MARGIN_LOW", "MARGIN_HIGH", "BIG_WEIGHT", "SEASON_HIGH", "SEASON_LOW"]:
            if row.get(sig, False):
                signals.append(sig)
        return " + ".join(signals[:4]) if signals else "—"

    df_agg["señales"] = df_agg.apply(get_signals, axis=1)

    def get_prioridad(row):
        if row["conf"] < conf_min and row["venta_neta"] < high_sale:
            return "P2"
        if row["score_final"] >= 75 and row["conf"] >= conf_min:
            return "P0"
        if row["score_final"] >= 55:
            return "P1"
        return "P2"

    df_agg["prioridad"] = df_agg.apply(get_prioridad, axis=1)

    # Gating final de seguridad
    low_conf_mask = (df_agg["conf"] < conf_min) & (df_agg["venta_neta"] < high_sale)
    df_agg.loc[low_conf_mask, "tipo_oportunidad"] = "MONITOREAR"
    df_agg.loc[low_conf_mask, "prioridad"] = "P2"
    df_agg.loc[low_conf_mask, "score_final"] *= 0.6

    if solo_big_weight:
        df_agg = df_agg[df_agg["peso_venta"] >= tau_peso_venta]

    cols = [
        "score_final", "prioridad", "tipo_oportunidad", "señales", "accion_palanca", "accion_direccion", "accion_rationale",
        "rol_rep", "bucket_cat", "venta_neta", "peso_venta", "posicionamiento", "posicionamiento_rol", "gap_pos",
        "margen", "margen_rol", "gap_margen", "share_cur", "delta_share", "delta_pct", "season_si", "conf",
        "macro_categoria", "categoria", "nombre_sku", "proveedor", "palanca_id",
    ]
    return df_agg[cols].copy()


# Dataframe base de oportunidades: siempre SKU, agregación se hace en la grilla
opp_sku = _score_level(df_show)

# Render único con agrupación Macro → Categoría → SKU
if opp_sku.empty:
    st.info("No hay oportunidades con los filtros seleccionados.")
else:
    df_opps = opp_sku.copy()
    if filtro_prioridad:
        df_opps = df_opps[df_opps["prioridad"].isin(filtro_prioridad)]
    if df_opps.empty:
        st.info("No hay oportunidades con los filtros seleccionados.")
    elif not AGGRID_AVAILABLE:
        st.dataframe(df_opps, use_container_width=True, height=520)
        st.download_button(
            "📥 Descargar CSV",
            data=df_opps.to_csv(index=False).encode("utf-8"),
            file_name="oportunidades_pricing.csv",
            mime="text/csv",
            key="download_opps_unico",
        )
    else:
        df_opps_grid = df_opps.where(pd.notna(df_opps), None)

        # Agregadores: sumas, ponderados, priorización y señales
        agg_sum = JsCode("""
        function(params) {
            if (!params.rowNode || !params.rowNode.allLeafChildren) return null;
            var s = 0;
            params.rowNode.allLeafChildren.forEach(function(c) {
                var v = c.data ? c.data[params.column.getColId()] : null;
                if (v != null && isFinite(Number(v))) s += Number(v);
            });
            return s;
        }
        """)
        agg_wavg = JsCode("""
        function(params) {
            if (!params.rowNode || !params.rowNode.allLeafChildren) return null;
            var sum = 0, wsum = 0;
            params.rowNode.allLeafChildren.forEach(function(c) {
                if (!c.data) return;
                var v = c.data[params.column.getColId()];
                var w = c.data['venta_neta'];
                if (v != null && w != null && isFinite(Number(v)) && isFinite(Number(w))) {
                    sum += Number(v) * Number(w);
                    wsum += Number(w);
                }
            });
            return wsum > 0 ? sum / wsum : null;
        }
        """)
        agg_priority = JsCode("""
        function(params){
            if (!params.values) return null;
            var order = {'P0':0,'P1':1,'P2':2};
            var best = null;
            params.values.forEach(function(v){
                if (v == null) return;
                if (best == null || (order[v] ?? 99) < (order[best] ?? 99)) best = v;
            });
            return best;
        }
        """)
        agg_tipo = JsCode("""
        function(params){
            if (!params.rowNode || !params.rowNode.allLeafChildren) return null;
            var cnt = {};
            params.rowNode.allLeafChildren.forEach(function(c){
                if (!c.data) return;
                var t = c.data['tipo_oportunidad'];
                var w = c.data['venta_neta'];
                if (t == null || w == null || !isFinite(Number(w))) return;
                cnt[t] = (cnt[t] || 0) + Number(w);
            });
            var best=null, bestW=-1;
            for (var k in cnt){ if (cnt[k] > bestW){ best=k; bestW=cnt[k]; } }
            return best;
        }
        """)
        agg_signals = JsCode("""
        function(params){
            if (!params.rowNode || !params.rowNode.allLeafChildren) return null;
            var seen = {};
            var out = [];
            params.rowNode.allLeafChildren.forEach(function(c){
                if (!c.data) return;
                var v = c.data['señales'];
                if (!v) return;
                v.split(' + ').forEach(function(s){
                    if (!s) return;
                    if (!seen[s]) { seen[s]=true; out.push(s); }
                });
            });
            return out.slice(0,4).join(' + ');
        }
        """)
        agg_mode = JsCode("""
        function(params){
            if (!params.rowNode || !params.rowNode.allLeafChildren) return null;
            var cnt = {};
            params.rowNode.allLeafChildren.forEach(function(c){
                if (!c.data) return;
                var v = c.data[params.column.getColId()];
                if (v == null) return;
                cnt[v] = (cnt[v] || 0) + 1;
            });
            var best=null, bestC=-1;
            for (var k in cnt){ if (cnt[k] > bestC){ best=k; bestC=cnt[k]; } }
            return best;
        }
        """)

        status_columns = ["tipo_oportunidad", "accion_palanca", "rol_rep", "prioridad"]
        status_scales = {}
        for col in status_columns:
            uniques = list(dict.fromkeys(df_opps[col].dropna().astype(str)))
            denom = max(len(uniques) - 1, 1)
            status_scales[col] = {val: idx / denom for idx, val in enumerate(uniques)}
        status_style = JsCode(f"""
        function(params) {{
            var v = params.value;
            if (v == null) return {{}};
            var scales = {json.dumps(status_scales, ensure_ascii=False)};
            var col = params.colDef.field;
            var map = scales[col] || {{}};
            var key = String(v);
            var score = map.hasOwnProperty(key) ? map[key] : null;
            if (score == null) return {{}};
            var r = Math.round(255 - 120 * score);
            var g = Math.round(140 + 120 * score);
            var b = 100;
            return {{'backgroundColor': 'rgba(' + r + ',' + g + ',' + b + ',0.45)', 'color': 'black'}};
        }}
        """)

        style_score = JsCode("""
        function(params) {
            var v = params.value;
            if (v == null) return {};
            v = Number(v);
            var color;
            if (v >= 75) color = 'rgba(39, 174, 96, 0.3)';
            else if (v >= 55) color = 'rgba(246, 227, 122, 0.3)';
            else color = 'rgba(248, 105, 107, 0.3)';
            return {'backgroundColor': color, 'color': 'black', 'fontWeight': 'bold'};
        }
        """)
        fmt_pct = "value == null ? '' : (Number(value) * 100).toFixed(1) + '%';"
        fmt_pct2 = "value == null ? '' : (Number(value) * 100).toFixed(2) + '%';"
        fmt_money = "value == null ? '' : '$' + Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0});"
        fmt_money0 = "value == null ? '' : '$' + Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0});"
        fmt_units = "value == null ? '' : Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0});"
        fmt_season = "value == null ? '' : ((Number(value) - 1) * 100).toFixed(0) + '%'"

        gb_opps = GridOptionsBuilder.from_dataframe(df_opps_grid)
        gb_opps.configure_default_column(resizable=True, filter=True, sortable=True)

        gb_opps.configure_column("macro_categoria", rowGroup=True, hide=True)
        gb_opps.configure_column("categoria", rowGroup=True, hide=True)
        gb_opps.configure_column("nombre_sku", hide=True)

        gb_opps.configure_column("score_final", headerName="Puntuación", type=["numericColumn"], aggFunc=agg_wavg, cellStyle=style_score, valueFormatter="value == null ? '' : Number(value).toFixed(1)", minWidth=90)
        gb_opps.configure_column("prioridad", headerName="Prioridad", aggFunc=agg_priority, cellStyle=status_style, minWidth=90)
        gb_opps.configure_column("tipo_oportunidad", headerName="Tipo de Oportunidad", aggFunc=agg_tipo, cellStyle=status_style, minWidth=160)
        gb_opps.configure_column("señales", headerName="Señales", aggFunc=agg_signals, minWidth=240)
        gb_opps.configure_column("accion_palanca", headerName="Palanca", aggFunc=agg_mode, cellStyle=status_style, minWidth=100)
        gb_opps.configure_column("accion_direccion", headerName="Dirección", aggFunc=agg_mode, minWidth=90)
        gb_opps.configure_column("accion_rationale", headerName="Razón", aggFunc=agg_mode, minWidth=220)
        gb_opps.configure_column("rol_rep", headerName="Rol", aggFunc=agg_mode, cellStyle=status_style, minWidth=110)
        gb_opps.configure_column("bucket_cat", headerName="Segmento", aggFunc=agg_mode, minWidth=100)
        gb_opps.configure_column("venta_neta", headerName="Venta Neta", type=["numericColumn"], aggFunc=agg_sum, valueFormatter=fmt_money, minWidth=120)
        gb_opps.configure_column("peso_venta", headerName="Peso Venta", type=["numericColumn"], aggFunc=agg_sum, valueFormatter=fmt_pct2, minWidth=100)
        gb_opps.configure_column("posicionamiento", headerName="Posicionamiento", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, minWidth=90)
        gb_opps.configure_column("posicionamiento_rol", headerName="Pos. Objetivo", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, minWidth=90)
        gb_opps.configure_column("gap_pos", headerName="Gap Pos.", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, minWidth=90)
        gb_opps.configure_column("margen", headerName="Margen", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, minWidth=90)
        gb_opps.configure_column("margen_rol", headerName="Margen Obj.", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, minWidth=90)
        gb_opps.configure_column("precio_cur", headerName="Precio Actual", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_money0, minWidth=130)
        gb_opps.configure_column("precio_comp", headerName="Precio Comp.", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_money0, minWidth=130)
        gb_opps.configure_column("precio_rec", headerName="Precio Recomendado", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_money0, minWidth=150)
        gb_opps.configure_column("delta_precio_pct", headerName="Δ Precio", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct2, minWidth=110)
        gb_opps.configure_column("share_cur", headerName="Share Actual", type=["numericColumn"], aggFunc=agg_sum, valueFormatter=fmt_pct2, minWidth=90)
        gb_opps.configure_column("delta_share", headerName="Δ Share", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct2, minWidth=90)
        gb_opps.configure_column("delta_pct", headerName="Δ Venta", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_pct, minWidth=90)
        gb_opps.configure_column("season_si", headerName="Estacionalidad", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter=fmt_season, minWidth=90)
        gb_opps.configure_column("conf", headerName="Confianza", type=["numericColumn"], aggFunc=agg_wavg, valueFormatter="value == null ? '' : Number(value).toFixed(2)", minWidth=70)

        gb_opps.configure_grid_options(suppressAggFuncInHeader=True)
        grid_opts_opps = gb_opps.build()
        grid_opts_opps["groupDefaultExpanded"] = 0
        grid_opts_opps["groupDisplayType"] = "singleColumn"
        grid_opts_opps["autoGroupColumnDef"] = {
            "headerName": "Macro → Categoría → SKU",
            "minWidth": 320,
            "cellRendererParams": {
                "suppressCount": True,
                "innerRenderer": JsCode("""
                class SKUNameRenderer {
                    init(params) {
                        this.params = params;
                        this.eGui = document.createElement('span');
                        const data = params.data || {};
                        const skuName = data['nombre_sku'] || data['sku'] || '(SKU)';
                        if (params.node.leaf) {
                            this.eGui.textContent = skuName;
                        } else {
                            this.eGui.textContent = params.value || skuName;
                        }
                    }
                    getGui() { return this.eGui; }
                }
                """),
            },
            "pinned": "left",
        }

        AgGrid(
            df_opps_grid,
            gridOptions=grid_opts_opps,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=True,
            update_mode="NO_UPDATE",
            height=520,
            key="oportunidades_grid_unico",
        )

        st.download_button(
            "📥 Descargar CSV",
            data=df_opps.to_csv(index=False).encode("utf-8"),
            file_name="oportunidades_pricing.csv",
            mime="text/csv",
            key="download_opps_unico",
        )
