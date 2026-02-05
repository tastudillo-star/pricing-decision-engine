# pages/1_Master_Posicionamiento.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Optional, Dict, List

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.mySQLHelper import execute_mysql_query  # Cliente MySQL
from utils.pdf_report_builder import ChiperHtmlPdfReport, ReportLayout, StyleRule

# Intentar importar st-aggrid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

PLOTLY_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "scale": 5
    }
}

# ======================================================
# Helpers SQL (compatibles con distintas firmas del helper)
# (misma idea que 07_Gestion_Segmentos_Reglas.py)
# ======================================================
def _mysql_call(query: str, *, fetch: bool) -> Optional[pd.DataFrame]:
    """
    Wrapper compatible:
    - Si execute_mysql_query soporta fetch=... -> usarlo.
    - Si no, fallback a firma antigua (execute_mysql_query(query)).
    """
    try:
        return execute_mysql_query(query, fetch=fetch)
    except TypeError:
        if fetch:
            return execute_mysql_query(query)
        execute_mysql_query(query)
        return None


def run_sql(sql: str) -> Optional[pd.DataFrame]:
    """SELECT con manejo de error UI."""
    try:
        return _mysql_call(sql, fetch=True)
    except Exception as e:
        st.error(f"Error SQL: {e}")
        return None


# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
st.set_page_config(
    page_title="Master — Posicionamiento + Evolución",
    layout="wide",
    page_icon="https://chiper.cl/wp-content/uploads/2023/06/cropped-favicon-192x192.png",
)
st.title("Master — Posicionamiento + Evolución")


# ======================================================
# SIDEBAR: PARÁMETROS (VENTANAS SEPARADAS)
# ======================================================
st.sidebar.subheader("Recargar data")
if st.sidebar.button("Limpiar caché"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Parámetros")

COMPETIDORES = {
    1: "Central Mayorista",
    2: "Alvi",
    3: "La Oferta",
    4: "Mejor precio (min entre 1–3)",
}

id_competidor = st.sidebar.selectbox(
    "Competidor",
    options=list(COMPETIDORES.keys()),
    format_func=lambda x: f"{x} – {COMPETIDORES.get(x, 'Competidor')}",
    index=3,
)


def last_sunday(d: date) -> date:
    days_since_sun = (d.weekday() - 6) % 7
    return d - timedelta(days=days_since_sun)


fecha_base = st.sidebar.date_input(
    "Fecha base (END de la última ventana Chiper)",
    value=last_sunday(date.today()),
)

VENTANA_PRESETS = {
    "Diario": 1,
    "Última semana (7 días)": 7,
    "Últimas 2 semanas (14 días)": 14,
    "Último mes (30 días)": 30,
    "Últimos 3 meses (90 días)": 90,
    "Personalizado": None,
}

st.sidebar.markdown("### Ventana Chiper")
preset_label_ch = st.sidebar.selectbox(
    "Ventana Chiper",
    options=list(VENTANA_PRESETS.keys()),
    index=list(VENTANA_PRESETS.keys()).index("Última semana (7 días)"),
    key="preset_ch",
)

if VENTANA_PRESETS[preset_label_ch] is None:
    ventana_chiper = st.sidebar.number_input(
        "Días ventana Chiper",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        key="win_ch_custom",
    )
else:
    ventana_chiper = int(VENTANA_PRESETS[preset_label_ch])

st.sidebar.markdown("### Ventana Competidor (móvil)")
preset_label_comp = st.sidebar.selectbox(
    "Ventana Competidor",
    options=list(VENTANA_PRESETS.keys()),
    index=list(VENTANA_PRESETS.keys()).index("Última semana (7 días)"),
    key="preset_comp",
)

if VENTANA_PRESETS[preset_label_comp] is None:
    ventana_comp = st.sidebar.number_input(
        "Días ventana Competidor",
        min_value=1,
        max_value=365,
        value=30,
        step=1,
        key="win_comp_custom",
    )
else:
    ventana_comp = int(VENTANA_PRESETS[preset_label_comp])

n_bloques = st.sidebar.number_input(
    "Número de ventanas (bloques) para evolución",
    min_value=2,
    max_value=60,
    value=6 if int(ventana_chiper) == 7 else 12 if int(ventana_chiper) == 30 else 14,
    step=1,
)

# === NUEVO: filtro típico transversal por ventana (no universo común) ===
aplicar_filtro_pos = st.sidebar.checkbox(
    "Aplicar filtro de posicionamiento (0.5 – 2.0)",
    value=True,
)

# === NUEVO: excluir días sin venta en Chiper (elimina cualquier día con SUM(venta_neta)=0) ===
excluir_dias_sin_venta_chiper = st.sidebar.checkbox(
    "Excluir días sin venta en Chiper",
    value=True,
)

st.markdown(
    f"""
**Configuración**
- Competidor: **{COMPETIDORES.get(id_competidor, id_competidor)}**
- Fecha base: **{fecha_base.strftime('%Y-%m-%d')}**
- Ventana Chiper: **{preset_label_ch}** (`{int(ventana_chiper)}` días)  → END = fecha base
- Ventana Competidor (móvil): **{preset_label_comp}** (`{int(ventana_comp)}` días)  → END = fecha base
- N ventanas evolución (solicitadas): **{int(n_bloques)}**
- Filtro posicionamiento (0.5–2.0): **{"ON" if aplicar_filtro_pos else "OFF"}**
- Excluir días sin venta Chiper: **{"ON" if excluir_dias_sin_venta_chiper else "OFF"}**
"""
)


# ======================================================
# UTIL: construir bloques contiguos hacia atrás (usa ventana_chiper como tamaño de bloque)
# ======================================================
def build_blocks(fecha_fin: date, block_days: int, n: int) -> list[dict]:
    blocks = []
    for i in range(int(n)):
        end_i = fecha_fin - timedelta(days=i * block_days)
        start_i = end_i - timedelta(days=block_days - 1)
        blocks.append(
            {
                "idx": i,
                "start": start_i,
                "end": end_i,
                "label": f"{start_i.strftime('%Y-%m-%d')} → {end_i.strftime('%Y-%m-%d')}",
                "end_str": end_i.strftime("%Y-%m-%d"),
                "ventana_chiper": int(block_days),
            }
        )
    return list(reversed(blocks))


blocks = build_blocks(fecha_base, int(ventana_chiper), int(n_bloques))


def apply_pos_filter(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Filtro por ventana: elimina toda fila cuyo posicionamiento esté fuera de [0.5, 2.0].
    Se aplica por snapshot independiente (pueden salir SKUs distintos en cada ventana).
    """
    if df_in is None or df_in.empty:
        return df_in
    if "posicionamiento" not in df_in.columns:
        return df_in
    d = df_in.copy()
    d["posicionamiento"] = pd.to_numeric(d["posicionamiento"], errors="coerce")
    d = d[d["posicionamiento"].notna()]
    if aplicar_filtro_pos:
        d = d[(d["posicionamiento"] >= 0.5) & (d["posicionamiento"] <= 2.0)]
    return d


# ======================================================
# CONSULTA 1 (posicionamiento): TOTAL VENTA PERIODO
# ======================================================
@st.cache_data(show_spinner=False)
def load_venta_total_periodo(fecha_str: str, ventana_chiper: int) -> float:
    q = f"""
    WITH params AS (
      SELECT
        CAST('{fecha_str}' AS DATE) AS fecha_actual,
        {ventana_chiper} AS dias_ventana_chiper
    ),
    valid_days AS (
      SELECT DATE(vc.fecha) AS fecha
      FROM ventas_chiper vc
      JOIN params p
      WHERE
        DATE(vc.fecha) >= DATE_SUB(p.fecha_actual, INTERVAL (p.dias_ventana_chiper - 1) DAY)
        AND DATE(vc.fecha) <= p.fecha_actual
      GROUP BY DATE(vc.fecha)
      HAVING SUM(COALESCE(vc.venta_neta, 0)) > 0
    )
    SELECT
      COALESCE(SUM(vc.venta_neta), 0) AS venta_total_periodo
    FROM ventas_chiper vc
    JOIN params p
    {"JOIN valid_days vd ON vd.fecha = DATE(vc.fecha)" if excluir_dias_sin_venta_chiper else ""}
    WHERE
      DATE(vc.fecha) >= DATE_SUB(p.fecha_actual, INTERVAL (p.dias_ventana_chiper - 1) DAY)
      AND DATE(vc.fecha) <= p.fecha_actual
      AND vc.venta_neta IS NOT NULL;
    """
    dfv = run_sql(q)
    if dfv is None or dfv.empty:
        return 0.0
    val = pd.to_numeric(dfv.iloc[0, 0], errors="coerce")
    return float(val) if pd.notna(val) else 0.0


# ======================================================
# CONSULTA 2 (posicionamiento): DATASET PRINCIPAL
#   - Incluye segmento (id_segmento + nombre)
# ======================================================
@st.cache_data(show_spinner=True)
def load_posicionamiento_categoria(
    id_competidor: int,
    fecha_str: str,
    ventana_chiper: int,
    ventana_comp: int,
) -> pd.DataFrame:
    query = f"""
    WITH
    params AS (
      SELECT
        {id_competidor}             AS id_competidor,
        CAST('{fecha_str}' AS DATE) AS fecha_actual,
        {ventana_chiper}            AS dias_ventana_chiper,
        {ventana_comp}              AS dias_ventana_comp
    ),

    valid_days AS (
      SELECT DATE(vc.fecha) AS fecha
      FROM ventas_chiper vc
      JOIN params p
      WHERE
          DATE(vc.fecha) >= DATE_SUB(p.fecha_actual, INTERVAL (p.dias_ventana_chiper - 1) DAY)
          AND DATE(vc.fecha) <= p.fecha_actual
      GROUP BY DATE(vc.fecha)
      HAVING SUM(COALESCE(vc.venta_neta, 0)) > 0
    ),

    base_chiper AS (
      SELECT
          vc.id_sku,
          DATE(vc.fecha) AS fecha,
          vc.precio_bruto,
          vc.venta_neta,
          vc.front,
          vc.back
      FROM ventas_chiper vc
      JOIN params p
      {"JOIN valid_days vd ON vd.fecha = DATE(vc.fecha)" if excluir_dias_sin_venta_chiper else ""}
      WHERE
          DATE(vc.fecha) >= DATE_SUB(p.fecha_actual, INTERVAL (p.dias_ventana_chiper - 1) DAY)
          AND DATE(vc.fecha) <= p.fecha_actual
          AND vc.precio_bruto IS NOT NULL
          AND vc.venta_neta IS NOT NULL
    ),

    chiper_diario AS (
      SELECT
          bc.id_sku,
          bc.fecha,
          SUM(bc.venta_neta) AS venta_neta_dia,

          (SUM(bc.precio_bruto * bc.venta_neta) / NULLIF(SUM(bc.venta_neta), 0)) AS precio_chiper_dia,
          (SUM(bc.front * bc.venta_neta) / NULLIF(SUM(bc.venta_neta), 0)) AS front_dia,
          (SUM(bc.back  * bc.venta_neta) / NULLIF(SUM(bc.venta_neta), 0)) AS back_dia
      FROM base_chiper bc
      GROUP BY bc.id_sku, bc.fecha
    ),

    chiper_skus AS (
      SELECT COUNT(DISTINCT cd.id_sku) AS total_skus_chiper
      FROM chiper_diario cd
      WHERE cd.venta_neta_dia > 0
    ),

    base_competidor_raw AS (
      SELECT
          pc.id_sku,
          pc.id_competidor,
          DATE(pc.fecha) AS fecha,
          pc.precio_lleno,
          pc.precio_descuento,
          CASE
            WHEN pc.precio_lleno IS NULL AND pc.precio_descuento IS NULL THEN NULL
            WHEN pc.precio_lleno IS NULL THEN pc.precio_descuento
            WHEN pc.precio_descuento IS NULL THEN pc.precio_lleno
            ELSE LEAST(pc.precio_lleno, pc.precio_descuento)
          END AS precio_competidor_min_dia
      FROM precio_competidor pc
      JOIN params p
      WHERE
          DATE(pc.fecha) >= DATE_SUB(p.fecha_actual, INTERVAL (p.dias_ventana_comp - 1) DAY)
          AND DATE(pc.fecha) <= p.fecha_actual
          AND (pc.precio_lleno IS NOT NULL OR pc.precio_descuento IS NOT NULL)
          AND (
            (p.id_competidor IN (1,2,3) AND pc.id_competidor = p.id_competidor)
            OR
            (p.id_competidor = 4 AND pc.id_competidor IN (1,2,3))
          )
    ),

    base_competidor AS (
      SELECT *
      FROM (
        SELECT
          r.*,
          ROW_NUMBER() OVER (
            PARTITION BY r.id_sku, r.fecha
            ORDER BY r.precio_competidor_min_dia ASC, r.id_competidor ASC
          ) AS rn
        FROM base_competidor_raw r
        WHERE r.precio_competidor_min_dia IS NOT NULL
      ) t
      WHERE t.rn = 1
    ),

    joined_diario AS (
      SELECT
          p.fecha_actual,
          p.dias_ventana_chiper,
          p.dias_ventana_comp,
          p.id_competidor,
          cd.id_sku,
          cd.fecha,

          cd.venta_neta_dia,
          cd.precio_chiper_dia,
          (cd.front_dia + cd.back_dia) AS margen_dia,

          bc.precio_lleno AS precio_lleno_dia,
          bc.precio_descuento AS precio_descuento_dia,
          bc.precio_competidor_min_dia
      FROM chiper_diario cd
      CROSS JOIN params p
      LEFT JOIN base_competidor bc
      ON bc.id_sku = cd.id_sku
     AND bc.fecha  = cd.fecha
      WHERE cd.venta_neta_dia > 0
    ),

    daily_metrics AS (
      SELECT
          jd.*,
          CASE
            WHEN jd.precio_chiper_dia IS NULL THEN NULL
            WHEN jd.precio_competidor_min_dia IS NULL THEN NULL
            WHEN jd.precio_competidor_min_dia = 0 THEN NULL
            ELSE jd.precio_chiper_dia / jd.precio_competidor_min_dia
          END AS posicionamiento_dia
      FROM joined_diario jd
    ),

    agg_sku AS (
      SELECT
          dm.fecha_actual,
          dm.dias_ventana_chiper,
          dm.dias_ventana_comp,
          dm.id_competidor,
          dm.id_sku,

          SUM(dm.venta_neta_dia) AS sum_venta_neta,

          (SUM(dm.precio_chiper_dia * dm.venta_neta_dia) / NULLIF(SUM(dm.venta_neta_dia), 0)) AS precio_chiper_pond_ventana,

          (SUM(dm.precio_lleno_dia * dm.venta_neta_dia) / NULLIF(SUM(CASE WHEN dm.precio_lleno_dia IS NULL THEN 0 ELSE dm.venta_neta_dia END), 0))
            AS precio_lleno_pond_ventana,

          (SUM(dm.precio_descuento_dia * dm.venta_neta_dia) / NULLIF(SUM(CASE WHEN dm.precio_descuento_dia IS NULL THEN 0 ELSE dm.venta_neta_dia END), 0))
            AS precio_descuento_pond_ventana,

          (SUM(dm.precio_competidor_min_dia * dm.venta_neta_dia) / NULLIF(SUM(CASE WHEN dm.precio_competidor_min_dia IS NULL THEN 0 ELSE dm.venta_neta_dia END), 0))
            AS precio_competidor_min_pond_ventana,

          (SUM(dm.margen_dia * dm.venta_neta_dia) / NULLIF(SUM(dm.venta_neta_dia), 0)) AS margen_pond_ventana,

          (SUM(dm.posicionamiento_dia * dm.venta_neta_dia) / NULLIF(SUM(CASE WHEN dm.posicionamiento_dia IS NULL THEN 0 ELSE dm.venta_neta_dia END), 0))
            AS posicionamiento_pond_ventana
      FROM daily_metrics dm
      GROUP BY
          dm.fecha_actual,
          dm.dias_ventana_chiper,
          dm.dias_ventana_comp,
          dm.id_competidor,
          dm.id_sku
    ),

    enriched AS (
      SELECT
          a.fecha_actual,
          a.dias_ventana_chiper,
          a.dias_ventana_comp,
          a.id_competidor,
          a.id_sku,
          s.sku,
          mc.nombre AS macro,
          c.nombre  AS categoria,
          pr.nombre AS proveedor,

          s.id_segmento AS id_segmento,
          seg.nombre    AS segmento,

          s.nombre  AS nombre,

          a.precio_chiper_pond_ventana    AS precio_chiper,
          a.precio_lleno_pond_ventana     AS precio_lleno_competidor,
          a.precio_descuento_pond_ventana AS precio_descuento_competidor,

          a.sum_venta_neta                AS venta_neta,
          a.posicionamiento_pond_ventana  AS posicionamiento,
          a.margen_pond_ventana           AS margen
      FROM agg_sku a
      JOIN sku s
        ON s.id = a.id_sku
      LEFT JOIN categoria c
        ON c.id = s.id_categoria
      LEFT JOIN macro_categoria mc
        ON mc.id = c.id_macro
      LEFT JOIN proveedor pr
        ON pr.id = s.id_proveedor
      LEFT JOIN segmento seg
        ON seg.id = s.id_segmento
    ),

    final AS (
      SELECT
          e.*,
          cs.total_skus_chiper,
          CASE
            WHEN SUM(e.venta_neta) OVER () = 0 THEN NULL
            ELSE e.venta_neta / SUM(e.venta_neta) OVER ()
          END AS peso_venta
      FROM enriched e
      CROSS JOIN chiper_skus cs
    )

    SELECT
        id_sku,
        sku,
        macro,
        categoria,
        proveedor,
        id_segmento,
        segmento,
        nombre,
        precio_chiper,
        precio_lleno_competidor,
        precio_descuento_competidor,
        venta_neta,
        posicionamiento,
        peso_venta,
        margen,
        total_skus_chiper
    FROM final
    ORDER BY sku;
    """
    dfq = run_sql(query)
    return dfq if isinstance(dfq, pd.DataFrame) else pd.DataFrame()


# ======================================================
# QUERY SNAPSHOT (evolución)
#  - MOD: agrega proveedor para permitir vista Proveedor en scatter + filtros comunes
# ======================================================
@st.cache_data(show_spinner=False)
def load_snapshot_posicionamiento(
    id_competidor: int,
    fecha_str: str,
    ventana_chiper: int,
    ventana_comp: int,
) -> pd.DataFrame:
    query = f"""
    WITH
    params AS (
      SELECT
        {id_competidor}             AS id_competidor,
        CAST('{fecha_str}' AS DATE) AS fecha_actual,
        {ventana_chiper}            AS dias_ventana_chiper,
        {ventana_comp}              AS dias_ventana_comp
    ),

    valid_days AS (
      SELECT DATE(vc.fecha) AS fecha
      FROM ventas_chiper vc
      JOIN params p
      WHERE
          DATE(vc.fecha) >= DATE_SUB(p.fecha_actual, INTERVAL (p.dias_ventana_chiper - 1) DAY)
          AND DATE(vc.fecha) <= p.fecha_actual
      GROUP BY DATE(vc.fecha)
      HAVING SUM(COALESCE(vc.venta_neta, 0)) > 0
    ),

    base_chiper AS (
      SELECT
          vc.id_sku,
          DATE(vc.fecha) AS fecha,
          vc.precio_bruto,
          vc.venta_neta,
          vc.front,
          vc.back
      FROM ventas_chiper vc
      JOIN params p
      {"JOIN valid_days vd ON vd.fecha = DATE(vc.fecha)" if excluir_dias_sin_venta_chiper else ""}
      WHERE
          DATE(vc.fecha) >= DATE_SUB(p.fecha_actual, INTERVAL (p.dias_ventana_chiper - 1) DAY)
          AND DATE(vc.fecha) <= p.fecha_actual
          AND vc.precio_bruto IS NOT NULL
          AND vc.venta_neta IS NOT NULL
    ),

    chiper_diario AS (
      SELECT
          bc.id_sku,
          bc.fecha,
          SUM(bc.venta_neta) AS venta_neta_dia,
          (SUM(bc.precio_bruto * bc.venta_neta) / NULLIF(SUM(bc.venta_neta), 0)) AS precio_chiper_dia,
          (SUM(bc.front * bc.venta_neta) / NULLIF(SUM(bc.venta_neta), 0)) AS front_dia,
          (SUM(bc.back  * bc.venta_neta) / NULLIF(SUM(bc.venta_neta), 0)) AS back_dia
      FROM base_chiper bc
      GROUP BY bc.id_sku, bc.fecha
    ),

    base_competidor_raw AS (
      SELECT
          pc.id_sku,
          pc.id_competidor,
          DATE(pc.fecha) AS fecha,
          pc.precio_lleno,
          pc.precio_descuento,
          CASE
            WHEN pc.precio_lleno IS NULL AND pc.precio_descuento IS NULL THEN NULL
            WHEN pc.precio_lleno IS NULL THEN pc.precio_descuento
            WHEN pc.precio_descuento IS NULL THEN pc.precio_lleno
            ELSE LEAST(pc.precio_lleno, pc.precio_descuento)
          END AS precio_competidor_min_dia
      FROM precio_competidor pc
      JOIN params p
      WHERE
          DATE(pc.fecha) >= DATE_SUB(p.fecha_actual, INTERVAL (p.dias_ventana_comp - 1) DAY)
          AND DATE(pc.fecha) <= p.fecha_actual
          AND (pc.precio_lleno IS NOT NULL OR pc.precio_descuento IS NOT NULL)
          AND (
            (p.id_competidor IN (1,2,3) AND pc.id_competidor = p.id_competidor)
            OR
            (p.id_competidor = 4 AND pc.id_competidor IN (1,2,3))
          )
    ),

    base_competidor AS (
      SELECT *
      FROM (
        SELECT
          r.*,
          ROW_NUMBER() OVER (
            PARTITION BY r.id_sku, r.fecha
            ORDER BY r.precio_competidor_min_dia ASC, r.id_competidor ASC
          ) AS rn
        FROM base_competidor_raw r
        WHERE r.precio_competidor_min_dia IS NOT NULL
      ) t
      WHERE t.rn = 1
    ),

    joined_diario AS (
      SELECT
          cd.id_sku,
          cd.fecha,
          cd.venta_neta_dia,
          cd.precio_chiper_dia,
          (cd.front_dia + cd.back_dia) AS margen_dia,
          bc.precio_competidor_min_dia
      FROM chiper_diario cd
      CROSS JOIN params p
      LEFT JOIN base_competidor bc
          ON bc.id_sku = cd.id_sku
         AND bc.fecha  = cd.fecha
      WHERE cd.venta_neta_dia > 0
    ),

    daily_metrics AS (
      SELECT
          jd.*,
          CASE
            WHEN jd.precio_chiper_dia IS NULL THEN NULL
            WHEN jd.precio_competidor_min_dia IS NULL THEN NULL
            WHEN jd.precio_competidor_min_dia = 0 THEN NULL
            ELSE jd.precio_chiper_dia / jd.precio_competidor_min_dia
          END AS posicionamiento_dia
      FROM joined_diario jd
    ),

    agg_sku AS (
      SELECT
          dm.id_sku,
          SUM(dm.venta_neta_dia) AS venta_neta,
          (SUM(dm.posicionamiento_dia * dm.venta_neta_dia) / NULLIF(SUM(CASE WHEN dm.posicionamiento_dia IS NULL THEN 0 ELSE dm.venta_neta_dia END), 0))
            AS posicionamiento,
          (SUM(dm.margen_dia * dm.venta_neta_dia) / NULLIF(SUM(dm.venta_neta_dia), 0)) AS margen
      FROM daily_metrics dm
      GROUP BY dm.id_sku
    )

    SELECT
        a.id_sku,
        s.sku,
        mc.nombre AS macro,
        c.nombre  AS categoria,
        pr.nombre AS proveedor,
        s.nombre  AS nombre,
        s.id_segmento AS id_segmento,
        seg.nombre    AS segmento,
        a.venta_neta,
        a.posicionamiento,
        a.margen
    FROM agg_sku a
    JOIN sku s
      ON s.id = a.id_sku
    LEFT JOIN categoria c
      ON c.id = s.id_categoria
    LEFT JOIN macro_categoria mc
      ON mc.id = c.id_macro
    LEFT JOIN proveedor pr
      ON pr.id = s.id_proveedor
    LEFT JOIN segmento seg
      ON seg.id = s.id_segmento
    ;
    """
    dfq = run_sql(query)
    return dfq if isinstance(dfq, pd.DataFrame) else pd.DataFrame()


# ======================================================
# Reglas de negocio (segmento) + override (SKU)
#   - regla_negocio: (id_segmento, fecha) -> pos_top, pos_fondo, margen?
#   - regla_negocio_override: (id_sku, fecha) -> pos_top, pos_fondo, margen?
# Precedencia:
#   override SKU (si existe) > regla segmento
# ======================================================
@st.cache_data(show_spinner=False, ttl=600)
def load_reglas_segmento(fecha_str: str) -> pd.DataFrame:
    q = f"""
    SELECT
      id_segmento,
      posicionamiento_top,
      posicionamiento_fondo,
      margen
    FROM regla_negocio
    WHERE fecha = CAST('{fecha_str}' AS DATE);
    """
    d = run_sql(q)
    return d if isinstance(d, pd.DataFrame) else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600)
def load_overrides_sku(fecha_str: str) -> pd.DataFrame:
    q = f"""
    SELECT
      id_sku,
      posicionamiento_top,
      posicionamiento_fondo,
      margen
    FROM regla_negocio_override
    WHERE fecha = CAST('{fecha_str}' AS DATE);
    """
    d = run_sql(q)
    return d if isinstance(d, pd.DataFrame) else pd.DataFrame()


# ======================================================
# Aplicar lógica Top80/Fondo20 por categoría y asignar
# "posicionamiento_objetivo" usando:
#   - override SKU si existe
#   - si no, regla por segmento
#
# Produce columnas:
#   - bucket_cat (TOP80 / FONDO20)
#   - share_cat, cum_share_cat
#   - pos_regla_top, pos_regla_fondo
#   - posicionamiento_objetivo (aplicado)
#   - gap_pos (pos_actual - pos_objetivo)
# ======================================================
def apply_reglas_top_fondo_por_categoria(df_in: pd.DataFrame, *, fecha_str: str) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return df_in

    d = df_in.copy()

    # Requisitos mínimos
    for c in ["id_sku", "id_segmento", "categoria", "venta_neta", "posicionamiento"]:
        if c not in d.columns:
            return d

    # Tipos
    d["id_sku"] = pd.to_numeric(d["id_sku"], errors="coerce")
    d["id_segmento"] = pd.to_numeric(d["id_segmento"], errors="coerce")
    d["venta_neta"] = pd.to_numeric(d["venta_neta"], errors="coerce").fillna(0.0)
    d["posicionamiento"] = pd.to_numeric(d["posicionamiento"], errors="coerce")

    d["categoria"] = d["categoria"].fillna("Sin categoría")

    # ---- 1) Bucket Top80/Fondo20 por categoría (según venta_neta) ----
    # Orden interno por categoría
    d = d.sort_values(["categoria", "venta_neta"], ascending=[True, False]).reset_index(drop=True)

    cat_total = d.groupby("categoria", dropna=False)["venta_neta"].transform("sum")
    d["share_cat"] = np.where(cat_total > 0, d["venta_neta"] / cat_total, np.nan)

    # cumsum por categoría respetando el sort previo
    d["cum_share_cat"] = d.groupby("categoria", dropna=False)["share_cat"].cumsum()

    # Regla: Top 80% = cum_share <= 0.80
    d["bucket_cat"] = np.where(d["cum_share_cat"] <= 0.80, "TOP80", "FONDO20")

    # ---- 2) Cargar reglas por fecha y merge ----
    rn = load_reglas_segmento(fecha_str)
    if not rn.empty:
        rn = rn.copy()
        rn["id_segmento"] = pd.to_numeric(rn["id_segmento"], errors="coerce")
        rn = rn.rename(
            columns={
                "posicionamiento_top": "rn_pos_top",
                "posicionamiento_fondo": "rn_pos_fondo",
                "margen": "rn_margen",
            }
        )
        d = d.merge(rn[["id_segmento", "rn_pos_top", "rn_pos_fondo", "rn_margen"]], on="id_segmento", how="left")
    else:
        d["rn_pos_top"] = np.nan
        d["rn_pos_fondo"] = np.nan
        d["rn_margen"] = np.nan

    ov = load_overrides_sku(fecha_str)
    if not ov.empty:
        ov = ov.copy()
        ov["id_sku"] = pd.to_numeric(ov["id_sku"], errors="coerce")
        ov = ov.rename(
            columns={
                "posicionamiento_top": "ov_pos_top",
                "posicionamiento_fondo": "ov_pos_fondo",
                "margen": "ov_margen",
            }
        )
        d = d.merge(ov[["id_sku", "ov_pos_top", "ov_pos_fondo", "ov_margen"]], on="id_sku", how="left")
    else:
        d["ov_pos_top"] = np.nan
        d["ov_pos_fondo"] = np.nan
        d["ov_margen"] = np.nan

    # ---- 3) Coalesce: override > regla segmento ----
    d["pos_regla_top"] = d["ov_pos_top"].combine_first(d["rn_pos_top"])
    d["pos_regla_fondo"] = d["ov_pos_fondo"].combine_first(d["rn_pos_fondo"])
    d["margen_regla"] = d["ov_margen"].combine_first(d["rn_margen"])

    # ---- 4) Aplicar según bucket ----
    d["posicionamiento_objetivo"] = np.where(
        d["bucket_cat"] == "TOP80",
        d["pos_regla_top"],
        d["pos_regla_fondo"],
    )

    d["gap_pos"] = d["posicionamiento"] - d["posicionamiento_objetivo"]

    return d

# ======================================================
# CARGA POSICIONAMIENTO (última ventana)
# ======================================================
venta_total_periodo = load_venta_total_periodo(
    fecha_str=fecha_base.strftime("%Y-%m-%d"),
    ventana_chiper=int(ventana_chiper),
)

df = load_posicionamiento_categoria(
    id_competidor=id_competidor,
    fecha_str=fecha_base.strftime("%Y-%m-%d"),
    ventana_chiper=int(ventana_chiper),
    ventana_comp=int(ventana_comp),
)

if df is None or df.empty:
    st.error("No se encontraron datos para la ventana seleccionada.")
    st.stop()

for col in [
    "precio_chiper",
    "precio_lleno_competidor",
    "precio_descuento_competidor",
    "venta_neta",
    "posicionamiento",
    "peso_venta",
    "margen",
    "id_segmento",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "total_skus_chiper" in df.columns:
    try:
        total_skus_chiper = int(df["total_skus_chiper"].iloc[0])
    except Exception:
        total_skus_chiper = df["total_skus_chiper"].iloc[0]
else:
    total_skus_chiper = df["sku"].nunique()

skus_con_posicionamiento = df[df["posicionamiento"].notna()]["sku"].nunique()
representatividad = (skus_con_posicionamiento / total_skus_chiper) if total_skus_chiper else np.nan

# Bases (antes del filtro 0.5–2)
df_all = df.copy()
df_pos = df[df["posicionamiento"].notna()].copy()

# Mantener solo pos no nulo como siempre
df = df[df["posicionamiento"].notna()].copy()

# === Aplicar filtro 0.5–2.0 SOLO si checkbox ON ===
df = apply_pos_filter(df)
df_pos = apply_pos_filter(df_pos)  # coherencia si filtro ON
# === NUEVO: aplicar reglas top/fondo por categoría (última ventana) ===
df = apply_reglas_top_fondo_por_categoria(df, fecha_str=fecha_base.strftime("%Y-%m-%d"))

# Recalcular representatividad visible si filtro ON
if aplicar_filtro_pos:
    skus_con_posicionamiento = df["sku"].nunique()
    representatividad = (skus_con_posicionamiento / total_skus_chiper) if total_skus_chiper else np.nan

venta_total_filtrada = float(df["venta_neta"].sum(skipna=True) or 0.0)

pos_pond_total = np.nan
margen_pond_total = np.nan
if venta_total_filtrada and not np.isclose(venta_total_filtrada, 0):
    pos_pond_total = float((df["posicionamiento"] * df["venta_neta"]).sum(skipna=True) / venta_total_filtrada)
    margen_pond_total = float((df["margen"] * df["venta_neta"]).sum(skipna=True) / venta_total_filtrada)

pct_venta_representada = np.nan
if venta_total_periodo and not np.isclose(venta_total_periodo, 0):
    pct_venta_representada = venta_total_filtrada / venta_total_periodo


# ======================================================
# CARGA EVOLUCIÓN: N snapshots (filtrado por ventana)
# - Si una ventana queda sin venta (venta_total == 0), se EXCLUYE COMPLETA:
#   no entra a df_line, ni macro_lines, ni snapshots, ni gráficos/pivots/PDF.
# ======================================================
snapshots = []
line_points = []
macro_lines = []

# Bloques efectivos (solo aquellos con venta > 0 luego del filtro)
blocks_eff = []

with st.spinner("Calculando snapshots por ventana..."):
    for b in blocks:
        dfi = load_snapshot_posicionamiento(
            id_competidor=id_competidor,
            fecha_str=b["end_str"],
            ventana_chiper=int(b["ventana_chiper"]),
            ventana_comp=int(ventana_comp),
        )

        if dfi is None or dfi.empty:
            continue

        for col in ["venta_neta", "posicionamiento", "margen", "id_segmento"]:
            if col in dfi.columns:
                dfi[col] = pd.to_numeric(dfi[col], errors="coerce")

        # pos válido + filtro opcional por ventana (0.5–2.0)
        dfi = apply_pos_filter(dfi)

        # === NUEVO: aplicar reglas top/fondo por categoría (por fecha del bloque) ===
        dfi = apply_reglas_top_fondo_por_categoria(dfi, fecha_str=b["end_str"])


        # Si luego del filtro no queda venta, se salta COMPLETO el bloque
        venta_total = float(dfi["venta_neta"].sum(skipna=True) or 0.0)
        if dfi.empty or venta_total <= 0:
            continue

        blocks_eff.append(b)

        pos_total = float((dfi["posicionamiento"] * dfi["venta_neta"]).sum(skipna=True) / venta_total)

        line_points.append(
            {
                "bloque": b["label"],
                "end": b["end_str"],
                "pos_total": pos_total,
                "venta_total": venta_total,
            }
        )

        g = (
            dfi.groupby("macro", dropna=False)
            .apply(
                lambda grp: pd.Series(
                    {
                        "venta_macro": float(grp["venta_neta"].sum(skipna=True) or 0.0),
                        "pos_macro": (
                            float((grp["posicionamiento"] * grp["venta_neta"]).sum(skipna=True) / float(grp["venta_neta"].sum(skipna=True)))
                            if float(grp["venta_neta"].sum(skipna=True) or 0.0) > 0
                            else np.nan
                        ),
                    }
                )
            )
            .reset_index()
        )
        g["bloque"] = b["label"]
        macro_lines.append(g)

        dfi["window_label"] = b["label"]
        snapshots.append(dfi)

if len(blocks_eff) == 0:
    st.warning("No hay ventanas con venta (después de excluir días sin venta en Chiper y aplicar filtro POS).")
    st.stop()

n_bloques_eff = int(len(blocks_eff))

df_line = pd.DataFrame(line_points)

df_macro_line = (
    pd.concat(macro_lines, ignore_index=True)
    if len(macro_lines)
    else pd.DataFrame(columns=["macro", "bloque", "venta_macro", "pos_macro"])
)

df_line["bloque"] = pd.Categorical(df_line["bloque"], categories=[b["label"] for b in blocks_eff], ordered=True)
df_line = df_line.sort_values("bloque")


# ======================================================
# 1) KPIs (dos columnas)
# ======================================================
st.subheader("KPIs (última ventana)")

col1, col15, col5, col4 = st.columns(4)
col2, col3, col6  = st.columns(3)

with col1:
    st.metric("Venta total (periodo completo)", f"${venta_total_periodo:,.0f}")
with col15:
    st.metric("Venta total (representado)", f"${venta_total_filtrada:,.0f}")
with col5:
    st.metric("% Venta representada", f"{pct_venta_representada:.2%}" if not np.isnan(pct_venta_representada) else "N/A")
with col2:
    st.metric("Posicionamiento ponderado", f"{pos_pond_total*100:.2f}%" if not np.isnan(pos_pond_total) else "N/A")
with col3:
    st.metric("Margen ponderado (front + back)", f"{margen_pond_total*100:.2f}%" if not np.isnan(margen_pond_total) else "N/A")
with col4:
    st.metric("Representatividad SKUs", f"{representatividad:.2%}" if not np.isnan(representatividad) else "N/A")
with col6:
    st.metric("N ventanas (efectivas)", f"{n_bloques_eff}")

st.caption(
    f"Última ventana Chiper (efectiva): {blocks_eff[-1]['label']} | "
    f"Ventana comp: {int(ventana_comp)} días (móvil, END = bloque). | "
    f"Ventanas solicitadas: {int(n_bloques)}"
)

st.markdown("---")


# ======================================================
# 2) Evolución total + 3) Scatter (misma sección, 2 columnas)
#    MOD: controles arriba que afectan a AMBOS gráficos del bloque.
# ======================================================
st.subheader("Evolución y relación (compacto)")

# ---------- helpers para filtros comunes (solo esta sección) ----------
NAME_TO_QUADRANT = {
    "contribuyente": 1,
    "poderosa": 2,
    "magnetica": 3,
    "magnética": 3,
    "oportunista": 4,
}
QUADRANT_TO_LABEL = {
    1: "Contribuyente",
    2: "Poderosa",
    3: "Magnética",
    4: "Oportunista",
}
QUADRANT_COLOR = {
    1: "#2F80ED",
    2: "#9B51E0",
    3: "#F2994A",
    4: "#27AE60",
}


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


def _norm_seg_name(s: str) -> str:
    return _safe_str(s).strip().lower()


def compute_rep_segment(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Retorna df: [group_col, id_segmento_rep, segmento_rep]
    usando max venta_neta por segmento dentro de cada grupo.
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=[group_col, "id_segmento_rep", "segmento_rep"])

    d = df_in.copy()
    if "venta_neta" not in d.columns:
        return pd.DataFrame(columns=[group_col, "id_segmento_rep", "segmento_rep"])

    if "id_segmento" in d.columns:
        d["id_segmento"] = pd.to_numeric(d["id_segmento"], errors="coerce")
    else:
        d["id_segmento"] = np.nan

    if "segmento" not in d.columns:
        d["segmento"] = ""

    d[group_col] = d[group_col].fillna(f"Sin {group_col}")

    seg_sales = (
        d.dropna(subset=[group_col, "id_segmento"])
        .groupby([group_col, "id_segmento", "segmento"], dropna=False)["venta_neta"]
        .sum()
        .reset_index()
    )
    if seg_sales.empty:
        return pd.DataFrame(columns=[group_col, "id_segmento_rep", "segmento_rep"])

    rep_seg = (
        seg_sales
        # desempate estable: si dos segmentos tienen misma venta en el grupo,
        # elige el de menor id_segmento (o sea, consistente siempre).
        .sort_values([group_col, "venta_neta", "id_segmento"], ascending=[True, False, True])
        .drop_duplicates(subset=[group_col], keep="first")
        .rename(
            columns={
                "id_segmento": "id_segmento_rep",
                "segmento": "segmento_rep",
                "venta_neta": "venta_segmento_rep",
            }
        )
    )

    return rep_seg[[group_col, "id_segmento_rep", "segmento_rep"]].copy()


def agg_weighted_level(grp: pd.DataFrame) -> pd.Series:
    venta_total = float(grp["venta_neta"].sum(skipna=True) or 0.0)
    if venta_total and not np.isclose(venta_total, 0):
        pos_pond = float((grp["posicionamiento"] * grp["venta_neta"]).sum(skipna=True) / venta_total)
        margen_pond = float((grp["margen"] * grp["venta_neta"]).sum(skipna=True) / venta_total)
    else:
        pos_pond = np.nan
        margen_pond = np.nan
    return pd.Series(
        {
            "venta_neta_level": venta_total,
            "posicionamiento_pond": pos_pond,
            "margen_pond": margen_pond,
        }
    )


def apply_scope_filter(df_in: pd.DataFrame, *, mode: str, macro_sel: Optional[str], cat_sel: Optional[str]) -> pd.DataFrame:
    """
    Filtra df_in según:
      - mode == "Macro": sin filtro (total)
      - mode == "Categorías": filtra por macro == macro_sel
      - mode == "Proveedores": filtra por macro == macro_sel y categoria == cat_sel
    """
    if df_in is None or df_in.empty:
        return df_in
    d = df_in.copy()

    d["macro"] = d.get("macro", pd.Series([], dtype=str)).fillna("Sin macro")
    d["categoria"] = d.get("categoria", pd.Series([], dtype=str)).fillna("Sin categoría")
    d["proveedor"] = d.get("proveedor", pd.Series([], dtype=str)).fillna("Sin proveedor")

    if mode == "Categorías" and macro_sel:
        d = d[d["macro"] == macro_sel]
    elif mode == "Proveedores" and macro_sel and cat_sel:
        d = d[(d["macro"] == macro_sel) & (d["categoria"] == cat_sel)]
    return d


def build_df_line_scoped(
    df_long: pd.DataFrame,
    labels_ordered: List[str],
    *,
    mode: str,
    macro_sel: Optional[str],
    cat_sel: Optional[str],
) -> pd.DataFrame:
    """
    Recalcula evolución (pos_total / venta_total) usando el MISMO filtro scope que el scatter.
    """
    rows = []
    for lbl in labels_ordered:
        dwin = df_long[df_long["window_label"] == lbl].copy()
        if dwin.empty:
            continue
        dwin = apply_scope_filter(dwin, mode=mode, macro_sel=macro_sel, cat_sel=cat_sel)
        if dwin.empty:
            continue

        vtot = float(dwin["venta_neta"].sum(skipna=True) or 0.0)
        if vtot <= 0:
            continue

        pos = float((dwin["posicionamiento"] * dwin["venta_neta"]).sum(skipna=True) / vtot)
        rows.append({"bloque": lbl, "pos_total": pos, "venta_total": vtot})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["bloque"] = pd.Categorical(out["bloque"], categories=labels_ordered, ordered=True)
    out = out.sort_values("bloque")
    return out


# ---------- df_long base para recalcular evolución "scoped" ----------
df_long_scatter = pd.concat(snapshots, ignore_index=True) if len(snapshots) else pd.DataFrame()
if not df_long_scatter.empty:
    # normalización mínima para filtros
    for c in ["macro", "categoria", "proveedor"]:
        if c in df_long_scatter.columns:
            df_long_scatter[c] = df_long_scatter[c].fillna(f"Sin {c}")
    for c in ["venta_neta", "posicionamiento", "margen", "id_segmento"]:
        if c in df_long_scatter.columns:
            df_long_scatter[c] = pd.to_numeric(df_long_scatter[c], errors="coerce")

# ---------- Controles arriba de ambos gráficos ----------
ctrl = st.container()
with ctrl:
    c0, c1, c2, c3 = st.columns([1.35, 1.35, 1.35, 1.0], gap="large")

    with c0:
        scatter_mode = st.radio(
            "Vista",
            options=["Macro", "Categorías", "Proveedores"],
            index=0,
            horizontal=True,
            key="scatter_mode_top",
        )

    # opciones desde la última ventana (df)
    df_last_for_opts = df.copy()
    df_last_for_opts["macro"] = df_last_for_opts["macro"].fillna("Sin macro")
    df_last_for_opts["categoria"] = df_last_for_opts["categoria"].fillna("Sin categoría")
    df_last_for_opts["proveedor"] = df_last_for_opts["proveedor"].fillna("Sin proveedor")

    macros_opt = sorted(df_last_for_opts["macro"].dropna().unique().tolist())
    macro_sel = None
    cat_sel = None

    if scatter_mode == "Categorías":
        with c1:
            macro_sel = st.selectbox(
                "Macro a detallar",
                options=macros_opt if macros_opt else ["Sin macro"],
                index=0,
                key="scatter_macro_sel_cat",
            )
        with c2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.caption("Scatter: categorías dentro de la macro seleccionada.")
        with c3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.caption("Evolución: filtrada por esa macro.")

    elif scatter_mode == "Proveedores":
        with c1:
            macro_sel = st.selectbox(
                "Macro",
                options=macros_opt if macros_opt else ["Sin macro"],
                index=0,
                key="scatter_macro_sel_prov",
            )
        cats_opt = sorted(df_last_for_opts[df_last_for_opts["macro"] == macro_sel]["categoria"].dropna().unique().tolist()) if macro_sel else []
        with c2:
            cat_sel = st.selectbox(
                "Categoría",
                options=cats_opt if cats_opt else ["Sin categoría"],
                index=0,
                key="scatter_cat_sel_prov",
            )
        with c3:
            st.caption("Scatter: proveedores dentro de la categoría seleccionada.\nEvolución: filtrada por macro+categ.")

    else:
        with c1:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.caption("Scatter: macrocategorías (vista actual).")
        with c2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.caption("Evolución: total (sin filtro).")
        with c3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.caption("")


# ---------- Data "scoped" para evolución (izquierda) ----------
labels_eff = [b["label"] for b in blocks_eff]
df_line_scoped = build_df_line_scoped(
    df_long_scatter,
    labels_eff,
    mode=scatter_mode,
    macro_sel=macro_sel,
    cat_sel=cat_sel,
)

# fallback si el filtro deja todo vacío
if df_line_scoped.empty:
    st.info("El filtro seleccionado deja 0 venta en todas las ventanas. Mostrando evolución total como fallback.")
    df_line_scoped = df_line.copy()


# ---------- Data "scoped" para scatter (derecha) ----------
df_last_scoped = apply_scope_filter(df, mode=scatter_mode, macro_sel=macro_sel, cat_sel=cat_sel)

# ---------- Render 2 columnas ----------
col_left, col_right = st.columns(2, gap="large")

with col_left:
    # título dinámico
    if scatter_mode == "Categorías" and macro_sel:
        t_left = f"Evolución (Macro: {macro_sel})"
    elif scatter_mode == "Proveedores" and macro_sel and cat_sel:
        t_left = f"Evolución (Categoría: {macro_sel} / {cat_sel})"
    else:
        t_left = "Evolución (total)"

    st.markdown(f"#### {t_left}")
    fig_total = make_subplots(specs=[[{"secondary_y": True}]])

    fig_total.add_trace(
        go.Scatter(
            x=df_line_scoped["bloque"],
            y=df_line_scoped["pos_total"],
            mode="lines+markers",
            name="Posicionamiento (ratio)",
            hovertemplate="Ventana: %{x}<br>POS: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig_total.add_trace(
        go.Scatter(
            x=df_line_scoped["bloque"],
            y=df_line_scoped["venta_total"],
            mode="lines+markers",
            name="Venta (CLP)",
            hovertemplate="Ventana: %{x}<br>Venta: $%{y:,.0f}<extra></extra>",
            line=dict(color="#ff0000"),
            marker=dict(color="#ff0000"),
        ),
        secondary_y=True,
    )

    fig_total.add_hline(y=1.0, line_width=1, line_dash="dash", opacity=0.8)

    # =========================================================
    # Expansión 20% abajo + 20% arriba PARA AMBOS EJES Y
    # (misma lógica / mismo factor)
    # =========================================================
    EXPAND = 0.50

    def expanded_range(series: pd.Series, expand: float) -> list[float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return [0.0, 1.0]

        vmin = float(s.min())
        vmax = float(s.max())
        span = vmax - vmin

        # Si todos los valores son iguales, crea un span artificial
        if span == 0:
            span = 1.0 if vmax == 0 else abs(vmax) * 0.10
            if span == 0:
                span = 1.0

        return [vmin - expand * span, vmax + expand * span]

    y1_range = expanded_range(df_line_scoped["pos_total"], EXPAND)
    y2_range = expanded_range(df_line_scoped["venta_total"], EXPAND)

    # (opcional) evitar negativos en venta
    if y2_range[0] < 0:
        y2_range[0] = 0.0

    fig_total.update_layout(
        height=520,
        margin=dict(t=20, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    fig_total.update_yaxes(
        title_text="Posicionamiento (ratio)",
        secondary_y=False,
        range=y1_range,
    )
    fig_total.update_yaxes(
        title_text="Venta (CLP)",
        secondary_y=True,
        tickformat=",.0f",
        range=y2_range,
    )



    st.plotly_chart(fig_total, use_container_width=True, config=PLOTLY_CONFIG)

with col_right:
    # título dinámico
    if scatter_mode == "Categorías" and macro_sel:
        t_right = f"Posicionamiento vs Margen (Categorías en: {macro_sel})"
        label_col = "categoria"
        group_cols = ["categoria"]
        label_title = "Categoría"
    elif scatter_mode == "Proveedores" and macro_sel and cat_sel:
        t_right = f"Posicionamiento vs Margen (Proveedores en: {macro_sel} / {cat_sel})"
        label_col = "proveedor"
        group_cols = ["proveedor"]
        label_title = "Proveedor"
    else:
        t_right = "Posicionamiento vs Margen (Macro)"
        label_col = "macro"
        group_cols = ["macro"]
        label_title = "Macro"

    st.markdown(f"#### {t_right}")

    if df_last_scoped is None or df_last_scoped.empty:
        st.info("No hay datos suficientes para el scatter con el filtro seleccionado.")
        fig_sc = None
    else:
        # agregación nivel elegido
        d0 = df_last_scoped.copy()
        for c in ["macro", "categoria", "proveedor"]:
            if c in d0.columns:
                d0[c] = d0[c].fillna(f"Sin {c}")

        d_sc = d0.groupby(group_cols, dropna=False).apply(agg_weighted_level).reset_index()
        d_sc = d_sc.dropna(subset=["posicionamiento_pond", "margen_pond"])
        d_sc = d_sc[d_sc["venta_neta_level"] > 0]

        if d_sc.empty:
            st.info("No hay datos suficientes para el scatter a este nivel.")
            fig_sc = None
        else:
            # segmento representativo por nivel (mismo criterio: mayor venta)
            rep_seg = compute_rep_segment(d0, label_col)
            if not rep_seg.empty:
                d_sc = d_sc.merge(rep_seg, on=label_col, how="left")
            else:
                d_sc["segmento_rep"] = None
                d_sc["id_segmento_rep"] = None

            d_sc["segmento_rep_norm"] = d_sc["segmento_rep"].astype(str).map(_norm_seg_name)
            d_sc["quadrant"] = d_sc["segmento_rep_norm"].map(NAME_TO_QUADRANT).astype("Int64")
            d_sc["quadrant_label"] = d_sc["quadrant"].map(QUADRANT_TO_LABEL)
            d_sc["color"] = d_sc["quadrant"].map(QUADRANT_COLOR).fillna("#7F7F7F")

            d_sc["pos_pct"] = d_sc["posicionamiento_pond"] * 100.0
            d_sc["margen_pct"] = d_sc["margen_pond"] * 100.0

            # referencias cuadrante (fijas)
            x_ref = 100.0
            y_ref = 17.72

            # ------------------------------
            # RANGOS con "zoom aireado":
            # 20% de espacio a cada lado (x) y arriba/abajo (y)
            # ------------------------------
            x_min_data = float(d_sc["pos_pct"].min())
            x_max_data = float(d_sc["pos_pct"].max())
            y_min_data = float(d_sc["margen_pct"].min())
            y_max_data = float(d_sc["margen_pct"].max())

            x_range_data = (x_max_data - x_min_data)
            y_range_data = (y_max_data - y_min_data)

            if x_range_data <= 0:
                x_range_data = 10.0
            if y_range_data <= 0:
                y_range_data = 5.0

            x_margin = 0.20 * x_range_data
            y_margin = 0.20 * y_range_data

            xmin = x_min_data - x_margin
            xmax = x_max_data + x_margin
            ymin = y_min_data - y_margin
            ymax = y_max_data + y_margin

            # ------------------------------
            # tamaño marker (sqrt venta)
            # ------------------------------
            venta_vals = d_sc["venta_neta_level"].astype(float).values
            size_vals = np.sqrt(np.clip(venta_vals, 0, None))
            if np.nanmax(size_vals) > 0:
                size_vals = 10 + 45 * (size_vals / np.nanmax(size_vals))
            else:
                size_vals = np.full_like(size_vals, 18.0)

            fig_sc = go.Figure()

            # ------------------------------
            # Cuadrantes "infinitos" PERO en coords de datos (xref='x', yref='y')
            # => se mueven con pan/zoom
            # ------------------------------
            INF = 1e9  # suficientemente grande para valores en %

            def add_quad_data(x0, x1, y0, y1, q):
                fig_sc.add_shape(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor=QUADRANT_COLOR[q],
                    opacity=0.12,
                    line=dict(width=0),
                    layer="below",
                )

            # Q1 arriba-derecha, Q2 arriba-izquierda, Q3 abajo-izquierda, Q4 abajo-derecha
            add_quad_data(x_ref,  INF,  y_ref,  INF, 1)  # contribuyente
            add_quad_data(-INF,  x_ref, y_ref,  INF, 2)  # poderosa
            add_quad_data(-INF,  x_ref, -INF,  y_ref, 3) # magnética
            add_quad_data(x_ref,  INF, -INF,  y_ref, 4)  # oportunista

            # ------------------------------
            # puntos + labels
            # ------------------------------
            d_sc[label_col] = d_sc[label_col].fillna(f"Sin {label_col}")

            fig_sc.add_trace(
                go.Scatter(
                    x=d_sc["pos_pct"],
                    y=d_sc["margen_pct"],
                    mode="markers+text",
                    text=d_sc[label_col],
                    textposition="top center",
                    marker=dict(
                        size=size_vals,
                        color=d_sc["color"],
                        opacity=0.80,
                        line=dict(width=0.8, color="black"),
                    ),
                    customdata=np.stack(
                        [
                            d_sc[label_col].astype(str).values,
                            d_sc["segmento_rep"].astype(str).fillna("").values,
                            d_sc["quadrant_label"].astype(str).fillna("").values,
                            d_sc["venta_neta_level"].astype(float).values,
                            d_sc["pos_pct"].astype(float).values,
                            d_sc["margen_pct"].astype(float).values,
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        f"<b>{label_title}:</b> %{{customdata[0]}}<br>"
                        "<b>Segmento rep:</b> %{customdata[1]}<br>"
                        "<b>Cuadrante:</b> %{customdata[2]}<br>"
                        "Posicionamiento: <b>%{customdata[4]:.2f}%</b><br>"
                        "Margen: <b>%{customdata[5]:.2f}%</b><br>"
                        "Venta: <b>$%{customdata[3]:,.0f}</b><br>"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

            # líneas de corte (en coords del eje)
            fig_sc.add_vline(x=x_ref, line_width=1, line_dash="dash", opacity=0.8)
            fig_sc.add_hline(y=y_ref, line_width=1, line_dash="dash", opacity=0.8)

            # etiqueta pequeña sobre la línea horizontal
            fig_sc.add_annotation(
                x=1.0,  # derecha del gráfico
                xref="paper",  # 0..1 relativo al ancho del plot
                y=y_ref,
                yref="y",  # y en coordenadas reales
                text=f"Target: {y_ref:.2f}%",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",  # "sobre" la línea
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.25)",
                borderwidth=1,
                borderpad=2,
            )

            # anotaciones (en coords de datos; quedan dentro del rango visible que definimos)
            fig_sc.add_annotation(x=(x_ref + xmax) / 2, y=(y_ref + ymax) / 2, text="<b>Contribuyente</b>", showarrow=False, opacity=1)
            fig_sc.add_annotation(x=(xmin + x_ref) / 2, y=(y_ref + ymax) / 2, text="<b>Poderosa</b>", showarrow=False, opacity=1)
            fig_sc.add_annotation(x=(xmin + x_ref) / 2, y=(ymin + y_ref) / 2, text="<b>Magnética</b>", showarrow=False, opacity=1)
            fig_sc.add_annotation(x=(x_ref + xmax) / 2, y=(ymin + y_ref) / 2, text="<b>Oportunista</b>", showarrow=False, opacity=1)

            # layout + rangos finales (con el 20% de espacio por lado ya aplicado)
            fig_sc.update_layout(
                margin=dict(t=20, l=10, r=10, b=10),
                height=520,
                xaxis=dict(title="Posicionamiento (%)", range=[xmin, xmax]),
                yaxis=dict(title="Margen (%)", range=[ymin, ymax]),
            )

            st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown("---")


# ======================================================
# 4) Tabla pivote (última ventana)
# ======================================================
st.subheader("Reporte pivote (última ventana)")

df_pivot = df.rename(columns={"macro": "macro_categoria", "nombre": "nombre_sku"})[
    [
        "macro_categoria",
        "categoria",
        "nombre_sku",
        "venta_neta",
        "posicionamiento",
        "margen",
        "peso_venta",
    ]
].copy()


den_cat = (
    df_all.groupby(["macro", "categoria"])["sku"]
    .nunique()
    .reset_index(name="skus_total_cat")
    .rename(columns={"macro": "macro_categoria"})
)

num_cat = (
    df_pos.groupby(["macro", "categoria"])["sku"]
    .nunique()
    .reset_index(name="skus_repr_cat")
    .rename(columns={"macro": "macro_categoria"})
)

rep_cat = den_cat.merge(num_cat, on=["macro_categoria", "categoria"], how="left")
rep_cat["skus_repr_cat"] = rep_cat["skus_repr_cat"].fillna(0)
rep_cat["rep_cat"] = np.where(
    rep_cat["skus_total_cat"] > 0,
    rep_cat["skus_repr_cat"] / rep_cat["skus_total_cat"],
    np.nan
)

den_macro = (
    df_all.groupby(["macro"])["sku"]
    .nunique()
    .reset_index(name="skus_total_macro")
    .rename(columns={"macro": "macro_categoria"})
)

num_macro = (
    df_pos.groupby(["macro"])["sku"]
    .nunique()
    .reset_index(name="skus_repr_macro")
    .rename(columns={"macro": "macro_categoria"})
)

rep_macro = den_macro.merge(num_macro, on=["macro_categoria"], how="left")
rep_macro["skus_repr_macro"] = rep_macro["skus_repr_macro"].fillna(0)
rep_macro["rep_macro"] = np.where(
    rep_macro["skus_total_macro"] > 0,
    rep_macro["skus_repr_macro"] / rep_macro["skus_total_macro"],
    np.nan
)

df_pivot = df_pivot.merge(
    rep_cat[["macro_categoria", "categoria", "skus_total_cat", "skus_repr_cat", "rep_cat"]],
    on=["macro_categoria", "categoria"],
    how="left"
)
df_pivot = df_pivot.merge(
    rep_macro[["macro_categoria", "skus_total_macro", "skus_repr_macro", "rep_macro"]],
    on=["macro_categoria"],
    how="left"
)
df_pivot["representatividad"] = df_pivot["rep_cat"]

pivot_venta_total = float(df_pivot["venta_neta"].sum(skipna=True) or 0.0)
pivot_peso_total = float(df_pivot["peso_venta"].sum(skipna=True) or 0.0)

if pivot_peso_total > 0:
    pivot_pos_total = float((df_pivot["posicionamiento"] * df_pivot["peso_venta"]).sum(skipna=True) / pivot_peso_total)
    pivot_margen_total = float((df_pivot["margen"] * df_pivot["peso_venta"]).sum(skipna=True) / pivot_peso_total)
else:
    pivot_pos_total = None
    pivot_margen_total = None

total_skus_ventana = int(df_all["sku"].nunique())
total_skus_repr = int(df_pos["sku"].nunique())
pivot_repr_total = (total_skus_repr / total_skus_ventana) if total_skus_ventana else None

pivot_totals_row = {
    "macro_categoria": "TOTAL",
    "categoria": "",
    "sku": "",
    "nombre_sku": "",
    "venta_neta": pivot_venta_total,
    "posicionamiento": pivot_pos_total,
    "margen": pivot_margen_total,
    "peso_venta": pivot_peso_total,
    "representatividad": pivot_repr_total,
}

if not AGGRID_AVAILABLE:
    st.info("Instala `streamlit-aggrid` para pivote desplegable. Mostrando alternativa estática.")
    st.dataframe(df_pivot, width="stretch", height=600)
else:
    weighted_avg_agg = JsCode("""
        function(params) {
            var field = params.column.getColId();
            var rowNode = params.rowNode;
            if (!rowNode || !rowNode.allLeafChildren) return null;

            var children = rowNode.allLeafChildren;
            var sum = 0.0;
            var weightSum = 0.0;

            for (var i = 0; i < children.length; i++) {
                var data = children[i].data;
                if (!data) continue;

                var v = data[field];
                var w = data['peso_venta'];

                if (v == null || w == null) continue;

                sum += v * w;
                weightSum += w;
            }

            if (weightSum === 0) return null;
            return sum / weightSum;
        }
    """)

    totals_row_style = JsCode("""
        function(params) {
            if (params.data && params.data.macro_categoria === 'TOTAL') {
                return {'fontWeight':'bold','backgroundColor':'#F2F2F2','color':'black'};
            }
            return {};
        }
    """)

    posicionamiento_cell_style = JsCode("""
        function(params) {
            if (params.data && params.data.macro_categoria === 'TOTAL') {
                return {'fontWeight':'bold','backgroundColor':'#F2F2F2','color':'black'};
            }
            var v = null;
            if (params.data && params.data.posicionamiento != null) v = params.data.posicionamiento;
            else if (params.value != null) v = params.value;
            if (v == null) return {};

            var minVal = 0.5;
            var midVal = 1.0;
            var maxVal = 2.0;

            function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
            function lerp(a, b, t)  { return a + (b - a) * t; }
            function hexToRgb(hex) {
                var h = hex.replace('#','');
                var bigint = parseInt(h, 16);
                return { r: (bigint >> 16) & 255, g: (bigint >> 8) & 255, b: bigint & 255 };
            }

            var red   = hexToRgb('#F8696B');
            var green = hexToRgb('#63BE7B');
            var white = {r: 255, g: 255, b: 255};
            var c;

            if (v < midVal) {
                var t = clamp((midVal - v) / (midVal - minVal), 0, 1);
                c = { r: Math.round(lerp(white.r, green.r, t)),
                      g: Math.round(lerp(white.g, green.g, t)),
                      b: Math.round(lerp(white.b, green.b, t)) };
            } else if (v > midVal) {
                var t = clamp((v - midVal) / (maxVal - midVal), 0, 1);
                c = { r: Math.round(lerp(white.r, red.r, t)),
                      g: Math.round(lerp(white.g, red.g, t)),
                      b: Math.round(lerp(white.b, red.b, t)) };
            } else {
                c = white;
            }

            return { 'backgroundColor': 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')', 'color': 'black' };
        }
    """)

    peso_venta_cell_style = JsCode("""
        function(params) {
            if (params.data && params.data.macro_categoria === 'TOTAL') {
                return {'fontWeight':'bold','backgroundColor':'#F2F2F2','color':'black'};
            }
            var v = (params.data && params.data.peso_venta != null) ? params.data.peso_venta : params.value;
            if (v == null) return {};

            var scale = v / 0.05;
            scale = Math.max(0, Math.min(1, scale));

            var yellow = {r: 246, g: 227, b: 122};
            var white  = {r: 255, g: 255, b: 255};

            function lerp(a, b, t) { return a + (b - a) * t; }

            var c = {
                r: Math.round(lerp(white.r, yellow.r, scale)),
                g: Math.round(lerp(white.g, yellow.g, scale)),
                b: Math.round(lerp(white.b, yellow.b, scale))
            };

            return { 'backgroundColor': 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')', 'color': 'black' };
        }
    """)

    rep_cell_style = JsCode("""
        function(params) {
            if (params.data && params.data.macro_categoria === 'TOTAL') {
                return {'fontWeight':'bold','backgroundColor':'#F2F2F2','color':'black'};
            }
            var v = (params.data && params.data.representatividad != null) ? params.data.representatividad : params.value;
            if (v == null) return {};

            var scale = v / 1;
            scale = Math.max(0, Math.min(1, scale));

            var yellow = {r: 246, g: 227, b: 122};
            var white  = {r: 255, g: 255, b: 255};

            function lerp(a, b, t) { return a + (b - a) * t; }

            var c = {
                r: Math.round(lerp(white.r, yellow.r, scale)),
                g: Math.round(lerp(white.g, yellow.g, scale)),
                b: Math.round(lerp(white.b, yellow.b, scale))
            };

            return { 'backgroundColor': 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')', 'color': 'black' };
        }
    """)

    margen_cell_style = JsCode("""
        function(params) {
            if (params.data && params.data.macro_categoria === 'TOTAL') {
                return {'fontWeight':'bold','backgroundColor':'#F2F2F2','color':'black'};
            }
            var v = null;
            if (params.data && params.data.margen != null) v = params.data.margen;
            else if (params.value != null) v = params.value;
            if (v == null) return {};

            var minVal = -1.0;
            var midVal =  0.0;
            var maxVal =  1.0;

            function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
            function lerp(a, b, t)  { return a + (b - a) * t; }
            function hexToRgb(hex) {
                var h = hex.replace('#','');
                var bigint = parseInt(h, 16);
                return { r: (bigint >> 16) & 255, g: (bigint >> 8) & 255, b: bigint & 255 };
            }

            var red   = hexToRgb('#F8696B');
            var green = hexToRgb('#63BE7B');
            var white = {r: 255, g: 255, b: 255};
            var c;

            if (v < midVal) {
                var t = clamp((midVal - v) / (midVal - minVal), 0, 1);
                c = { r: Math.round(lerp(white.r, red.r, t)),
                      g: Math.round(lerp(white.g, red.g, t)),
                      b: Math.round(lerp(white.b, red.b, t)) };
            } else if (v > midVal) {
                var t = clamp((v - midVal) / (maxVal - midVal), 0, 1);
                c = { r: Math.round(lerp(white.r, green.r, t)),
                      g: Math.round(lerp(white.g, green.g, t)),
                      b: Math.round(lerp(white.b, green.b, t)) };
            } else {
                c = white;
            }

            return { 'backgroundColor': 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')', 'color': 'black' };
        }
    """)

    rep_level_agg = JsCode("""
    function(params) {
        var rowNode = params.rowNode;
        if (!rowNode) return null;

        if (!rowNode.group) {
            return params.values && params.values.length ? params.values[0] : null;
        }

        if (!rowNode.allLeafChildren || rowNode.allLeafChildren.length === 0) return null;
        var d = rowNode.allLeafChildren[0].data;
        if (!d) return null;

        if (rowNode.level === 0) {
            var a = d.skus_repr_macro;
            var b = d.skus_total_macro;
            if (a == null || b == null || b === 0) return null;
            return a / b;
        }

        if (rowNode.level === 1) {
            var a = d.skus_repr_cat;
            var b = d.skus_total_cat;
            if (a == null || b == null || b === 0) return null;
            return a / b;
        }

        return null;
    }
    """)

    gb = GridOptionsBuilder.from_dataframe(df_pivot)
    gb.configure_grid_options(aggFuncs={"weightedAvg": weighted_avg_agg, "repLevel": rep_level_agg})

    gb.configure_default_column(
        groupable=True,
        value=True,
        enableRowGroup=True,
        sortable=True,
        filter=True,
        editable=False,
        resizable=True,
    )

    gb.configure_column("macro_categoria", rowGroup=True, hide=True)
    gb.configure_column("categoria", rowGroup=True, hide=True)
    gb.configure_column("nombre_sku", rowGroup=True, hide=True)

    aux_cols = [
        "skus_total_cat", "skus_repr_cat", "rep_cat",
        "skus_total_macro", "skus_repr_macro", "rep_macro",
    ]
    for c in aux_cols:
        if c in df_pivot.columns:
            gb.configure_column(c, hide=True)

    gb.configure_column(
        "venta_neta",
        header_name="Venta SKU",
        type=["numericColumn"],
        aggFunc="sum",
        valueFormatter=(
            "value == null ? '' : "
            "'$' + value.toLocaleString('es-CL', {minimumFractionDigits: 0, maximumFractionDigits: 0})"
        ),
        cellStyle=totals_row_style,
    )

    gb.configure_column(
        "posicionamiento",
        header_name="Posicionamiento",
        type=["numericColumn"],
        aggFunc="weightedAvg",
        valueFormatter="value == null ? '' : (Number(value) * 100).toFixed(2) + '%'",
        cellStyle=posicionamiento_cell_style,
    )

    gb.configure_column(
        "margen",
        header_name="Margen (front + back)",
        type=["numericColumn"],
        aggFunc="weightedAvg",
        valueFormatter="value == null ? '' : (Number(value) * 100).toFixed(2) + '%'",
        cellStyle=margen_cell_style,
    )

    gb.configure_column(
        "peso_venta",
        header_name="Peso venta",
        type=["numericColumn"],
        aggFunc="sum",
        valueFormatter="value == null ? '' : (Number(value) * 100).toFixed(4) + '%'",
        cellStyle=peso_venta_cell_style,
    )

    gb.configure_column(
        "representatividad",
        header_name="Representatividad (macro/cat)",
        type=["numericColumn"],
        aggFunc="repLevel",
        valueFormatter="value == null ? '' : (Number(value) * 100).toFixed(2) + '%'",
        cellStyle=rep_cell_style,
    )

    grid_options = gb.build()
    grid_options["pinnedBottomRowData"] = [pivot_totals_row]
    grid_options["getRowStyle"] = totals_row_style.js_code
    grid_options["groupDefaultExpanded"] = 0
    grid_options["autoGroupColumnDef"] = {"headerName": "Macro / Categoría / SKU"}

    AgGrid(
        df_pivot,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        height=375,
    )

st.markdown("---")


# ======================================================
# 5) Mini gráficos evolución por macrocategoría
# ======================================================
mini_macro_figs = {}
st.subheader("Evolución por macrocategoría")

if df_macro_line.empty:
    st.info("No hay datos macro para mini gráficos.")
else:
    df_macro_line["macro"] = df_macro_line["macro"].fillna("Sin macro")
    df_macro_line["bloque"] = pd.Categorical(
        df_macro_line["bloque"],
        categories=[b["label"] for b in blocks_eff],
        ordered=True,
    )
    macros = sorted(df_macro_line["macro"].unique().tolist())

    ncols = 3
    rows = (len(macros) + ncols - 1) // ncols
    idx = 0

    mini_macro_figs = {}
    for r in range(rows):
        cols = st.columns(ncols)
        for c in range(ncols):
            if idx >= len(macros):
                break

            m = macros[idx]
            idx += 1

            dmm = df_macro_line[df_macro_line["macro"] == m].copy().sort_values("bloque")

            fig_m = make_subplots(specs=[[{"secondary_y": True}]])

            fig_m.add_trace(
                go.Scatter(
                    x=dmm["bloque"],
                    y=dmm["pos_macro"],
                    mode="lines+markers",
                    name="POS",
                    hovertemplate="Ventana: %{x}<br>POS: %{y:.4f}<extra></extra>",
                ),
                secondary_y=False,
            )

            fig_m.add_trace(
                go.Scatter(
                    x=dmm["bloque"],
                    y=dmm["venta_macro"],
                    mode="lines+markers",
                    name="Venta",
                    hovertemplate="Ventana: %{x}<br>Venta: $%{y:,.0f}<extra></extra>",
                    line=dict(color="#ff0000"),
                    marker=dict(color="#ff0000"),
                ),
                secondary_y=True,
            )

            fig_m.add_hline(y=1.0, line_width=1, line_dash="dash", opacity=0.7)

            fig_m.update_layout(
                title=m,
                height=260,
                margin=dict(t=50, l=10, r=10, b=10),
                showlegend=False,
            )
            fig_m.update_yaxes(title_text="POS", secondary_y=False)
            fig_m.update_yaxes(title_text="CLP", secondary_y=True, tickformat=",.0f")

            mini_macro_figs[m] = fig_m

            with cols[c]:
                st.plotly_chart(fig_m, use_container_width=True)

st.markdown("---")


# ======================================================
# 6) Tabla pivote evolución
# ======================================================
st.subheader("Tabla pivote (macro / categoría / SKU) — evolución")

df_long = pd.concat(snapshots, ignore_index=True) if len(snapshots) else pd.DataFrame()

if df_long.empty:
    st.warning("No hay datos para las ventanas seleccionadas.")
else:
    df_long["sku"] = df_long["sku"].astype(str)
    df_long["macro"] = df_long["macro"].fillna("Sin macro")
    df_long["categoria"] = df_long["categoria"].fillna("Sin categoría")
    df_long["nombre"] = df_long["nombre"].fillna("")

    pos_wide = df_long.pivot_table(
        index=["macro", "categoria", "sku", "nombre"],
        columns="window_label",
        values="posicionamiento",
        aggfunc="first",
    )

    venta_wide = df_long.pivot_table(
        index=["macro", "categoria", "sku", "nombre"],
        columns="window_label",
        values="venta_neta",
        aggfunc="first",
    )

    pos_wide = pos_wide.reset_index()
    venta_wide = venta_wide.reset_index()

    def col_safe(label: str) -> str:
        return label.replace(" → ", "_").replace("-", "")

    pos_cols_map = {lbl: f"pos_{col_safe(lbl)}" for lbl in pos_wide.columns if lbl not in ["macro", "categoria", "sku", "nombre"]}
    venta_cols_map = {lbl: f"venta_{col_safe(lbl)}" for lbl in venta_wide.columns if lbl not in ["macro", "categoria", "sku", "nombre"]}

    pos_wide = pos_wide.rename(columns=pos_cols_map)
    venta_wide = venta_wide.rename(columns=venta_cols_map)

    df_pivot_evo = pos_wide.merge(
        venta_wide,
        on=["macro", "categoria", "sku", "nombre"],
        how="left",
    )

    df_pivot_evo = df_pivot_evo.sort_values(["macro", "categoria", "sku"], ascending=True).reset_index(drop=True)

    # === IMPORTANTE: ordenar por ventanas efectivas ===
    ordered_labels = [b["label"] for b in blocks_eff]
    ordered_pos_cols = [pos_cols_map[lbl] for lbl in ordered_labels if lbl in pos_cols_map]
    ordered_venta_cols = [venta_cols_map[lbl] for lbl in ordered_labels if lbl in venta_cols_map]

    df_pivot_show = df_pivot_evo.rename(columns={"nombre": "nombre_sku"}).copy()
    df_pivot_show["macro_categoria"] = df_pivot_show["macro"]
    df_pivot_show["categoria"] = df_pivot_show["categoria"]

    front_cols = ["macro_categoria", "categoria", "sku", "nombre_sku"]
    df_pivot_show = df_pivot_show[front_cols + ordered_pos_cols + ordered_venta_cols].copy()

    if not AGGRID_AVAILABLE:
        st.info("Instala `streamlit-aggrid` para pivote desplegable. Mostrando alternativa estática.")
        st.dataframe(df_pivot_show[front_cols + ordered_pos_cols], width="stretch", height=650)
    else:
        weighted_pos_by_window = JsCode("""
            function(params) {
                var field = params.column.getColId(); // pos_...
                var rowNode = params.rowNode;
                if (!rowNode || !rowNode.allLeafChildren) return null;

                var weightField = field.replace(/^pos_/, 'venta_');

                var children = rowNode.allLeafChildren;
                var sum = 0.0;
                var wsum = 0.0;

                for (var i = 0; i < children.length; i++) {
                    var data = children[i].data;
                    if (!data) continue;

                    var v = data[field];
                    var w = data[weightField];

                    if (v == null || w == null) continue;

                    sum += (Number(v) * Number(w));
                    wsum += Number(w);
                }

                if (wsum === 0) return null;
                return sum / wsum;
            }
        """)

        pos_cell_style = JsCode("""
            function(params) {
                var v = params.value;
                if (v == null) return {};

                var minVal = 0.5;
                var midVal = 1.0;
                var maxVal = 2.0;

                function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
                function lerp(a, b, t)  { return a + (b - a) * t; }
                function hexToRgb(hex) {
                    var h = hex.replace('#','');
                    var bigint = parseInt(h, 16);
                    return { r: (bigint >> 16) & 255, g: (bigint >> 8) & 255, b: bigint & 255 };
                }

                var red   = hexToRgb('#F8696B');
                var green = hexToRgb('#63BE7B');
                var white = {r: 255, g: 255, b: 255};
                var c;

                if (v < midVal) {
                    var t = clamp((midVal - v) / (midVal - minVal), 0, 1);
                    c = { r: Math.round(lerp(white.r, green.r, t)),
                          g: Math.round(lerp(white.g, green.g, t)),
                          b: Math.round(lerp(white.b, green.b, t)) };
                } else if (v > midVal) {
                    var t = clamp((v - midVal) / (maxVal - midVal), 0, 1);
                    c = { r: Math.round(lerp(white.r, red.r, t)),
                          g: Math.round(lerp(white.g, red.g, t)),
                          b: Math.round(lerp(white.b, red.b, t)) };
                } else {
                    c = white;
                }

                return { 'backgroundColor': 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')', 'color': 'black' };
            }
        """)

        gb = GridOptionsBuilder.from_dataframe(df_pivot_show)

        gb.configure_grid_options(aggFuncs={"weightedPos": weighted_pos_by_window})

        gb.configure_default_column(
            groupable=True,
            value=True,
            enableRowGroup=True,
            sortable=True,
            filter=True,
            editable=False,
            resizable=True,
        )

        gb.configure_column("macro_categoria", rowGroup=True, hide=True)
        gb.configure_column("categoria", rowGroup=True, hide=True)

        gb.configure_column("sku", header_name="SKU", width=120)
        gb.configure_column("nombre_sku", header_name="Nombre", width=320)

        for c in ordered_pos_cols:
            gb.configure_column(
                c,
                header_name=c,
                type=["numericColumn"],
                aggFunc="weightedPos",
                valueFormatter="value == null ? '' : (Number(value) * 100).toFixed(2) + '%'",
                cellStyle=pos_cell_style,
                width=170,
            )

        for c in ordered_venta_cols:
            gb.configure_column(c, hide=True)

        grid_options = gb.build()
        grid_options["groupDefaultExpanded"] = 0
        grid_options["autoGroupColumnDef"] = {"headerName": "Macro / Categoría"}

        AgGrid(
            df_pivot_show,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=True,
            height=650,
        )

st.markdown("---")


# ======================================================
# 7) Mapa posicionamiento (última ventana)
# ======================================================
st.subheader("Mapa de posicionamiento (última ventana)")

def agg_categoria(grp: pd.DataFrame) -> pd.Series:
    venta_total = float(grp["venta_neta"].sum(skipna=True) or 0.0)
    peso_total = float(grp["peso_venta"].sum(skipna=True) or 0.0)

    if venta_total and not np.isclose(venta_total, 0):
        pos_pond = float((grp["posicionamiento"] * grp["venta_neta"]).sum(skipna=True) / venta_total)
        margen_pond = float((grp["margen"] * grp["venta_neta"]).sum(skipna=True) / venta_total)
    else:
        pos_pond = np.nan
        margen_pond = np.nan

    return pd.Series(
        {
            "venta_categoria": venta_total,
            "peso_venta_categoria": peso_total,
            "posicionamiento_pond": pos_pond,
            "margen_pond": margen_pond,
        }
    )

df_cat = df.groupby(["macro", "categoria"], dropna=False).apply(agg_categoria).reset_index()
df_ag = df_cat.rename(columns={"macro": "macro_categoria", "categoria": "categoria"})

df_tree = df_ag.copy()
df_tree = df_tree[
    df_tree["peso_venta_categoria"].notna()
    & (df_tree["peso_venta_categoria"] > 0)
    & df_tree["posicionamiento_pond"].notna()
]

fig_treemap = None

if df_tree.empty:
    st.info("No hay datos suficientes para graficar (requiere peso>0 y posicionamiento válido).")
else:
    alpha = 0.55
    size_raw = np.power(df_tree["peso_venta_categoria"].values.astype(float), alpha)
    min_frac = 0.008
    min_size = float(np.nanmax(size_raw) * min_frac) if np.nanmax(size_raw) > 0 else 1e-9
    df_tree["size_visual"] = np.maximum(size_raw, min_size)

    df_tree["pos_text"] = (df_tree["posicionamiento_pond"] * 100).round(2).astype(str) + "%"
    df_tree["peso_text"] = (df_tree["peso_venta_categoria"] * 100).round(4).astype(str) + "%"
    df_tree["venta_fmt"] = df_tree["venta_categoria"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    df_tree["margen_text"] = np.where(
        df_tree["margen_pond"].notna(),
        (df_tree["margen_pond"] * 100).round(2).astype(str) + "%",
        "N/A"
    )

    color_scale = [
        (0.0, "#63BE7B"),
        (0.5, "#FFFFFF"),
        (1.0, "#F8696B"),
    ]

    fig = px.treemap(
        df_tree,
        path=["macro_categoria", "categoria"],
        values="size_visual",
        color="posicionamiento_pond",
        color_continuous_scale=color_scale,
        color_continuous_midpoint=1.0,
    )

    customdata = np.stack(
        [
            df_tree["pos_text"].values,
            df_tree["peso_text"].values,
            df_tree["venta_fmt"].values,
            df_tree["margen_text"].values,
            df_tree["posicionamiento_pond"].values,
        ],
        axis=-1
    )

    fig.update_traces(
        customdata=customdata,
        texttemplate="%{label}<br>%{customdata[0]}<br>%{customdata[1]}",
        textposition="middle center",
        insidetextfont=dict(size=16),
        hovertemplate=(
            "<b>Categoría:</b> %{label}<br>"
            "<b>Macro:</b> %{parent}<br>"
            "Posicionamiento: <b>%{customdata[0]}</b> (ratio: %{customdata[4]:.4f})<br>"
            "Peso venta: <b>%{customdata[1]}</b><br>"
            "Venta (periodo): <b>%{customdata[2]}</b><br>"
            "Margen (front+back): <b>%{customdata[3]}</b><br>"
            "<extra></extra>"
        ),
        root_color="lightgrey",
    )

    fig.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
        height=750,
        coloraxis_colorbar=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.1, yanchor="top",
            len=0.6,
            thicknessmode="pixels",
            thickness=20,
            title=None,
            tickformat=".2f",
        ),
    )

    fig_treemap = fig
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ======================================================
# 8) Detalle por SKU (última ventana)
# ======================================================
st.subheader("Detalle por SKU (última ventana)")

colf1, colf2, colf3, colf4 = st.columns([1.2, 1.2, 1.2, 1.4])

macros = sorted([x for x in df["macro"].dropna().unique().tolist()])
cats = sorted([x for x in df["categoria"].dropna().unique().tolist()])
provs = sorted([x for x in df["proveedor"].dropna().unique().tolist()])

with colf1:
    sel_macros = st.multiselect("Macro", options=macros, default=[])
with colf2:
    if sel_macros:
        cats_dep = sorted(df[df["macro"].isin(sel_macros)]["categoria"].dropna().unique().tolist())
    else:
        cats_dep = cats
    sel_cats = st.multiselect("Categoría", options=cats_dep, default=[])
with colf3:
    df_tmp = df.copy()
    if sel_macros:
        df_tmp = df_tmp[df_tmp["macro"].isin(sel_macros)]
    if sel_cats:
        df_tmp = df_tmp[df_tmp["categoria"].isin(sel_cats)]
    provs_dep = sorted(df_tmp["proveedor"].dropna().unique().tolist())
    sel_provs = st.multiselect("Proveedor", options=provs_dep, default=[])
with colf4:
    q = st.text_input("Buscar SKU / nombre", value="").strip()

df_sku = df.copy()
if sel_macros:
    df_sku = df_sku[df_sku["macro"].isin(sel_macros)]
if sel_cats:
    df_sku = df_sku[df_sku["categoria"].isin(sel_cats)]
if sel_provs:
    df_sku = df_sku[df_sku["proveedor"].isin(sel_provs)]
if q:
    q_lower = q.lower()
    df_sku = df_sku[
        df_sku["sku"].astype(str).str.lower().str.contains(q_lower, na=False)
        | df_sku["nombre"].astype(str).str.lower().str.contains(q_lower, na=False)
    ]

df_sku = df_sku.sort_values("venta_neta", ascending=False)

venta_filtrada_sku = float(df_sku["venta_neta"].sum(skipna=True) or 0.0)
peso_total_filtrado = float(df_sku["peso_venta"].sum(skipna=True) or 0.0)

if venta_filtrada_sku > 0:
    pos_total_filtrado = float((df_sku["posicionamiento"] * df_sku["venta_neta"]).sum(skipna=True) / venta_filtrada_sku)
    margen_total_filtrado = float((df_sku["margen"] * df_sku["venta_neta"]).sum(skipna=True) / venta_filtrada_sku)
else:
    pos_total_filtrado = None
    margen_total_filtrado = None

totals_row = {
    "sku": "TOTAL",
    "nombre": "Totales (filtrado)",
    "macro": "",
    "categoria": "",
    "proveedor": "",
    "precio_chiper": None,
    "precio_lleno_competidor": None,
    "precio_descuento_competidor": None,
    "venta_neta": venta_filtrada_sku,
    "posicionamiento": pos_total_filtrado,
    "margen": margen_total_filtrado,
    "peso_venta": peso_total_filtrado,
}

df_sku_show = df_sku[
    [
        "sku",
        "nombre",
        "macro",
        "categoria",
        "proveedor",
        "precio_chiper",
        "precio_lleno_competidor",
        "precio_descuento_competidor",
        "venta_neta",
        "posicionamiento",
        "bucket_cat",
        "margen",
        "peso_venta",
    ]
].copy()


if not AGGRID_AVAILABLE:
    st.dataframe(df_sku_show, width="stretch", height=560)
else:
    totals_row_style = JsCode("""
        function(params) {
            if (params.data && params.data.sku === 'TOTAL') {
                return {'fontWeight':'bold','backgroundColor':'#F2F2F2','color':'black'};
            }
            return {};
        }
    """)

    posicionamiento_cell_style = JsCode("""
        function(params) {
            if (params.data && params.data.sku === 'TOTAL') {
                return {'fontWeight':'bold','backgroundColor':'#F2F2F2','color':'black'};
            }
            var v = params.value;
            if (v == null) return {};

            var minVal = 0.5;
            var midVal = 1.0;
            var maxVal = 2.0;

            function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
            function lerp(a, b, t)  { return a + (b - a) * t; }
            function hexToRgb(hex) {
                var h = hex.replace('#','');
                var bigint = parseInt(h, 16);
                return { r: (bigint >> 16) & 255, g: (bigint >> 8) & 255, b: bigint & 255 };
            }

            var red   = hexToRgb('#F8696B');
            var green = hexToRgb('#63BE7B');
            var white = {r: 255, g: 255, b: 255};
            var c;

            if (v < midVal) {
                var t = clamp((midVal - v) / (midVal - minVal), 0, 1);
                c = { r: Math.round(lerp(white.r, green.r, t)),
                      g: Math.round(lerp(white.g, green.g, t)),
                      b: Math.round(lerp(white.b, green.b, t)) };
            } else if (v > midVal) {
                var t = clamp((v - midVal) / (maxVal - midVal), 0, 1);
                c = { r: Math.round(lerp(white.r, red.r, t)),
                      g: Math.round(lerp(white.g, red.g, t)),
                      b: Math.round(lerp(white.b, red.b, t)) };
            } else {
                c = white;
            }

            return { 'backgroundColor': 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')', 'color': 'black' };
        }
    """)

    margen_cell_style = JsCode("""
        function(params) {
            if (params.data && params.data.sku === 'TOTAL') {
                return {'fontWeight':'bold','backgroundColor':'#F2F2F2','color':'black'};
            }
            var v = params.value;
            if (v == null) return {};

            var minVal = -1.0;
            var midVal =  0.0;
            var maxVal =  1.0;

            function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
            function lerp(a, b, t)  { return a + (b - a) * t; }
            function hexToRgb(hex) {
                var h = hex.replace('#','');
                var bigint = parseInt(h, 16);
                return { r: (bigint >> 16) & 255, g: (bigint >> 8) & 255, b: bigint & 255 };
            }

            var red   = hexToRgb('#F8696B');
            var green = hexToRgb('#63BE7B');
            var white = {r: 255, g: 255, b: 255};
            var c;

            if (v < midVal) {
                var t = clamp((midVal - v) / (midVal - minVal), 0, 1);
                c = { r: Math.round(lerp(white.r, red.r, t)),
                      g: Math.round(lerp(white.g, red.g, t)),
                      b: Math.round(lerp(white.b, red.b, t)) };
            } else if (v > midVal) {
                var t = clamp((v - midVal) / (maxVal - midVal), 0, 1);
                c = { r: Math.round(lerp(white.r, green.r, t)),
                      g: Math.round(lerp(white.g, green.g, t)),
                      b: Math.round(lerp(white.b, green.b, t)) };
            } else {
                c = white;
            }

            return { 'backgroundColor': 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')', 'color': 'black' };
        }
    """)

    peso_venta_cell_style = JsCode("""
        function(params) {
            if (params.data && params.data.sku === 'TOTAL') {
                return {'fontWeight':'bold','backgroundColor':'#F2F2F2','color':'black'};
            }
            var v = params.value;
            if (v == null) return {};

            var scale = v / 0.05;
            scale = Math.max(0, Math.min(1, scale));

            var yellow = {r: 246, g: 227, b: 122};
            var white  = {r: 255, g: 255, b: 255};

            function lerp(a, b, t) { return a + (b - a) * t; }

            var c = {
                r: Math.round(lerp(white.r, yellow.r, scale)),
                g: Math.round(lerp(white.g, yellow.g, scale)),
                b: Math.round(lerp(white.b, yellow.b, scale))
            };

            return { 'backgroundColor': 'rgb(' + c.r + ',' + c.g + ',' + c.b + ')', 'color': 'black' };
        }
    """)

    gb = GridOptionsBuilder.from_dataframe(df_sku_show)
    gb.configure_default_column(sortable=True, filter=True, resizable=True, editable=False)

    gb.configure_column("sku", header_name="SKU", width=110)
    gb.configure_column("nombre", header_name="Nombre", width=280)
    gb.configure_column("macro", header_name="Macro", width=180)
    gb.configure_column("categoria", header_name="Categoría", width=180)
    gb.configure_column("proveedor", header_name="Proveedor", width=180)

    gb.configure_column(
        "precio_chiper",
        header_name="Precio Chiper",
        type=["numericColumn"],
        valueFormatter="value == null ? '' : '$' + Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0})",
        cellStyle=totals_row_style,
    )
    gb.configure_column(
        "precio_lleno_competidor",
        header_name="Precio lleno comp.",
        type=["numericColumn"],
        valueFormatter="value == null ? '' : '$' + Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0})",
        cellStyle=totals_row_style,
    )
    gb.configure_column(
        "precio_descuento_competidor",
        header_name="Precio desc. comp.",
        type=["numericColumn"],
        valueFormatter="value == null ? '' : '$' + Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0})",
        cellStyle=totals_row_style,
    )

    gb.configure_column(
        "venta_neta",
        header_name="Venta",
        type=["numericColumn"],
        valueFormatter="value == null ? '' : '$' + Number(value).toLocaleString('es-CL', {maximumFractionDigits: 0})",
        cellStyle=totals_row_style,
    )

    gb.configure_column(
        "posicionamiento",
        header_name="Posicionamiento",
        type=["numericColumn"],
        valueFormatter="value == null ? '' : (Number(value) * 100).toFixed(2) + '%'",
        cellStyle=posicionamiento_cell_style,
    )
    gb.configure_column(
        "margen",
        header_name="Margen (front+back)",
        type=["numericColumn"],
        valueFormatter="value == null ? '' : (Number(value) * 100).toFixed(2) + '%'",
        cellStyle=margen_cell_style,
    )

    gb.configure_column(
        "peso_venta",
        header_name="Peso venta",
        type=["numericColumn"],
        valueFormatter="value == null ? '' : (Number(value) * 100).toFixed(4) + '%'",
        cellStyle=peso_venta_cell_style,
    )

    grid_options = gb.build()
    grid_options["pinnedBottomRowData"] = [totals_row]
    grid_options["getRowStyle"] = totals_row_style.js_code

    AgGrid(
        df_sku_show,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
        height=1000,
    )

st.markdown("---")


# ======================================================
# PDF (HTML/CSS → PDF Playwright, SVG Plotly embebido) — 1:1
# ======================================================

def _agg_macro_table_pdf(grp: pd.DataFrame) -> pd.Series:
    venta_total = float(grp["venta_neta"].sum(skipna=True) or 0.0)
    if venta_total and not np.isclose(venta_total, 0):
        pos_pond = float((grp["posicionamiento"] * grp["venta_neta"]).sum(skipna=True) / venta_total)
        margen_pond = float((grp["margen"] * grp["venta_neta"]).sum(skipna=True) / venta_total)
    else:
        pos_pond = np.nan
        margen_pond = np.nan
    return pd.Series(
        {
            "venta_neta_macro": venta_total,
            "posicionamiento_pond": pos_pond,
            "margen_pond": margen_pond,
        }
    )

df_macro_table = (
    df.groupby("macro", dropna=False)
      .apply(_agg_macro_table_pdf)
      .reset_index()
)

df_macro_table["peso_venta_macro"] = (
    df.groupby("macro", dropna=False)["peso_venta"]
      .sum(min_count=1)
      .reset_index(drop=True)
)

if "rep_macro" in locals() and rep_macro is not None and not rep_macro.empty:
    df_macro_table = df_macro_table.merge(
        rep_macro[["macro_categoria", "rep_macro"]].rename(
            columns={"macro_categoria": "macro", "rep_macro": "representatividad_macro"}
        ),
        on="macro",
        how="left",
    )
else:
    df_macro_table["representatividad_macro"] = np.nan

df_macro_table["Macro"] = df_macro_table["macro"].fillna("Sin macro")
df_macro_table = df_macro_table.drop(columns=["macro"], errors="ignore")
df_macro_table = df_macro_table[
    ["Macro", "venta_neta_macro", "peso_venta_macro", "posicionamiento_pond", "margen_pond", "representatividad_macro"]
].copy()


def _agg_cat_table_pdf(grp: pd.DataFrame) -> pd.Series:
    venta_total = float(grp["venta_neta"].sum(skipna=True) or 0.0)
    peso_total = float(grp["peso_venta"].sum(skipna=True) or 0.0)
    if venta_total and not np.isclose(venta_total, 0):
        pos_pond = float((grp["posicionamiento"] * grp["venta_neta"]).sum(skipna=True) / venta_total)
        margen_pond = float((grp["margen"] * grp["venta_neta"]).sum(skipna=True) / venta_total)
    else:
        pos_pond = np.nan
        margen_pond = np.nan
    return pd.Series(
        {
            "venta_neta_categoria": venta_total,
            "peso_venta_categoria": peso_total,
            "posicionamiento_pond": pos_pond,
            "margen_pond": margen_pond,
        }
    )

df_cat_table = (
    df.groupby(["macro", "categoria"], dropna=False)
      .apply(_agg_cat_table_pdf)
      .reset_index()
)

if "rep_cat" in locals() and rep_cat is not None and not rep_cat.empty:
    df_cat_table = df_cat_table.merge(
        rep_cat[["macro_categoria", "categoria", "rep_cat"]].rename(
            columns={"macro_categoria": "macro", "rep_cat": "representatividad_categoria"}
        ),
        on=["macro", "categoria"],
        how="left",
    )
else:
    df_cat_table["representatividad_categoria"] = np.nan

df_cat_table["Macro"] = df_cat_table["macro"].fillna("Sin macro")
df_cat_table["Categoria"] = df_cat_table["categoria"].fillna("Sin categoría")

df_cat_table = df_cat_table.drop(columns=["macro", "categoria"], errors="ignore")
df_cat_table = df_cat_table[
    ["Macro", "Categoria", "venta_neta_categoria", "peso_venta_categoria", "posicionamiento_pond", "margen_pond", "representatividad_categoria"]
].copy()

df_pivot_last_pdf = df_pivot.copy() if "df_pivot" in locals() else None

df_pivot_evo_pos_pdf = None
if "df_pivot_show" in locals() and df_pivot_show is not None and not df_pivot_show.empty:
    pos_cols = [c for c in df_pivot_show.columns if str(c).startswith("pos_")]
    base_cols = [c for c in ["macro_categoria", "categoria", "sku", "nombre_sku"] if c in df_pivot_show.columns]
    if base_cols and pos_cols:
        df_pivot_evo_pos_pdf = df_pivot_show[base_cols + pos_cols].copy()

df_sku_pdf = df_sku_show.copy() if "df_sku_show" in locals() else None

last_venta = float(df_line["venta_total"].iloc[-1]) if not df_line.empty else np.nan

params_pdf = {
    "Competidor": COMPETIDORES.get(id_competidor, str(id_competidor)),
    "Fecha base": fecha_base.strftime("%Y-%m-%d"),
    "Ventana Chiper": f"{preset_label_ch} ({int(ventana_chiper)} días)",
    "Ventana Competidor": f"{preset_label_comp} ({int(ventana_comp)} días)",
    "N ventanas evolución (efectivas)": str(n_bloques_eff),
    "N ventanas evolución (solicitadas)": str(int(n_bloques)),
    "Filtro POS 0.5–2.0": "ON" if aplicar_filtro_pos else "OFF",
    "Última ventana Chiper": blocks_eff[-1]["label"] if blocks_eff else "",
}

kpis_pdf = {
    "venta_total_periodo": venta_total_periodo,
    "venta_total_representada": venta_total_filtrada,
    "pct_venta_representada": pct_venta_representada,
    "pos_pond_total": pos_pond_total,
    "margen_pond_total": margen_pond_total,
    "representatividad": representatividad,
    "venta_ultima_ventana": last_venta,
    "n_ventanas": n_bloques_eff,
}

layout = ReportLayout(
    page_format="A4",
    landscape=False,
    margin_mm=6,
    title="Reporte Maestro Posicionamiento",
    subtitle=f"Fecha base {fecha_base:%Y-%m-%d}",
    chart_w=1100,
    chart_h_main=520,
    chart_h_mini=260,
    mini_cols=3,
    mini_max_items=18,
)

rpt = ChiperHtmlPdfReport(layout=layout)

rpt.add_header(badge="Pricing - Chiper BI")

rpt.add_params(
    title="Configuración",
    params=params_pdf,
    cols=2,
)

rpt.add_two_kpi_cards(
    section_title="KPIs",
    left_title="Posicionamiento (última ventana)",
    left_kpis=[
        ("Venta total (periodo completo)", rpt._fmt_clp0(kpis_pdf.get("venta_total_periodo"))),
        ("Venta total (representada)", rpt._fmt_clp0(kpis_pdf.get("venta_total_representada"))),
        ("% Venta representada", rpt._fmt_pct2(kpis_pdf.get("pct_venta_representada"))),
        ("Posicionamiento ponderado", rpt._fmt_pct2_from_ratio(kpis_pdf.get("pos_pond_total"))),
        ("Margen ponderado (front + back)", rpt._fmt_pct2_from_ratio(kpis_pdf.get("margen_pond_total"))),
        ("Representatividad SKUs", rpt._fmt_pct2(kpis_pdf.get("representatividad"))),
    ],
    right_title="Evolución (N ventanas)",
    right_kpis=[
        ("Pos. pond última", rpt._fmt_pct2_from_ratio(kpis_pdf.get("pos_pond_total"))),
        ("Venta última ventana", rpt._fmt_clp0(kpis_pdf.get("venta_ultima_ventana"))),
        ("N ventanas", rpt._fmt_int(kpis_pdf.get("n_ventanas"))),
        ("Ventana última", rpt._esc(str(params_pdf.get("Última ventana Chiper", "")))),
        ("Ventana comp", rpt._esc(str(params_pdf.get("Ventana Competidor", "")))),
        ("Filtro POS 0.5–2.0", rpt._esc(str(params_pdf.get("Filtro POS 0.5–2.0", "")))),
    ],
    left_cols=2,
    right_cols=2,
)

figs_evo = [
    ("Evolución (total)", fig_total),
    ("Posicionamiento vs Margen (Macro)", fig_sc),
]
figs_evo = [(t, f) for (t, f) in figs_evo if f is not None]

rpt.add_fig_grid(
    section_title="Evolución y relación",
    figures=figs_evo,
    height=1250,
    cols=2,
)


rpt.page_break()

rpt.add_mini_figs(
    section_title="Evolución por macrocategoría",
    mini_figs=mini_macro_figs,
    cols=3,
    max_items=18,
)

rpt.page_break()

figs_map = [("Treemap", fig_treemap)]
figs_map = [(t, f) for (t, f) in figs_map if f is not None]

rpt.add_fig_grid(
    section_title="Mapa de posicionamiento (última ventana)",
    figures=figs_map,
    cols=1,
    height=950,
)


rpt.page_break()

rpt.add_table(
    title="Tabla Macro (última ventana)",
    df=df_macro_table,
    max_rows=999,
    sticky_header=True,
    numeric_formats={
        "venta_neta_macro": "clp0",
        "peso_venta_macro": "pct4",
        "posicionamiento_pond": "pct2_from_ratio",
        "margen_pond": "pct2_from_ratio",
        "representatividad_macro": "pct2",
    },
    style_rules={
        "posicionamiento_pond": StyleRule("pos", vmin=layout.pos_min, vmid=layout.pos_mid, vmax=layout.pos_max),
        "margen_pond": StyleRule("margen", vmin=layout.margen_min, vmid=layout.margen_mid, vmax=layout.margen_max),
        "peso_venta_macro": StyleRule("peso", vmax=0.05),
        "representatividad_macro": StyleRule("rep"),
    },
    sort_by="venta_neta_macro",
    sort_desc=True,
)

rpt.add_table(
    title="Tabla Categoría (última ventana)",
    df=df_cat_table,
    max_rows=999,
    sticky_header=True,
    numeric_formats={
        "venta_neta_categoria": "clp0",
        "peso_venta_categoria": "pct4",
        "posicionamiento_pond": "pct2_from_ratio",
        "margen_pond": "pct2_from_ratio",
        "representatividad_categoria": "pct2",
    },
    style_rules={
        "posicionamiento_pond": StyleRule("pos", vmin=layout.pos_min, vmid=layout.pos_mid, vmax=layout.pos_max),
        "margen_pond": StyleRule("margen", vmin=layout.margen_min, vmid=layout.margen_mid, vmax=layout.margen_max),
        "peso_venta_categoria": StyleRule("peso", vmax=0.05),
        "representatividad_categoria": StyleRule("rep"),
    },
    sort_by="Macro",
    sort_desc=True,
)

pdf_bytes = rpt.build_pdf()

st.download_button(
    "Descargar reporte PDF",
    data=pdf_bytes,
    file_name=f"reporte_maestro_posicionamiento_{fecha_base:%Y%m%d}.pdf",
    mime="application/pdf",
)
