"""Data layer for Master Pricing.

Solo contiene: SQL/queries, limpieza, preparacion de dataframes, calculo de metricas y caching.
No tiene logica de UI ni controles Streamlit.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils.mySQLHelper import execute_mysql_query


# ---------------------------------------------------------------------------
# Helpers SQL (compatibles con distintas firmas del helper)
# ---------------------------------------------------------------------------
def _mysql_call(query: str, *, fetch: bool) -> Optional[pd.DataFrame]:
    try:
        return execute_mysql_query(query, fetch=fetch)
    except TypeError:
        if fetch:
            return execute_mysql_query(query)
        execute_mysql_query(query)
        return None


def run_sql(sql: str) -> pd.DataFrame:
    try:
        df = _mysql_call(sql, fetch=True)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error SQL: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Normalizacion y utils seguros
# ---------------------------------------------------------------------------
def norm_cat_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace(" ", "_").replace("-", "_").replace("\ufeff", "")
    return s


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


def _norm_seg_name(s: str) -> str:
    return _safe_str(s).strip().lower()


def safe_pct_change(cur: float, prev: float | None) -> float:
    if prev is None or pd.isna(prev) or prev == 0:
        return np.nan
    return (cur / prev) - 1.0


def default_focus_month(today: Optional[date] = None) -> pd.Timestamp:
    """Mes foco por defecto: mes actual del ano anterior + 1 mes."""
    d = today or date.today()
    return (pd.Timestamp(d.year - 1, d.month, 1) + pd.DateOffset(months=1)).to_period("M").to_timestamp()


# ---------------------------------------------------------------------------
# Cache de catalogos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_categoria_utt_map() -> pd.DataFrame:
    q = """
    SELECT
      id_categoria AS id_categoria_chiper,
      nombre       AS categoria_utt
    FROM categoria_utt
    """
    df = run_sql(q)
    if df.empty:
        return df
    df["categoria_utt_norm"] = df["categoria_utt"].map(norm_cat_name)
    df["id_categoria_chiper"] = pd.to_numeric(df["id_categoria_chiper"], errors="coerce")
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_categoria_dim() -> pd.DataFrame:
    q = """
    SELECT
      c.id AS id_categoria_chiper,
      c.nombre AS categoria_chiper,
      mc.nombre AS macro_categoria
    FROM categoria c
    LEFT JOIN macro_categoria mc ON mc.id = c.id_macro
    """
    df = run_sql(q)
    if df.empty:
        return df
    df["id_categoria_chiper"] = pd.to_numeric(df["id_categoria_chiper"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Lectura de CSV de foco mercado
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def read_flexible_csv_from_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("No existe el archivo CSV de mercado en el servidor.")
    try:
        return pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:  # noqa: BLE001
        pass
    for sep in [";", ","]:
        try:
            return pd.read_csv(path, sep=sep, dtype=str)
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("No se pudo leer el CSV. Revisa separador/encoding/estructura.")


def coerce_foco_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_cat_name(c) for c in df.columns]

    alias = {
        "fecha": "fecha",
        "mes": "fecha",
        "month": "fecha",
        "periodo": "fecha",
        "categoria": "categoria",
        "category": "categoria",
        "cat": "categoria",
        "marca": "marca",
        "brand": "marca",
        "venta": "venta",
        "ventas": "venta",
        "sales": "venta",
        "revenue": "venta",
        "monto": "venta",
    }
    df = df.rename(columns={c: alias.get(c, c) for c in df.columns})

    required = {"fecha", "categoria", "marca", "venta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV foco: faltan columnas requeridas: {sorted(missing)}")

    df["fecha"] = df["fecha"].astype(str).str.strip()
    df["fecha"] = pd.to_datetime(df["fecha"] + "-01", errors="coerce").dt.to_period("M").dt.to_timestamp()

    df["categoria"] = df["categoria"].astype(str)
    df["categoria_norm"] = df["categoria"].map(norm_cat_name)

    df["venta"] = pd.to_numeric(df["venta"], errors="coerce").fillna(0.0)
    return df


@st.cache_data(show_spinner=True, ttl=1800)
def load_foco_cat_table(csv_path: str) -> pd.DataFrame:
    df_raw = read_flexible_csv_from_path(csv_path)
    return coerce_foco_schema(df_raw)


# ---------------------------------------------------------------------------
# Foco de mercado (agregacion y deltas)
# ---------------------------------------------------------------------------
def _wavg_series(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = v.notna() & (w > 0)
    if not mask.any():
        return np.nan
    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def compute_foco_categoria(df_f: pd.DataFrame, sel_month: pd.Timestamp, mode: str) -> pd.DataFrame:
    """Agrega ventas por categoria UTT y calcula share/deltas. No promedia ratios; usa sumas."""
    cur = (
        df_f[df_f["fecha"] == sel_month]
        .groupby(["categoria", "categoria_norm"], as_index=False)["venta"]
        .sum()
        .rename(columns={"venta": "venta_cur"})
    )

    total_cur = float(cur["venta_cur"].sum() or 0.0)
    cur["share_cur"] = np.where(total_cur > 0, cur["venta_cur"] / total_cur, 0.0)

    cmp_month = None
    if mode.startswith("MOM"):
        cmp_month = (sel_month - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    elif mode.startswith("YOY"):
        cmp_month = (sel_month - pd.DateOffset(years=1)).to_period("M").to_timestamp()

    if cmp_month is not None:
        prev = (
            df_f[df_f["fecha"] == cmp_month]
            .groupby(["categoria", "categoria_norm"], as_index=False)["venta"]
            .sum()
            .rename(columns={"venta": "venta_prev"})
        )
        total_prev = float(prev["venta_prev"].sum() or 0.0)
        prev["share_prev"] = np.where(total_prev > 0, prev["venta_prev"] / total_prev, 0.0)

        cat_focus = cur.merge(prev[["categoria_norm", "venta_prev", "share_prev"]], on="categoria_norm", how="left")
        cat_focus["delta_pct"] = cat_focus.apply(
            lambda r: safe_pct_change(
                float(r["venta_cur"]),
                float(r["venta_prev"]) if pd.notna(r["venta_prev"]) else None,
            ),
            axis=1,
        )
        cat_focus["delta_share"] = cat_focus["share_cur"] - cat_focus["share_prev"].fillna(0.0)
    else:
        cat_focus = cur.copy()
        cat_focus[["venta_prev", "share_prev", "delta_pct", "delta_share"]] = np.nan

    df_ts = df_f.groupby(["fecha", "categoria_norm"], as_index=False)["venta"].sum()
    df_ts["m"] = df_ts["fecha"].dt.month
    month_num = int(pd.to_datetime(sel_month).month)

    def seasonal_index(cat_norm: str, m: int) -> float:
        sub = df_ts[df_ts["categoria_norm"] == cat_norm]
        if sub.empty:
            return np.nan
        mean_all = float(sub["venta"].mean())
        if mean_all == 0 or pd.isna(mean_all):
            return np.nan
        mean_m = sub[sub["m"] == m]["venta"].mean()
        if pd.isna(mean_m):
            return np.nan
        return float(mean_m / mean_all)

    cat_focus["season_si"] = cat_focus["categoria_norm"].map(lambda c: seasonal_index(c, month_num))
    cat_focus["sel_month"] = sel_month
    cat_focus["cmp_month"] = cmp_month
    return cat_focus


# ---------------------------------------------------------------------------
# Posicionamiento (query principal)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=1800)
def load_posicionamiento_categoria(
    id_competidor: int,
    fecha_str: str,
    ventana_chiper: int,
    ventana_comp: int,
    excluir_dias_sin_venta_chiper: bool,
) -> pd.DataFrame:
    join_valid_days = "JOIN valid_days vd ON vd.fecha = DATE(vc.fecha)" if excluir_dias_sin_venta_chiper else ""

    query = f"""
    WITH
    params AS (
      SELECT
        {int(id_competidor)}        AS id_competidor,
        CAST('{fecha_str}' AS DATE) AS fecha_actual,
        {int(ventana_chiper)}       AS dias_ventana_chiper,
        {int(ventana_comp)}         AS dias_ventana_comp
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
      {join_valid_days}
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
          a.id_sku,
          s.sku,
          mc.nombre AS macro,
          c.id      AS id_categoria,
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
    )

    SELECT
        id_sku,
        sku,
        macro,
        id_categoria,
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
        margen
    FROM enriched
    ORDER BY sku;
    """
    dfq = run_sql(query)
    return dfq if isinstance(dfq, pd.DataFrame) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Reglas de negocio
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def load_reglas_segmento(fecha_str: str) -> pd.DataFrame:
    q = f"""
    SELECT rn.id_segmento, rn.posicionamiento_top, rn.posicionamiento_fondo, rn.margen
    FROM regla_negocio rn
    INNER JOIN (
      SELECT id_segmento, MAX(fecha) AS max_fecha
      FROM regla_negocio
      WHERE fecha <= CAST('{fecha_str}' AS DATE)
      GROUP BY id_segmento
    ) latest ON rn.id_segmento = latest.id_segmento AND rn.fecha = latest.max_fecha
    """
    d = run_sql(q)
    return d if isinstance(d, pd.DataFrame) else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600)
def load_overrides_sku(fecha_str: str) -> pd.DataFrame:
    q = f"""
    SELECT ro.id_sku, ro.posicionamiento_top, ro.posicionamiento_fondo, ro.margen
    FROM regla_negocio_override ro
    INNER JOIN (
      SELECT id_sku, MAX(fecha) AS max_fecha
      FROM regla_negocio_override
      WHERE fecha <= CAST('{fecha_str}' AS DATE)
      GROUP BY id_sku
    ) latest ON ro.id_sku = latest.id_sku AND ro.fecha = latest.max_fecha
    """
    d = run_sql(q)
    return d if isinstance(d, pd.DataFrame) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Transformaciones y targets por rol
# ---------------------------------------------------------------------------
def compute_rep_segment(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
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
        seg_sales.sort_values([group_col, "venta_neta", "id_segmento"], ascending=[True, False, True])
        .drop_duplicates(subset=[group_col], keep="first")
        .rename(columns={"id_segmento": "id_segmento_rep", "segmento": "segmento_rep"})
    )
    return rep_seg[[group_col, "id_segmento_rep", "segmento_rep"]].copy()


def apply_pos_filter(df_in: pd.DataFrame, apply_filter: bool) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return df_in
    if "posicionamiento" not in df_in.columns:
        return df_in
    d = df_in.copy()
    d["posicionamiento"] = pd.to_numeric(d["posicionamiento"], errors="coerce")
    d = d[d["posicionamiento"].notna()]
    if apply_filter:
        d = d[(d["posicionamiento"] >= 0.5) & (d["posicionamiento"] <= 2.0)]
    return d


def compute_bucket_top80_fondo20(df_in: pd.DataFrame, *, cat_col: str, venta_col: str) -> pd.DataFrame:
    """Clasifica TOP80/FONDO20 dentro de cada categoria usando share acumulado."""
    if df_in is None or df_in.empty:
        return df_in
    if cat_col not in df_in.columns or venta_col not in df_in.columns:
        return df_in

    d = df_in.copy()
    d[cat_col] = d[cat_col].fillna("Sin categoria")
    d[venta_col] = pd.to_numeric(d[venta_col], errors="coerce").fillna(0.0)

    d = d.sort_values([cat_col, venta_col], ascending=[True, False]).reset_index(drop=True)

    cat_total = d.groupby(cat_col, dropna=False)[venta_col].transform("sum")
    d["share_cat_sku"] = np.where(cat_total > 0, d[venta_col] / cat_total, np.nan)
    d["cum_share_cat_sku"] = d.groupby(cat_col, dropna=False)["share_cat_sku"].cumsum()
    d["bucket_cat"] = np.where(d["cum_share_cat_sku"] <= 0.80, "TOP80", "FONDO20")
    return d


def apply_role_targets_top_fondo(df_in: pd.DataFrame, *, fecha_str: str) -> pd.DataFrame:
    """Asigna objetivos por bucket TOP80/FONDO20 y reglas/overrides. No promedia ratios, usa coalesce."""
    if df_in is None or df_in.empty:
        return df_in

    d = df_in.copy()
    needed = {"id_sku", "id_categoria", "venta_neta", "posicionamiento", "margen", "id_segmento_rep"}
    if not needed.issubset(set(d.columns)):
        if "id_segmento_rep" not in d.columns:
            d["id_segmento_rep"] = np.nan
        if not needed.issubset(set(d.columns)):
            return d

    d["id_sku"] = pd.to_numeric(d["id_sku"], errors="coerce")
    d["id_categoria"] = pd.to_numeric(d["id_categoria"], errors="coerce")
    d["venta_neta"] = pd.to_numeric(d["venta_neta"], errors="coerce").fillna(0.0)
    d["id_segmento_rep"] = pd.to_numeric(d["id_segmento_rep"], errors="coerce")

    d = compute_bucket_top80_fondo20(d, cat_col="id_categoria", venta_col="venta_neta")

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
        d = d.merge(
            rn[["id_segmento", "rn_pos_top", "rn_pos_fondo", "rn_margen"]],
            left_on="id_segmento_rep",
            right_on="id_segmento",
            how="left",
        )
        d = d.drop(columns=["id_segmento"], errors="ignore")
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

    d["pos_regla_top"] = d["ov_pos_top"].combine_first(d["rn_pos_top"])
    d["pos_regla_fondo"] = d["ov_pos_fondo"].combine_first(d["rn_pos_fondo"])
    d["margen_regla"] = d["ov_margen"].combine_first(d["rn_margen"])

    d["pos_regla_top"] = pd.to_numeric(d["pos_regla_top"], errors="coerce")
    d["pos_regla_fondo"] = pd.to_numeric(d["pos_regla_fondo"], errors="coerce")
    d["margen_regla"] = pd.to_numeric(d["margen_regla"], errors="coerce")

    d["posicionamiento_rol"] = np.where(
        d["bucket_cat"] == "TOP80",
        d["pos_regla_top"],
        d["pos_regla_fondo"],
    )
    d["margen_rol"] = d["margen_regla"]
    return d


# ---------------------------------------------------------------------------
# Recomendacion de precios + impacto de venta (solo palanca PRICE)
# ---------------------------------------------------------------------------

def _safe_clip(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").clip(lower=lo, upper=hi)


def compute_price_reco(
    df: pd.DataFrame,
    *,
    delta_max: float,
    eps_low: float,
    eps_mid: float,
    eps_high: float,
    lambda_mkt: float,
    g_mkt_cap: float,
    momentum_cap: float,
    price_floor_col: str | None = None,
) -> pd.DataFrame:
    """Calcula precio recomendado y proyeccion de venta para PRICE (posicionamiento).

    - P_pos = pos_objetivo * precio_competidor (min disponible).
    - Clamp delta precio a ±delta_max y opcional piso de margen (price_floor_col).
    - Impacto de venta via elasticidades constantes (eps_low/mid/high) y g_mkt=delta_pct.
    - Momentum M solo con delta_share y season_si; se acota con momentum_cap.
    """
    if df is None or df.empty:
        return df

    d = df.copy()
    for c in ["precio_chiper", "precio_lleno_competidor", "precio_descuento_competidor", "posicionamiento_rol", "delta_pct", "delta_share", "season_si", "venta_neta"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d["precio_comp"] = d[["precio_lleno_competidor", "precio_descuento_competidor"]].min(axis=1, skipna=True)
    d["precio_cur"] = d["precio_chiper"]

    if price_floor_col and price_floor_col in d.columns:
        d[price_floor_col] = pd.to_numeric(d[price_floor_col], errors="coerce")

    d["g_mkt"] = _safe_clip(d.get("delta_pct", 0.0), -g_mkt_cap, g_mkt_cap)
    d["season_term"] = d.get("season_si", 1.0)
    d["delta_share_term"] = _safe_clip(d.get("delta_share", 0.0), -momentum_cap, momentum_cap)
    d["momentum_m"] = (1.0 + d["delta_share_term"]) * d["season_term"].fillna(1.0)
    d["momentum_m"] = d["momentum_m"].clip(lower=1.0 - momentum_cap, upper=1.0 + momentum_cap)

    def clamp_price(row) -> tuple[float | np.nan, str]:
        p0 = row.get("precio_cur")
        pcomp = row.get("precio_comp")
        pos_obj = row.get("posicionamiento_rol")
        p_floor = row.get(price_floor_col) if price_floor_col else None
        if pd.isna(p0) or pd.isna(pcomp) or pd.isna(pos_obj) or p0 <= 0 or pcomp <= 0:
            return np.nan, "HOLD"
        p_pos = pos_obj * pcomp
        p_low = p0 * (1 - delta_max)
        if p_floor is not None and pd.notna(p_floor):
            p_low = max(p_low, p_floor)
        p_high = p0 * (1 + delta_max)
        p1 = min(max(p_pos, p_low), p_high)
        if p1 < p0 * (1 - 1e-4):
            action = "PRICE DOWN"
        elif p1 > p0 * (1 + 1e-4):
            action = "PRICE UP"
        else:
            action = "HOLD"
        return p1, action

    reco = d.apply(clamp_price, axis=1, result_type="expand")
    d["precio_rec"] = reco[0]
    d["accion_precio"] = reco[1]
    d["delta_precio_pct"] = np.where(d["precio_cur"] > 0, d["precio_rec"] / d["precio_cur"] - 1.0, np.nan)

    d["q0"] = np.where(d["precio_cur"] > 0, d["venta_neta"] / d["precio_cur"], np.nan)
    d["v0"] = d["venta_neta"]

    def project(row, eps: float):
        p0 = row["precio_cur"]
        p1 = row["precio_rec"]
        q0 = row["q0"]
        g = row["g_mkt"] if pd.notna(row["g_mkt"]) else 0.0
        m = row["momentum_m"] if pd.notna(row["momentum_m"]) else 1.0
        if pd.isna(p0) or pd.isna(p1) or pd.isna(q0) or p0 <= 0 or p1 <= 0 or q0 <= 0:
            return np.nan, np.nan, np.nan
        r = p1 / p0
        q1 = q0 * (r ** eps) * ((1 + g) ** lambda_mkt)
        q1_adj = q1 * m
        v1 = p1 * q1_adj
        v0 = row.get("v0", np.nan)
        dv = v1 - v0 if pd.notna(v0) else np.nan
        dv_pct = v1 / v0 - 1.0 if pd.notna(v0) and v0 != 0 else np.nan
        return v1, dv, dv_pct

    for tag, eps in [("low", eps_low), ("mid", eps_mid), ("high", eps_high)]:
        v1, dv, dv_pct = zip(*d.apply(lambda r: project(r, eps), axis=1))
        d[f"v1_{tag}"] = pd.to_numeric(v1, errors="coerce")
        d[f"delta_v_{tag}"] = pd.to_numeric(dv, errors="coerce")
        d[f"delta_v_pct_{tag}"] = pd.to_numeric(dv_pct, errors="coerce")

    return d


# ---------------------------------------------------------------------------
# Orquestador principal: dataset maestro SKU por parametros
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=1200)
def build_master_dataset(
    *,
    id_competidor: int,
    fecha_base: date,
    ventana_chiper: int,
    ventana_comp: int,
    aplicar_filtro_pos: bool,
    excluir_dias_sin_venta_chiper: bool,
    csv_path: str,
    sel_month: pd.Timestamp,
    compare_mode: str,
    norm_seg_mapper: Optional[Callable[[str], str]] = None,
    delta_max_precio: float = 0.06,
    eps_low: float = -0.15,
    eps_mid: float = -0.25,
    eps_high: float = -0.40,
    lambda_mkt: float = 0.5,
    g_mkt_cap: float = 0.50,
    momentum_cap: float = 0.30,
) -> tuple[pd.DataFrame, dict]:
    """
    Retorna dataset maestro a nivel SKU (por mes foco) y metadatos.
    - No promedia ratios: usa sumas/ponderaciones correctas.
    - Valida columnas requeridas y entrega mensajes claros.
    - Usa cache por parametros para no repetir queries.
    """
    if sel_month is None:
        raise ValueError("sel_month es requerido")

    mode_key = compare_mode.split(" ")[0] if compare_mode else "SIN"
    mode = "SIN" if mode_key == "SIN" else mode_key

    df_foco = load_foco_cat_table(csv_path)
    cat_focus_utt = compute_foco_categoria(df_foco, pd.to_datetime(sel_month), mode)

    df_map = load_categoria_utt_map()
    df_cat = load_categoria_dim()
    if df_map.empty:
        raise ValueError("Tabla categoria_utt vacia o inaccesible para mapear foco.")

    cat_focus_m = cat_focus_utt.merge(
        df_map[["categoria_utt_norm", "id_categoria_chiper"]],
        left_on="categoria_norm",
        right_on="categoria_utt_norm",
        how="left",
    )

    total_cur_market = float(pd.to_numeric(cat_focus_m["venta_cur"], errors="coerce").fillna(0.0).sum() or 0.0)
    total_prev_market = float(pd.to_numeric(cat_focus_m["venta_prev"], errors="coerce").fillna(0.0).sum() or 0.0)

    mapped_only = cat_focus_m[cat_focus_m["id_categoria_chiper"].notna()].copy()
    mapped_only["venta_cur"] = pd.to_numeric(mapped_only["venta_cur"], errors="coerce").fillna(0.0)
    mapped_only["venta_prev"] = pd.to_numeric(mapped_only["venta_prev"], errors="coerce")
    mapped_only["season_si"] = pd.to_numeric(mapped_only["season_si"], errors="coerce")

    if mapped_only.empty:
        raise ValueError("No hay categorias UTT mapeadas a categorias Chiper para el mes seleccionado.")

    def agg_group(g: pd.DataFrame) -> pd.Series:
        venta_cur = float(g["venta_cur"].sum() or 0.0)
        venta_prev = float(g["venta_prev"].fillna(0.0).sum() or 0.0)
        season_si = _wavg_series(g["season_si"], g["venta_cur"])
        return pd.Series({"venta_cur": venta_cur, "venta_prev": venta_prev, "season_si": season_si})

    cat_focus_ch = mapped_only.groupby("id_categoria_chiper", as_index=False).apply(agg_group)
    if isinstance(cat_focus_ch, pd.DataFrame) and "id_categoria_chiper" not in cat_focus_ch.columns:
        cat_focus_ch = cat_focus_ch.reset_index()

    cat_focus_ch["share_cur"] = np.where(total_cur_market > 0, cat_focus_ch["venta_cur"] / total_cur_market, 0.0)
    cat_focus_ch["share_prev"] = np.where(total_prev_market > 0, cat_focus_ch["venta_prev"] / total_prev_market, 0.0)

    cat_focus_ch["delta_pct"] = cat_focus_ch.apply(
        lambda r: safe_pct_change(
            float(r["venta_cur"]),
            float(r["venta_prev"]) if pd.notna(r["venta_prev"]) else None,
        ),
        axis=1,
    )
    cat_focus_ch["delta_share"] = cat_focus_ch["share_cur"] - cat_focus_ch["share_prev"].fillna(0.0)

    cat_focus_ch = cat_focus_ch.merge(df_cat, on="id_categoria_chiper", how="left")

    fecha_str = pd.to_datetime(fecha_base).strftime("%Y-%m-%d")
    df_pos = load_posicionamiento_categoria(
        id_competidor=id_competidor,
        fecha_str=fecha_str,
        ventana_chiper=ventana_chiper,
        ventana_comp=ventana_comp,
        excluir_dias_sin_venta_chiper=excluir_dias_sin_venta_chiper,
    )

    if df_pos.empty:
        raise ValueError("Posicionamiento vacio para los parametros seleccionados.")

    num_cols = [
        "precio_chiper",
        "precio_lleno_competidor",
        "precio_descuento_competidor",
        "venta_neta",
        "posicionamiento",
        "margen",
    ]
    for c in num_cols:
        if c in df_pos.columns:
            df_pos[c] = pd.to_numeric(df_pos[c], errors="coerce")

    df_pos = apply_pos_filter(df_pos, aplicar_filtro_pos)

    venta_total = pd.to_numeric(df_pos["venta_neta"], errors="coerce").sum(skipna=True)
    venta_total = float(venta_total) if pd.notna(venta_total) else 0.0
    df_pos["peso_venta"] = np.where(
        venta_total > 0,
        pd.to_numeric(df_pos["venta_neta"], errors="coerce") / float(venta_total),
        np.nan,
    )

    df_out = df_pos.merge(
        cat_focus_ch[
            [
                "id_categoria_chiper",
                "categoria_chiper",
                "macro_categoria",
                "share_cur",
                "delta_share",
                "delta_pct",
                "season_si",
            ]
        ],
        left_on="id_categoria",
        right_on="id_categoria_chiper",
        how="left",
    )

    df_out["cat_key"] = df_out["macro"].fillna("Sin macro") + " | " + df_out["categoria"].fillna("Sin categoria")
    rep = compute_rep_segment(df_out, "cat_key")
    df_out = df_out.merge(rep, on="cat_key", how="left")

    norm_fn = norm_seg_mapper or _norm_seg_name
    df_out["segmento_rep_norm"] = df_out["segmento_rep"].map(norm_fn)

    name_to_quadrant = {
        "contribuyente": 1,
        "poderosa": 2,
        "magnetica": 3,
        "magnetica": 3,
        "magnetica": 3,
        "magnetica": 3,
        "magnetica": 3,
    }
    quadrant_to_label = {
        1: "Contribuyente",
        2: "Poderosa",
        3: "Magnetica",
        4: "Oportunista",
    }
    df_out["quadrant"] = df_out["segmento_rep_norm"].map(name_to_quadrant).astype("Int64")
    df_out["rol_rep"] = df_out["quadrant"].map(quadrant_to_label)

    df_out = apply_role_targets_top_fondo(df_out, fecha_str=fecha_str)

    df_out = compute_price_reco(
        df_out,
        delta_max=delta_max_precio,
        eps_low=eps_low,
        eps_mid=eps_mid,
        eps_high=eps_high,
        lambda_mkt=lambda_mkt,
        g_mkt_cap=g_mkt_cap,
        momentum_cap=momentum_cap,
    )

    df_pivot = df_out.rename(columns={"macro": "macro_categoria", "nombre": "nombre_sku"}).copy()
    df_pivot = df_pivot.loc[:, ~df_pivot.columns.duplicated()]

    drop_cols = ["segmento", "segmento_rep", "segmento_rep_norm", "quadrant", "cat_key"]
    df_pivot = df_pivot.drop(columns=[c for c in drop_cols if c in df_pivot.columns], errors="ignore")

    show_cols = [
        "macro_categoria",
        "categoria",
        "nombre",
        "proveedor",
        "rol_rep",
        "bucket_cat",
        "peso_venta",
        "posicionamiento",
        "posicionamiento_rol",
        "margen",
        "margen_rol",
        "delta_pct",
        "delta_share",
        "season_si",
        "share_cur",
        "venta_neta",
        "precio_cur",
        "precio_comp",
        "precio_rec",
        "delta_precio_pct",
        "q0",
        "v0",
        "v1_low",
        "v1_mid",
        "v1_high",
        "delta_v_low",
        "delta_v_mid",
        "delta_v_high",
        "delta_v_pct_low",
        "delta_v_pct_mid",
        "delta_v_pct_high",
        "accion_precio",
        "sku",
        "id_sku",
        "id_categoria",
    ]
    show_cols = [c for c in show_cols if c in df_pivot.columns]
    df_show = df_pivot[show_cols].copy()

    numeric_cols = [
        "venta_neta",
        "peso_venta",
        "posicionamiento",
        "margen",
        "posicionamiento_rol",
        "margen_rol",
        "share_cur",
        "delta_share",
        "delta_pct",
        "season_si",
        "precio_cur",
        "precio_comp",
        "precio_rec",
        "delta_precio_pct",
        "q0",
        "v0",
        "v1_low",
        "v1_mid",
        "v1_high",
        "delta_v_low",
        "delta_v_mid",
        "delta_v_high",
        "delta_v_pct_low",
        "delta_v_pct_mid",
        "delta_v_pct_high",
    ]
    for c in numeric_cols:
        if c in df_show.columns:
            df_show[c] = pd.to_numeric(df_show[c], errors="coerce")

    meta = {
        "total_cur_market": total_cur_market,
        "total_prev_market": total_prev_market,
        "n_utt_total": int(cat_focus_m["categoria_norm"].nunique()) if not cat_focus_m.empty else 0,
        "n_utt_mapped": int(cat_focus_m["id_categoria_chiper"].notna().sum()) if not cat_focus_m.empty else 0,
    }
    meta.update(
        {
            "delta_max_precio": delta_max_precio,
            "eps_low": eps_low,
            "eps_mid": eps_mid,
            "eps_high": eps_high,
            "lambda_mkt": lambda_mkt,
            "g_mkt_cap": g_mkt_cap,
            "momentum_cap": momentum_cap,
        }
    )
    return df_show, meta


__all__ = [
    "run_sql",
    "load_categoria_utt_map",
    "load_categoria_dim",
    "load_foco_cat_table",
    "compute_foco_categoria",
    "load_posicionamiento_categoria",
    "load_reglas_segmento",
    "load_overrides_sku",
    "compute_rep_segment",
    "apply_pos_filter",
    "compute_bucket_top80_fondo20",
    "apply_role_targets_top_fondo",
    "compute_price_reco",
    "build_master_dataset",
]
