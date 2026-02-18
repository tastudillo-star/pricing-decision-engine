# front/figures/scatter_pos_margen.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ======================================================
# Config
# ======================================================
@dataclass
class ScatterConfig:
    """Configuración del componente ScatterPosMargen."""

    # --- Column names (input df) ---
    sales_col: str = "venta_neta"
    pos_col: str = "posicionamiento"  # ratio (e.g. 1.03)
    margin_col: str = "front"  # ratio (e.g. 0.1772)

    sku_col: str = "sku"
    name_col: str = "nombre"  # optional
    macro_col: str = "macro"
    cat_col: str = "categoria"
    prov_col: str = "proveedor"

    segment_col: str = "segmento"  # optional
    segment_id_col: str = "id_segmento"  # optional

    # --- Grouping ---
    groupby_options: List[str] = field(
        default_factory=lambda: ["macro", "categoria", "proveedor", "sku"]
    )
    default_groupby: str = "macro"

    # --- Filter panel behavior ---
    filter_cols: Optional[List[str]] = None
    max_multiselect_options: int = 200

    # --- Quadrant & targets (in % units for axes) ---
    x_ref: float = 100.0
    y_ref: float = 10.00

    # Mapeo de nombre de segmento (normalizado) a color
    segment_color_map: Dict[str, str] = field(
        default_factory=lambda: {
            "contribuyente": "#2F80ED",
            "poderosa": "#9B51E0",
            "magnetica": "#F2994A",
            "magnética": "#F2994A",
            "oportunista": "#27AE60",
        }
    )
    quadrant_color: Dict[int, str] = field(
        default_factory=lambda: {
            1: "#2F80ED",  # Contribuyente (arriba derecha)
            2: "#9B51E0",  # Poderosa (arriba izquierda)
            3: "#F2994A",  # Magnética (abajo izquierda)
            4: "#27AE60",  # Oportunista (abajo derecha)
        }
    )
    quadrant_label: Dict[int, str] = field(
        default_factory=lambda: {
            1: "Contribuyente",
            2: "Poderosa",
            3: "Magnética",
            4: "Oportunista",
        }
    )
    fallback_point_color: str = "#7F7F7F"
    quad_bg_opacity: float = 0.10

    # --- Marker sizing ---
    size_min: float = 10.0
    size_max: float = 75.0

    # --- Labels ---
    show_labels: bool = True
    max_labeled_points: int = 100
    label_font_size: int = 16  # Tamaño de fuente para las etiquetas de puntos

    # --- Axis range padding ---
    axis_margin_frac: float = 0.30

    # --- Height ---
    height: int = 750

    # --- Label por agrupación: qué columna mostrar como texto en cada punto ---
    label_col_by_group: Dict[str, str] = field(
        default_factory=lambda: {
            "macro": "macro",
            "categoria": "categoria",
            "proveedor": "proveedor",
            "sku": "nombre",
        }
    )
    label_fallback_to_key: bool = True

    # --- Columna clave por agrupación ---
    group_key_col_by_group: Dict[str, str] = field(
        default_factory=lambda: {
            "macro": "macro",
            "categoria": "categoria",
            "proveedor": "proveedor",
            "sku": "sku",
        }
    )

    # --- Trail (trayectoria histórica) ---
    window_col: str = "ventana"
    show_trails: bool = True
    trail_line_width: float = 1.5
    trail_opacity: float = 1.0
    trail_marker_size: float = 5.0
    max_trail_groups: int = 100


# ======================================================
# Component
# ======================================================
class ScatterPosMargen:
    """
    Componente Streamlit para scatter de Posicionamiento vs Margen.
    - Recibe un DataFrame
    - Renderiza filtros (multiselect)
    - Agrega métricas ponderadas por nivel seleccionado
    - Retorna una figura Plotly con cuadrantes
    """

    def __init__(self, df: pd.DataFrame, config: Optional[ScatterConfig | Dict[str, Any]] = None):
        self.cfg = self._coerce_config(config)
        self._df_raw = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        self._df_norm = self._validate_and_normalize(self._df_raw)
        self._df_filtered: pd.DataFrame = pd.DataFrame()
        self._df_agg: pd.DataFrame = pd.DataFrame()

    # -----------------------------
    # Public API
    # -----------------------------
    def render(self, *, key_prefix: str = "posmargen") -> go.Figure:
        """
        Renderiza filtros + gráfico en el contexto Streamlit actual.
        Retorna la figura Plotly.
        """
        df0 = self._df_norm.copy()
        if df0.empty:
            st.info("No hay datos para graficar.")
            return go.Figure()

        # Panel de filtros
        state = self._filter_panel(df0, key_prefix=key_prefix)

        # Aplicar filtros
        df_f = self._apply_filters(df0, state)
        self._df_filtered = df_f.copy()

        # Obtener group_by del estado
        group_by = self._col_to_groupby(state.get("group_by", self._map_groupby_to_col(self.cfg.default_groupby)))
        show_trails = state.get("show_trails", self.cfg.show_trails)

        # Agregar
        df_agg = self._aggregate(df_f, group_by=group_by)
        self._df_agg = df_agg.copy()

        # Construir figura
        fig = self._build_figure(df_agg, group_by=group_by, show_trails=show_trails)
        return fig

    def get_filtered_df(self) -> pd.DataFrame:
        return self._df_filtered.copy()

    def get_agg_df(self) -> pd.DataFrame:
        return self._df_agg.copy()

    # -----------------------------
    # Config / normalize
    # -----------------------------
    def _coerce_config(self, config: Optional[ScatterConfig | Dict[str, Any]]) -> ScatterConfig:
        if config is None:
            return ScatterConfig()
        if isinstance(config, ScatterConfig):
            return config
        if isinstance(config, dict):
            base = ScatterConfig()
            for k, v in config.items():
                if hasattr(base, k):
                    setattr(base, k, v)
            return base
        return ScatterConfig()

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        if df is None or df.empty:
            return pd.DataFrame()

        d = df.copy()

        # Verificar columnas requeridas
        required = [cfg.sales_col, cfg.pos_col, cfg.margin_col, cfg.sku_col, cfg.macro_col, cfg.cat_col, cfg.prov_col]
        missing = [c for c in required if c not in d.columns]
        if missing:
            st.error(f"Faltan columnas requeridas en el df: {missing}")
            return pd.DataFrame()

        # Normalizar tipos numéricos
        for c in [cfg.sales_col, cfg.pos_col, cfg.margin_col]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        # Filtrar filas sin venta o métricas
        d = d[d[cfg.sales_col].notna() & (d[cfg.sales_col] > 0)]
        d = d[d[cfg.pos_col].notna() & d[cfg.margin_col].notna()]

        # Rellenar columnas de dimensión
        dim_fill = {
            cfg.macro_col: "Sin macro",
            cfg.cat_col: "Sin categoría",
            cfg.prov_col: "Sin proveedor",
            cfg.sku_col: "Sin SKU",
        }
        for col, val in dim_fill.items():
            d[col] = d[col].astype(str).fillna(val).replace({"nan": val, "None": val})

        # Columnas opcionales
        if cfg.name_col in d.columns:
            d[cfg.name_col] = d[cfg.name_col].astype(str).fillna("").replace({"nan": "", "None": ""})

        if cfg.segment_col in d.columns:
            d[cfg.segment_col] = d[cfg.segment_col].astype(str).fillna("").replace({"nan": "", "None": ""})
        else:
            d[cfg.segment_col] = ""

        if cfg.segment_id_col in d.columns:
            d[cfg.segment_id_col] = pd.to_numeric(d[cfg.segment_id_col], errors="coerce")
        else:
            d[cfg.segment_id_col] = np.nan

        # Columna ventana (opcional, para trails)
        if cfg.window_col in d.columns:
            d[cfg.window_col] = pd.to_numeric(d[cfg.window_col], errors="coerce")
            # Eliminar filas con ventana NaN si la columna existe
            d = d[d[cfg.window_col].notna()]

        return d.reset_index(drop=True)

    # -----------------------------
    # Filter panel
    # -----------------------------
    def _default_filter_cols(self) -> List[str]:
        cfg = self.cfg
        return [cfg.macro_col, cfg.cat_col, cfg.prov_col, cfg.segment_col]

    def _filter_panel(self, df: pd.DataFrame, *, key_prefix: str) -> Dict[str, Any]:
        cfg = self.cfg

        # Columnas de filtro
        filter_cols = cfg.filter_cols or self._default_filter_cols()
        filter_cols = [c for c in filter_cols if c in df.columns]

        # Mapeo para selectbox de agrupación
        groupby_label_map = {
            cfg.macro_col: "Macro",
            cfg.cat_col: "Categoría",
            cfg.prov_col: "Proveedor",
            cfg.sku_col: "SKU",
        }
        groupby_ui = [self._map_groupby_to_col(g) for g in cfg.groupby_options]

        # UI Header
        with st.container():
            top_cols = st.columns([1.5, 1.0, 1.0, 1.0], gap="large")

            with top_cols[0]:
                default_idx = 0
                default_col = self._map_groupby_to_col(cfg.default_groupby)
                if default_col in groupby_ui:
                    default_idx = groupby_ui.index(default_col)

                group_by = st.selectbox(
                    "Agrupar por",
                    options=groupby_ui,
                    index=default_idx,
                    key=f"{key_prefix}__group_by",
                    format_func=lambda x: groupby_label_map.get(x, str(x)),
                )

            with top_cols[1]:
                st.caption(f"Filas: {len(df):,}".replace(",", "."))

            with top_cols[2]:
                # Checkbox para mostrar/ocultar trails (solo si hay columna ventana)
                has_window_col = cfg.window_col in df.columns
                if has_window_col:
                    show_trails = st.checkbox(
                        "Mostrar trayectoria",
                        value=cfg.show_trails,
                        key=f"{key_prefix}__show_trails",
                    )
                else:
                    show_trails = False

            with top_cols[3]:
                if st.button("Limpiar filtros", key=f"{key_prefix}__clear_filters"):
                    self._clear_filter_state(filter_cols, key_prefix=key_prefix)
                    st.rerun()

        # Filtros multiselect en grid
        state: Dict[str, Any] = {"group_by": group_by, "show_trails": show_trails, "filters": {}}

        ncols = 4
        cols_ui = st.columns(ncols, gap="medium")

        for i, colname in enumerate(filter_cols):
            with cols_ui[i % ncols]:
                fkey = f"{key_prefix}__f__{colname}"
                df_opts = self._df_for_options(df, colname, filter_cols, key_prefix)
                opts = sorted(df_opts[colname].astype(str).dropna().unique().tolist())

                # Limitar opciones si hay demasiadas
                if len(opts) > cfg.max_multiselect_options:
                    opts = opts[:cfg.max_multiselect_options]

                st.multiselect(
                    colname,
                    options=opts,
                    default=st.session_state.get(fkey, []),
                    key=fkey,
                )
                state["filters"][colname] = {
                    "type": "multiselect",
                    "value": st.session_state.get(fkey),
                }

        return state

    def _map_groupby_to_col(self, groupby: str) -> str:
        cfg = self.cfg
        mapping = {
            "macro": cfg.macro_col,
            "categoria": cfg.cat_col,
            "proveedor": cfg.prov_col,
            "sku": cfg.sku_col,
        }
        return mapping.get(groupby, groupby)

    def _col_to_groupby(self, col: str) -> str:
        cfg = self.cfg
        reverse_mapping = {
            cfg.macro_col: "macro",
            cfg.cat_col: "categoria",
            cfg.prov_col: "proveedor",
            cfg.sku_col: "sku",
        }
        return reverse_mapping.get(col, col)

    def _clear_filter_state(self, filter_cols: List[str], *, key_prefix: str) -> None:
        for c in filter_cols:
            fkey = f"{key_prefix}__f__{c}"
            if fkey in st.session_state:
                del st.session_state[fkey]

    def _df_for_options(self, df: pd.DataFrame, target_col: str, filter_cols: List[str], key_prefix: str) -> pd.DataFrame:
        """Opciones en cascada: filtra por otros filtros ya aplicados."""
        d = df.copy()
        for c in filter_cols:
            if c == target_col:
                continue
            fkey = f"{key_prefix}__f__{c}"
            val = st.session_state.get(fkey, [])
            if isinstance(val, list) and len(val) > 0:
                d = d[d[c].astype(str).isin([str(x) for x in val])]
        return d

    # -----------------------------
    # Apply filters
    # -----------------------------
    def _apply_filters(self, df: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
        d = df.copy()
        filters = state.get("filters", {}) or {}

        for colname, payload in filters.items():
            val = payload.get("value", None)
            if isinstance(val, list) and len(val) > 0:
                d = d[d[colname].astype(str).isin([str(x) for x in val])]

        return d.reset_index(drop=True)

    # -----------------------------
    # Aggregate
    # -----------------------------
    def _aggregate(self, df: pd.DataFrame, *, group_by: str) -> pd.DataFrame:
        """
        Agrega a nivel group_by y retorna un df con:
          - group_key, label, segmento_rep, venta_neta_level, posicionamiento_pond, margen_pond, pos_pct, margen_pct
          - Si hay ventana: también incluye columna 'ventana' con múltiples filas por grupo
        """
        cfg = self.cfg

        if df is None or df.empty:
            return pd.DataFrame()

        group_key_col = cfg.group_key_col_by_group.get(group_by)
        if not group_key_col:
            return pd.DataFrame()

        label_col = cfg.label_col_by_group.get(group_by, group_key_col)

        d = df.copy()

        # Normalizar columnas
        for c in [cfg.pos_col, cfg.margin_col, cfg.sales_col]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        for c in [group_key_col, label_col, cfg.segment_col]:
            if c in d.columns:
                d[c] = d[c].astype(str).fillna("")

        d = d.dropna(subset=[cfg.sales_col, cfg.pos_col, cfg.margin_col])
        d = d[d[cfg.sales_col] > 0]

        if d.empty:
            return pd.DataFrame()

        # Detectar si hay columna ventana
        has_window = cfg.window_col in d.columns and d[cfg.window_col].notna().any()

        # Agregación ponderada
        def agg_weighted(grp: pd.DataFrame) -> pd.Series:
            venta_total = float(grp[cfg.sales_col].sum(skipna=True) or 0.0)
            if venta_total > 0:
                pos_pond = float((grp[cfg.pos_col] * grp[cfg.sales_col]).sum(skipna=True) / venta_total)
                margen_pond = float((grp[cfg.margin_col] * grp[cfg.sales_col]).sum(skipna=True) / venta_total)
            else:
                pos_pond = np.nan
                margen_pond = np.nan
            return pd.Series({
                "venta_neta_level": venta_total,
                "posicionamiento_pond": pos_pond,
                "margen_pond": margen_pond,
            })

        if has_window:
            # Modo con ventana: agrupar por [group_key_col, window_col]
            out = d.groupby([group_key_col, cfg.window_col], dropna=False).apply(agg_weighted).reset_index()
            out = out.rename(columns={group_key_col: "group_key", cfg.window_col: "ventana"})
        else:
            # Modo sin ventana: agrupar solo por group_key_col
            out = d.groupby([group_key_col], dropna=False).apply(agg_weighted).reset_index()
            out = out.rename(columns={group_key_col: "group_key"})

        out["group_key"] = out["group_key"].astype(str).fillna("")

        # Construir label representativo (el de mayor venta total dentro del grupo, sin considerar ventana)
        if label_col == group_key_col:
            out["label"] = out["group_key"]
        else:
            # Crear mapeo group_key -> label (tomando el de mayor venta)
            tmp = d[[group_key_col, label_col, cfg.sales_col]].copy()
            tmp[group_key_col] = tmp[group_key_col].astype(str).fillna("")
            tmp[label_col] = tmp[label_col].astype(str).fillna("")

            # Sumar ventas por group_key + label y tomar el de mayor venta total
            tmp_agg = tmp.groupby([group_key_col, label_col], dropna=False)[cfg.sales_col].sum().reset_index()
            tmp_agg = tmp_agg.sort_values(cfg.sales_col, ascending=False).drop_duplicates(subset=[group_key_col], keep="first")
            label_map = dict(zip(tmp_agg[group_key_col], tmp_agg[label_col]))

            out["label"] = out["group_key"].map(label_map)

            if cfg.label_fallback_to_key:
                out["label"] = out["label"].fillna(out["group_key"])
            out["label"] = out["label"].astype(str).fillna("")

        # Construir segmento representativo (el de mayor venta total dentro del grupo, sin considerar ventana)
        if cfg.segment_col in d.columns:
            tmp_seg = d[[group_key_col, cfg.segment_col, cfg.sales_col]].copy()
            tmp_seg = tmp_seg.rename(columns={group_key_col: "group_key"})
            tmp_seg["group_key"] = tmp_seg["group_key"].astype(str).fillna("")

            # Agrupar por group_key + segmento y sumar ventas
            seg_sales = (
                tmp_seg.groupby(["group_key", cfg.segment_col], dropna=False)[cfg.sales_col]
                .sum()
                .reset_index()
            )
            # Tomar el segmento con mayor venta por grupo
            seg_sales = seg_sales.sort_values(["group_key", cfg.sales_col], ascending=[True, False])
            seg_rep = seg_sales.drop_duplicates(subset=["group_key"], keep="first")
            seg_rep = seg_rep.rename(columns={cfg.segment_col: "segmento_rep"})[["group_key", "segmento_rep"]]

            out = out.merge(seg_rep, on="group_key", how="left")
            out["segmento_rep"] = out["segmento_rep"].astype(str).fillna("")
        else:
            out["segmento_rep"] = ""

        out = out.dropna(subset=["posicionamiento_pond", "margen_pond"])
        out = out[out["venta_neta_level"] > 0]

        out["pos_pct"] = out["posicionamiento_pond"] * 100.0
        out["margen_pct"] = out["margen_pond"] * 100.0

        return out.reset_index(drop=True)

    # -----------------------------
    # Figure
    # -----------------------------
    def _build_figure(self, df_agg: pd.DataFrame, *, group_by: str, show_trails: bool = None) -> go.Figure:
        cfg = self.cfg

        # Si no se especifica show_trails, usar el valor de configuración
        if show_trails is None:
            show_trails = cfg.show_trails

        if df_agg is None or df_agg.empty:
            fig = go.Figure()
            fig.update_layout(
                margin=dict(t=20, l=10, r=10, b=10),
                height=cfg.height,
                xaxis=dict(title="Posicionamiento (%)"),
                yaxis=dict(title="Margen (%)"),
            )
            return fig

        d = df_agg.copy()

        # Detectar si hay columna ventana
        has_window = "ventana" in d.columns and d["ventana"].notna().any()

        # Si hay ventana, separar en df_last (puntos finales) y df_all (para trails)
        if has_window:
            # Último punto por grupo (ventana máxima)
            idx_last = d.groupby("group_key")["ventana"].idxmax()
            df_last = d.loc[idx_last].reset_index(drop=True)
            df_all = d.copy()  # Para trails
        else:
            df_last = d.copy()
            df_all = None

        # Datos base (para rangos de ejes usamos todos los puntos si hay ventana)
        x_all = pd.to_numeric(d["pos_pct"], errors="coerce")
        y_all = pd.to_numeric(d["margen_pct"], errors="coerce")

        # Rangos con margen
        x_min_data, x_max_data = float(x_all.min()), float(x_all.max())
        y_min_data, y_max_data = float(y_all.min()), float(y_all.max())

        x_range = max(x_max_data - x_min_data, 10.0)
        y_range = max(y_max_data - y_min_data, 5.0)

        x_margin = cfg.axis_margin_frac * x_range
        y_margin = cfg.axis_margin_frac * y_range

        xmin, xmax = x_min_data - x_margin, x_max_data + x_margin
        ymin, ymax = y_min_data - y_margin, y_max_data + y_margin

        # Datos para puntos principales (df_last)
        x = pd.to_numeric(df_last["pos_pct"], errors="coerce")
        y = pd.to_numeric(df_last["margen_pct"], errors="coerce")
        venta = pd.to_numeric(df_last["venta_neta_level"], errors="coerce").fillna(0.0)

        # Tamaño marker (sqrt venta)
        size_vals = np.sqrt(np.clip(venta.values.astype(float), 0, None))
        max_size = np.nanmax(size_vals)
        if max_size > 0:
            size_vals = cfg.size_min + (cfg.size_max - cfg.size_min) * (size_vals / max_size)
        else:
            size_vals = np.full_like(size_vals, (cfg.size_min + cfg.size_max) / 2.0)

        group_key = df_last["group_key"].astype(str).fillna("").values
        label = df_last["label"].astype(str).fillna("").values
        segmento_rep = df_last["segmento_rep"].astype(str).fillna("").values if "segmento_rep" in df_last.columns else [""] * len(df_last)

        # Asignar color según segmento (columna segmento_rep)
        def get_segment_color(seg: str) -> str:
            """Retorna el color según el nombre del segmento."""
            seg_norm = seg.strip().lower()
            return cfg.segment_color_map.get(seg_norm, cfg.fallback_point_color)

        point_colors = [get_segment_color(seg) for seg in segmento_rep]

        # Mapeo group_key -> color para trails
        group_color_map = dict(zip(group_key, point_colors))

        # Customdata para hover
        custom = np.stack([
            group_key,
            label,
            venta.values.astype(float),
            x.values.astype(float),
            y.values.astype(float),
            segmento_rep,
        ], axis=-1)

        fig = go.Figure()

        # Cuadrantes (fondo)
        INF = 1e9

        def add_quad(x0, x1, y0, y1, fill):
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=x0, x1=x1, y0=y0, y1=y1,
                fillcolor=fill, opacity=cfg.quad_bg_opacity,
                line=dict(width=0), layer="below",
            )

        add_quad(cfg.x_ref, INF, cfg.y_ref, INF, cfg.quadrant_color.get(1, "#2F80ED"))
        add_quad(-INF, cfg.x_ref, cfg.y_ref, INF, cfg.quadrant_color.get(2, "#9B51E0"))
        add_quad(-INF, cfg.x_ref, -INF, cfg.y_ref, cfg.quadrant_color.get(3, "#F2994A"))
        add_quad(cfg.x_ref, INF, -INF, cfg.y_ref, cfg.quadrant_color.get(4, "#27AE60"))

        # Agregar etiquetas de cuadrantes en el centro de cada uno
        # Contribuyente (arriba derecha)
        fig.add_annotation(
            x=(cfg.x_ref + xmax) * 0.5,
            y=(cfg.y_ref + ymax) * 0.5,
            text=f"<b>{cfg.quadrant_label[1]}</b>",
            showarrow=False,
            font=dict(size=20, color="rgba(0,0,0,0.4)"),
            xref="x", yref="y",
        )

        # Poderosa (arriba izquierda)
        fig.add_annotation(
            x=(xmin + cfg.x_ref) * 0.5,
            y=(cfg.y_ref + ymax) * 0.5,
            text=f"<b>{cfg.quadrant_label[2]}</b>",
            showarrow=False,
            font=dict(size=20, color="rgba(0,0,0,0.4)"),
            xref="x", yref="y",
        )

        # Magnética (abajo izquierda)
        fig.add_annotation(
            x=(xmin + cfg.x_ref) * 0.5,
            y=(ymin + cfg.y_ref) * 0.5,
            text=f"<b>{cfg.quadrant_label[3]}</b>",
            showarrow=False,
            font=dict(size=20, color="rgba(0,0,0,0.4)"),
            xref="x", yref="y",
        )

        # Oportunista (abajo derecha)
        fig.add_annotation(
            x=(cfg.x_ref + xmax) * 0.5,
            y=(ymin + cfg.y_ref) * 0.5,
            text=f"<b>{cfg.quadrant_label[4]}</b>",
            showarrow=False,
            font=dict(size=20, color="rgba(0,0,0,0.4)"),
            xref="x", yref="y",
        )

        # --- Trails (trayectorias históricas) ---
        if has_window and show_trails and df_all is not None:
            df_trails: pd.DataFrame = df_all
            unique_groups = df_trails["group_key"].unique()
            draw_trails = len(unique_groups) <= cfg.max_trail_groups

            if draw_trails:
                for gkey in unique_groups:
                    grp = df_trails[df_trails["group_key"] == gkey].sort_values("ventana")
                    if len(grp) < 2:
                        continue  # No trail si solo hay un punto

                    trail_x = grp["pos_pct"].values
                    trail_y = grp["margen_pct"].values
                    trail_color = group_color_map.get(gkey, cfg.fallback_point_color)

                    # Línea de trail
                    fig.add_trace(
                        go.Scatter(
                            x=trail_x,
                            y=trail_y,
                            mode="lines+markers",
                            line=dict(
                                width=cfg.trail_line_width,
                                color=trail_color,
                            ),
                            marker=dict(
                                size=cfg.trail_marker_size,
                                color=trail_color,
                            ),
                            opacity=cfg.trail_opacity,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

        # --- Puntos principales (último punto por grupo) ---
        show_text = cfg.show_labels and len(df_last) <= cfg.max_labeled_points
        text_labels = label.tolist() if show_text else [""] * len(label)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text" if show_text else "markers",
                text=text_labels,
                textposition="top center",
                textfont=dict(size=cfg.label_font_size),
                marker=dict(
                    size=size_vals,
                    color=point_colors,
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>Clave:</b> %{customdata[0]}<br>"
                    "<b>Nombre:</b> %{customdata[1]}<br>"
                    "<b>Segmento:</b> %{customdata[5]}<br>"
                    "Posicionamiento: <b>%{customdata[3]:.2f}%</b><br>"
                    "Margen: <b>%{customdata[4]:.2f}%</b><br>"
                    "Venta: <b>$%{customdata[2]:,.0f}</b><br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # Líneas de referencia
        fig.add_vline(x=cfg.x_ref, line_width=1, line_dash="dash", opacity=0.8)
        fig.add_hline(y=cfg.y_ref, line_width=1, line_dash="dash", opacity=0.8)

        fig.update_layout(
            margin=dict(t=20, l=10, r=10, b=10),
            height=cfg.height,
            xaxis=dict(title="Posicionamiento (%)", range=[xmin, xmax]),
            yaxis=dict(title="Margen (%)", range=[ymin, ymax]),
        )

        return fig


# ======================================================
# Example usage
# ======================================================
"""
from front.figures.scatter_pos_margen import ScatterPosMargen, ScatterConfig

# df debe incluir: sku, macro, categoria, proveedor, venta_neta, posicionamiento, front
# opcional: nombre, segmento, id_segmento

cfg = ScatterConfig(
    x_ref=100.0,
    y_ref=17.72,
    # Personalizar label por agrupación:
    label_col_by_group={
        "macro": "macro",
        "categoria": "categoria",
        "proveedor": "proveedor",
        "sku": "nombre",  # cuando agrupa por SKU, muestra "nombre"
    },
)

component = ScatterPosMargen(df, config=cfg)
fig = component.render(key_prefix="scatter_pos_margen")

st.plotly_chart(fig, use_container_width=True)
"""
