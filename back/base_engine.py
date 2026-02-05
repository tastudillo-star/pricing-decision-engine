"""
Back/base_engine.py

OBJETIVO
- Construir el base_df por SKU consolidando datos de ventas, precios y catálogos.
- Produce base_df con métricas agregadas por SKU.

RESPONSABILIDADES
- Acceder a las tablas via data_access.
- Calcular métricas agregadas: precio ponderado, posicionamiento, margen.
- Enriquecer con datos de catálogos (categoría, macro, proveedor, segmento).
- Aplicar filtros de estabilidad (posicionamiento en rango válido).

ENTRADA
- config (PipelineConfig o dict)

SALIDA
- base_df (por SKU) con columnas canónicas base.
"""

import numpy as np
import pandas as pd
from datetime import date
from typing import Any, Dict, Optional

from back.data_access import (
    get_ventas_chiper,
    get_precio_competidor,
    get_sku,
    get_categoria,
    get_macro_categoria,
    get_proveedor,
    get_segmento,
    get_fecha_rango,
    get_last_sunday,
)


def _coerce_config(config: Any) -> Dict[str, Any]:
    """
    Convierte el config de entrada a un dict normalizado.

    Args:
        config: Configuración en cualquier formato (dict, objeto, None).

    Returns:
        Dict con claves normalizadas.
    """
    if config is None:
        cfg = {}
    elif hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
        cfg = config.to_dict()
    elif hasattr(config, "dict") and callable(getattr(config, "dict")):
        cfg = dict(config.dict())
    elif isinstance(config, dict):
        cfg = dict(config)
    else:
        cfg = dict(getattr(config, "__dict__", {}))

    # Normalizar fecha_base
    fecha_base = cfg.get("fecha_base") or cfg.get("fecha") or cfg.get("fecha_actual")
    if fecha_base is None:
        fecha_base = get_last_sunday()
    elif isinstance(fecha_base, str):
        fecha_base = date.fromisoformat(fecha_base)

    return {
        "fecha_base": fecha_base,
        "id_competidor": int(cfg.get("id_competidor", 4)),
        "ventana_chiper": int(cfg.get("ventana_chiper", 30)),
        "ventana_comp": int(cfg.get("ventana_comp", cfg.get("ventana_chiper", 30))),
        "excluir_dias_sin_venta_chiper": bool(cfg.get("excluir_dias_sin_venta_chiper", True)),
    }


class BaseEngine:
    """
    Engine para construir el base_df consolidado por SKU.

    Método principal:
    - build_base_df(config) -> base_df
    """

    def build_base_df(self, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Construye el DataFrame base por SKU.

        Ejecuta el flujo completo:
        1. Obtener datos de ventas y precios.
        2. Calcular métricas diarias y agregarlas por SKU.
        3. Enriquecer con catálogos.
        4. Aplicar filtros de estabilidad.

        Args:
            config: Configuración del pipeline.

        Returns:
            DataFrame con una fila por SKU y columnas canónicas base.
        """
        cfg = _coerce_config(config)

        # Calcular rangos de fechas
        fecha_inicio_chiper, fecha_fin_chiper = get_fecha_rango(
            cfg["fecha_base"], cfg["ventana_chiper"]
        )
        fecha_inicio_comp, fecha_fin_comp = get_fecha_rango(
            cfg["fecha_base"], cfg["ventana_comp"]
        )

        # 1. Obtener datos de ventas
        ventas_df = get_ventas_chiper(fecha_inicio_chiper, fecha_fin_chiper)
        if ventas_df.empty:
            return self._empty_base_df()

        # 2. Filtrar días válidos si está configurado
        if cfg["excluir_dias_sin_venta_chiper"]:
            ventas_df = self._filter_valid_days(ventas_df)

        # 3. Calcular métricas diarias de Chiper
        chiper_diario = self._calc_chiper_diario(ventas_df)

        # 4. Obtener datos de competidor
        competidor_df = get_precio_competidor(
            fecha_inicio_comp, fecha_fin_comp, cfg["id_competidor"]
        )
        competidor_diario = self._calc_competidor_diario(competidor_df)

        # 5. Join diario y calcular métricas
        joined_df = self._join_diario(chiper_diario, competidor_diario)

        # 6. Agregar por SKU
        agg_df = self._aggregate_by_sku(joined_df)

        # 7. Enriquecer con catálogos
        enriched_df = self._enrich_with_catalogs(agg_df)

        # 8. Aplicar filtros y normalizar
        base_df = self._normalize_base_df(enriched_df)

        return base_df

    def _empty_base_df(self) -> pd.DataFrame:
        """Retorna un DataFrame vacío con las columnas canónicas."""
        return pd.DataFrame(columns=[
            "id_sku", "id_segmento", "segmento", "sku", "nombre",
            "macro", "categoria", "proveedor",
            "venta_neta", "cantidad", "peso_venta",
            "precio", "precio_competidor", "posicionamiento", "margen",
        ])

    def _filter_valid_days(self, ventas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra solo días con venta total > 0.

        Args:
            ventas_df: DataFrame de ventas diarias.

        Returns:
            DataFrame filtrado a días válidos.
        """
        if ventas_df.empty:
            return ventas_df

        # Calcular venta total por día
        venta_por_dia = ventas_df.groupby("fecha")["venta_neta"].sum().reset_index()
        dias_validos = venta_por_dia[venta_por_dia["venta_neta"] > 0]["fecha"]

        return ventas_df[ventas_df["fecha"].isin(dias_validos)]

    def _calc_chiper_diario(self, ventas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas diarias de Chiper por SKU.

        Args:
            ventas_df: DataFrame de ventas.

        Returns:
            DataFrame con métricas diarias por SKU.
        """
        if ventas_df.empty:
            return pd.DataFrame(columns=[
                "id_sku", "fecha", "venta_neta_dia", "precio_chiper_dia",
                "front_dia", "back_dia", "margen_dia"
            ])

        df = ventas_df.copy()
        df["venta_neta"] = pd.to_numeric(df["venta_neta"], errors="coerce").fillna(0)
        df["precio_bruto"] = pd.to_numeric(df["precio_bruto"], errors="coerce")
        df["front"] = pd.to_numeric(df["front"], errors="coerce").fillna(0)
        df["back"] = pd.to_numeric(df["back"], errors="coerce").fillna(0)

        # Agregar por SKU y fecha
        agg = df.groupby(["id_sku", "fecha"]).agg(
            venta_neta_dia=("venta_neta", "sum"),
            precio_x_venta=("precio_bruto", lambda x: (x * df.loc[x.index, "venta_neta"]).sum()),
            front_x_venta=("front", lambda x: (x * df.loc[x.index, "venta_neta"]).sum()),
            back_x_venta=("back", lambda x: (x * df.loc[x.index, "venta_neta"]).sum()),
        ).reset_index()

        # Calcular promedios ponderados
        agg["precio_chiper_dia"] = np.where(
            agg["venta_neta_dia"] > 0,
            agg["precio_x_venta"] / agg["venta_neta_dia"],
            np.nan
        )
        agg["front_dia"] = np.where(
            agg["venta_neta_dia"] > 0,
            agg["front_x_venta"] / agg["venta_neta_dia"],
            0
        )
        agg["back_dia"] = np.where(
            agg["venta_neta_dia"] > 0,
            agg["back_x_venta"] / agg["venta_neta_dia"],
            0
        )
        agg["margen_dia"] = agg["front_dia"] + agg["back_dia"]

        return agg[["id_sku", "fecha", "venta_neta_dia", "precio_chiper_dia", "margen_dia"]]

    def _calc_competidor_diario(self, competidor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula precio mínimo de competidor por SKU y día.

        Args:
            competidor_df: DataFrame de precios de competidor.

        Returns:
            DataFrame con precio mínimo por SKU y día.
        """
        if competidor_df.empty:
            return pd.DataFrame(columns=["id_sku", "fecha", "precio_competidor_min_dia"])

        df = competidor_df.copy()
        df["precio_lleno"] = pd.to_numeric(df["precio_lleno"], errors="coerce")
        df["precio_descuento"] = pd.to_numeric(df["precio_descuento"], errors="coerce")

        # Calcular precio mínimo entre lleno y descuento
        df["precio_min"] = df[["precio_lleno", "precio_descuento"]].min(axis=1)

        # Obtener el mínimo por SKU y fecha (entre todos los competidores)
        result = df.groupby(["id_sku", "fecha"]).agg(
            precio_competidor_min_dia=("precio_min", "min")
        ).reset_index()

        return result

    def _join_diario(
        self, chiper_df: pd.DataFrame, competidor_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Une datos diarios de Chiper con competidor.

        Args:
            chiper_df: DataFrame de métricas diarias de Chiper.
            competidor_df: DataFrame de precios de competidor.

        Returns:
            DataFrame unido con posicionamiento diario.
        """
        if chiper_df.empty:
            return pd.DataFrame()

        df = chiper_df.merge(competidor_df, on=["id_sku", "fecha"], how="left")

        # Calcular posicionamiento diario
        df["posicionamiento_dia"] = np.where(
            (df["precio_competidor_min_dia"].notna()) & (df["precio_competidor_min_dia"] > 0),
            df["precio_chiper_dia"] / df["precio_competidor_min_dia"],
            np.nan
        )

        # Calcular unidades diarias
        df["unidades_dia"] = np.where(
            (df["precio_chiper_dia"].notna()) & (df["precio_chiper_dia"] > 0),
            df["venta_neta_dia"] / df["precio_chiper_dia"],
            0
        )

        return df

    def _aggregate_by_sku(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega métricas diarias por SKU.

        Args:
            df: DataFrame con métricas diarias.

        Returns:
            DataFrame agregado por SKU.
        """
        if df.empty:
            return pd.DataFrame(columns=[
                "id_sku", "venta_neta", "cantidad", "precio", "precio_competidor",
                "posicionamiento", "margen"
            ])

        # Filtrar solo días con venta
        df = df[df["venta_neta_dia"] > 0].copy()

        if df.empty:
            return pd.DataFrame(columns=[
                "id_sku", "venta_neta", "cantidad", "precio", "precio_competidor",
                "posicionamiento", "margen"
            ])

        agg = df.groupby("id_sku").apply(
            lambda g: pd.Series({
                "venta_neta": g["venta_neta_dia"].sum(),
                "cantidad": g["unidades_dia"].sum(),
                "precio": (g["precio_chiper_dia"] * g["venta_neta_dia"]).sum() / g["venta_neta_dia"].sum()
                    if g["venta_neta_dia"].sum() > 0 else np.nan,
                "precio_competidor": (
                    g.loc[g["precio_competidor_min_dia"].notna(), "precio_competidor_min_dia"] *
                    g.loc[g["precio_competidor_min_dia"].notna(), "venta_neta_dia"]
                ).sum() / g.loc[g["precio_competidor_min_dia"].notna(), "venta_neta_dia"].sum()
                    if g.loc[g["precio_competidor_min_dia"].notna(), "venta_neta_dia"].sum() > 0 else np.nan,
                "posicionamiento": (
                    g.loc[g["posicionamiento_dia"].notna(), "posicionamiento_dia"] *
                    g.loc[g["posicionamiento_dia"].notna(), "venta_neta_dia"]
                ).sum() / g.loc[g["posicionamiento_dia"].notna(), "venta_neta_dia"].sum()
                    if g.loc[g["posicionamiento_dia"].notna(), "venta_neta_dia"].sum() > 0 else np.nan,
                "margen": (g["margen_dia"] * g["venta_neta_dia"]).sum() / g["venta_neta_dia"].sum()
                    if g["venta_neta_dia"].sum() > 0 else np.nan,
            })
        ).reset_index()

        # Calcular peso de venta
        total_venta = agg["venta_neta"].sum()
        agg["peso_venta"] = np.where(total_venta > 0, agg["venta_neta"] / total_venta, np.nan)

        return agg

    def _enrich_with_catalogs(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece el DataFrame con datos de catálogos.

        Args:
            agg_df: DataFrame agregado por SKU.

        Returns:
            DataFrame enriquecido con categoría, macro, proveedor, segmento.
        """
        if agg_df.empty:
            return agg_df

        # Obtener catálogos
        sku_df = get_sku()
        categoria_df = get_categoria()
        macro_df = get_macro_categoria()
        proveedor_df = get_proveedor()
        segmento_df = get_segmento()

        # Join con SKU
        df = agg_df.merge(
            sku_df.rename(columns={"id": "id_sku", "nombre": "nombre_sku"}),
            on="id_sku",
            how="left"
        )

        # Join con categoría
        df = df.merge(
            categoria_df.rename(columns={"id": "id_categoria", "nombre": "categoria"}),
            on="id_categoria",
            how="left"
        )

        # Join con macro categoría
        df = df.merge(
            macro_df.rename(columns={"id": "id_macro", "nombre": "macro"}),
            on="id_macro",
            how="left"
        )

        # Join con proveedor
        df = df.merge(
            proveedor_df.rename(columns={"id": "id_proveedor", "nombre": "proveedor"}),
            on="id_proveedor",
            how="left"
        )

        # Join con segmento
        df = df.merge(
            segmento_df.rename(columns={"id": "id_segmento", "nombre": "segmento"}),
            on="id_segmento",
            how="left"
        )

        # Renombrar columnas
        df = df.rename(columns={"nombre_sku": "nombre"})

        return df

    def _normalize_base_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza el DataFrame aplicando filtros y seleccionando columnas.

        Args:
            df: DataFrame enriquecido.

        Returns:
            DataFrame normalizado con columnas canónicas.
        """
        if df.empty:
            return self._empty_base_df()

        # Filtrar posicionamiento en rango válido [0.5, 2.0]
        df = df[
            (df["posicionamiento"].isna()) |
            (df["posicionamiento"].between(0.5, 2.0))
        ].copy()

        # Seleccionar y ordenar columnas
        cols = [
            "id_sku", "id_segmento", "segmento", "sku", "nombre",
            "macro", "categoria", "proveedor",
            "venta_neta", "cantidad", "peso_venta",
            "precio", "precio_competidor", "posicionamiento", "margen",
        ]

        # Asegurar que existen todas las columnas
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan

        return df[cols].reset_index(drop=True)
