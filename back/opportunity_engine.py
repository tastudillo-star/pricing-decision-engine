"""
Back/opportunity_engine.py

OBJETIVO
- En base al base_df (por SKU), asignar rol/bucket, targets por rol y diagnosticar oportunidades.
- Produce opp_df por SKU.

RESPONSABILIDADES (V0)
- Asignar rol y bucket por SKU (según reglas existentes o tabla).
- Traer targets por rol:
  - posicionamiento_rol
  - margen_rol
- Calcular señales simples:
  - POS_BAJO, MARGEN_BAJO, etc.
- Determinar tipo_oportunidad (determinista y simple)
- Componer string senales (ej: "POS_BAJO|MARGEN_BAJO")
- (Opcional) score_base/prioridad si lo necesitas luego, pero V0 puede omitirse.

ENTRADA
- base_df (por SKU)
- config

SALIDA
- opp_df (por SKU) con:
  sku, rol, bucket,
  posicionamiento_rol, margen_rol,
  tipo_oportunidad, senales
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from back.data_access import get_regla_negocio, get_regla_negocio_override


class OpportunityEngine:
    """
    Engine para detectar oportunidades de pricing por SKU.

    Método principal:
    - build_opportunities(base_df, config) -> opp_df
    """

    @staticmethod
    def _assign_bucket(df: pd.DataFrame) -> pd.DataFrame:
        """
        Asigna bucket (TOP80/FONDO20) a cada SKU por categoría.

        Ordena SKUs por venta_neta descendente dentro de cada categoría
        y asigna TOP80 a los que acumulan hasta 80% de la venta.

        Args:
            df: DataFrame con columnas 'categoria' y 'venta_neta'.

        Returns:
            DataFrame con columnas adicionales: share_cat, cum_share_cat, bucket.
        """
        if df is None or df.empty:
            return df
        d = df.copy()
        d["categoria"] = d.get("categoria", "").fillna("Sin categoria")
        d["venta_neta"] = pd.to_numeric(d.get("venta_neta"), errors="coerce").fillna(0.0)
        d = d.sort_values(["categoria", "venta_neta"], ascending=[True, False]).reset_index(drop=True)
        cat_total = d.groupby("categoria", dropna=False)["venta_neta"].transform("sum")
        d["share_cat"] = np.where(cat_total > 0, d["venta_neta"] / cat_total, np.nan)
        d["cum_share_cat"] = d.groupby("categoria", dropna=False)["share_cat"].cumsum()
        d["bucket"] = np.where(d["cum_share_cat"] <= 0.80, "TOP80", "FONDO20")
        return d

    @staticmethod
    def _attach_targets(df: pd.DataFrame, reglas: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
        """
        Adjunta targets de posicionamiento y margen a cada SKU.

        Usa reglas por segmento como base y aplica overrides por SKU
        si existen. Asigna posicionamiento_top o posicionamiento_fondo
        según el bucket del SKU.

        Args:
            df: DataFrame con SKUs y columnas id_sku, id_segmento, bucket.
            reglas: DataFrame de reglas por segmento.
            overrides: DataFrame de overrides por SKU.

        Returns:
            DataFrame con columnas adicionales: posicionamiento_rol, margen_rol.
        """
        d = df.copy()
        if "id_sku" not in d.columns:
            d["id_sku"] = pd.NA
        if "id_segmento" not in d.columns:
            d["id_segmento"] = pd.NA

        d["id_sku"] = pd.to_numeric(d["id_sku"], errors="coerce")
        d["id_segmento"] = pd.to_numeric(d["id_segmento"], errors="coerce")

        rn = reglas.copy() if isinstance(reglas, pd.DataFrame) else pd.DataFrame()
        if not rn.empty:
            rn["id_segmento"] = pd.to_numeric(rn["id_segmento"], errors="coerce")
            rn = rn.rename(columns={
                "posicionamiento_top": "rn_pos_top",
                "posicionamiento_fondo": "rn_pos_fondo",
                "margen": "rn_margen",
            })
            d = d.merge(rn[["id_segmento", "rn_pos_top", "rn_pos_fondo", "rn_margen"]], on="id_segmento", how="left")
        else:
            d["rn_pos_top"] = pd.NA
            d["rn_pos_fondo"] = pd.NA
            d["rn_margen"] = pd.NA

        ov = overrides.copy() if isinstance(overrides, pd.DataFrame) else pd.DataFrame()
        if not ov.empty:
            ov["id_sku"] = pd.to_numeric(ov["id_sku"], errors="coerce")
            ov = ov.rename(columns={
                "posicionamiento_top": "ov_pos_top",
                "posicionamiento_fondo": "ov_pos_fondo",
                "margen": "ov_margen",
            })
            d = d.merge(ov[["id_sku", "ov_pos_top", "ov_pos_fondo", "ov_margen"]], on="id_sku", how="left")
        else:
            d["ov_pos_top"] = pd.NA
            d["ov_pos_fondo"] = pd.NA
            d["ov_margen"] = pd.NA

        # Asignar posicionamiento_rol según bucket usando máscaras
        top_mask = d["bucket"] == "TOP80"
        fondo_mask = d["bucket"] == "FONDO20"

        d["posicionamiento_rol"] = pd.NA
        d.loc[top_mask, "posicionamiento_rol"] = d.loc[top_mask, "ov_pos_top"].combine_first(d.loc[top_mask, "rn_pos_top"])
        d.loc[fondo_mask, "posicionamiento_rol"] = d.loc[fondo_mask, "ov_pos_fondo"].combine_first(d.loc[fondo_mask, "rn_pos_fondo"])

        d["margen_rol"] = d["ov_margen"].combine_first(d["rn_margen"])
        return d

    @staticmethod
    def _diagnose(df: pd.DataFrame) -> pd.DataFrame:
        """
        Diagnostica oportunidades comparando métricas actuales vs targets.

        Genera señales (POS_ALTO, POS_BAJO, MARGEN_BAJO, MARGEN_ALTO)
        y determina el tipo de oportunidad según las señales detectadas.

        Args:
            df: DataFrame con métricas actuales y targets (posicionamiento,
                margen, posicionamiento_rol, margen_rol).

        Returns:
            DataFrame con columnas adicionales: senal_pos, senal_margen,
            tipo_oportunidad, senales, rol.
        """
        d = df.copy()
        d["posicionamiento"] = pd.to_numeric(d.get("posicionamiento"), errors="coerce")
        d["margen"] = pd.to_numeric(d.get("margen"), errors="coerce")
        d["posicionamiento_rol"] = pd.to_numeric(d.get("posicionamiento_rol"), errors="coerce")
        d["margen_rol"] = pd.to_numeric(d.get("margen_rol"), errors="coerce")

        # Usar máscaras booleanas que manejan NaN correctamente
        pos_alto_mask = d["posicionamiento"] > d["posicionamiento_rol"]
        pos_bajo_mask = d["posicionamiento"] < d["posicionamiento_rol"]
        d["senal_pos"] = None
        d.loc[pos_alto_mask, "senal_pos"] = "POS_ALTO"
        d.loc[pos_bajo_mask, "senal_pos"] = "POS_BAJO"

        margen_bajo_mask = d["margen"] < d["margen_rol"]
        margen_alto_mask = d["margen"] > d["margen_rol"]
        d["senal_margen"] = None
        d.loc[margen_bajo_mask, "senal_margen"] = "MARGEN_BAJO"
        d.loc[margen_alto_mask, "senal_margen"] = "MARGEN_ALTO"

        def tipo(row) -> str:
            pos = row.get("senal_pos")
            mar = row.get("senal_margen")
            if pos == "POS_BAJO" and mar == "MARGEN_BAJO":
                return "MEJORA_POS_Y_MARGEN"
            if pos == "POS_BAJO":
                return "MEJORA_POS"
            if mar == "MARGEN_BAJO":
                return "MEJORA_MARGEN"
            return "MANTENER"

        d["tipo_oportunidad"] = d.apply(tipo, axis=1)
        d["senales"] = d[["senal_pos", "senal_margen"]].apply(
            lambda r: "|".join([x for x in r if pd.notna(x)]) if any(pd.notna(r)) else "OK",
            axis=1,
        )
        d["rol"] = np.where(d["bucket"] == "TOP80", "DRIVER", "BUILDER")
        return d

    def build_opportunities(self, base_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Construye el DataFrame de oportunidades por SKU.

        Método principal de la clase. Ejecuta el flujo:
        assign_bucket -> attach_targets -> diagnose.

        Args:
            base_df: DataFrame base con datos por SKU.
            config: Configuración del pipeline (opcional).

        Returns:
            DataFrame opp_df con columnas: sku, rol, bucket,
            posicionamiento_rol, margen_rol, tipo_oportunidad, senales.
        """
        if base_df is None or base_df.empty:
            return pd.DataFrame(columns=[
                "sku", "rol", "bucket", "posicionamiento_rol",
                "margen_rol", "tipo_oportunidad", "senales",
            ])

        # Cargar reglas usando data_access
        reglas = get_regla_negocio()
        overrides = get_regla_negocio_override()

        d = self._assign_bucket(base_df)
        d = self._attach_targets(d, reglas, overrides)
        d = self._diagnose(d)

        cols = [
            "sku", "rol", "bucket", "posicionamiento_rol",
            "margen_rol", "tipo_oportunidad", "senales",
        ]
        return d[cols] if set(cols).issubset(d.columns) else d
