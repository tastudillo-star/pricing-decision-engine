"""
Back/action_engine.py

OBJETIVO
- Generar acción recomendada por SKU (V0: PRICE) y calcular impactos usando elasticidad.
- Produce act_df por SKU.

RESPONSABILIDADES (V0)
- palanca_recomendada = "PRICE"
- Definir precio_recomendado según tipo_oportunidad:
  - Mejorar margen: subir precio (clamp)
  - Mejorar posicionamiento: bajar precio hacia banda competitiva (clamp)
  - Rebalance: ajuste moderado
  - Mantener: precio actual
- Calcular impactos por SKU usando elasticidad:
  - delta_precio_pct
  - delta_unidades
  - delta_venta (moneda)
  - delta_margen (moneda o pp, definir en schemas)
  - impacto_venta
  - impacto_venta_pct_total_chiper (requiere total venta_neta agregado)
- Crear razon (texto corto) con tipo + señales

ENTRADA
- base_df (por SKU)
- opp_df (por SKU)
- elas_df (por SKU)
- config

SALIDA
- act_df (por SKU) con:
  sku,
  palanca_recomendada, precio_recomendado,
  delta_precio_pct, delta_unidades, delta_venta, delta_margen,
  impacto_venta, impacto_venta_pct_total_chiper,
  razon
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional


def _coerce_config(config: Any) -> Dict[str, Any]:
    """
    Convierte el config de entrada a un dict normalizado.

    Args:
        config: Configuración en cualquier formato (dict, objeto, None).

    Returns:
        Dict con la configuración normalizada.
    """
    if config is None:
        return {}
    if hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
        return dict(config.to_dict())
    if hasattr(config, "dict") and callable(getattr(config, "dict")):
        return dict(config.dict())
    if isinstance(config, dict):
        return dict(config)
    return dict(getattr(config, "__dict__", {}))


def _get_guardrails(cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Extrae los guardrails de límites de precio del config.

    Args:
        cfg: Diccionario de configuración.

    Returns:
        Dict con min_delta_precio_pct y max_delta_precio_pct.
        Valores por defecto: -0.20 y 0.20 respectivamente.
    """
    gr = cfg.get("guardrails")
    if gr is None:
        return {
            "min_delta_precio_pct": -0.20,
            "max_delta_precio_pct": 0.20,
        }
    if hasattr(gr, "__dict__"):
        return {
            "min_delta_precio_pct": getattr(gr, "min_delta_precio_pct", -0.20),
            "max_delta_precio_pct": getattr(gr, "max_delta_precio_pct", 0.20),
        }
    return {
        "min_delta_precio_pct": gr.get("min_delta_precio_pct", -0.20),
        "max_delta_precio_pct": gr.get("max_delta_precio_pct", 0.20),
    }


class ActionEngine:
    """
    Método principal esperado:
    - generate_actions(base_df, opp_df, elas_df, config) -> act_df
    """

    def generate_actions(
        self,
        base_df: pd.DataFrame,
        opp_df: pd.DataFrame,
        elas_df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Genera acciones de pricing recomendadas por SKU.

        Calcula precio_recomendado según tipo de oportunidad y estima
        impactos usando elasticidad con fórmula Q2 = Q1 * (P2/P1)^elasticidad.

        Args:
            base_df: DataFrame base con datos actuales por SKU.
            opp_df: DataFrame de oportunidades con tipo_oportunidad.
            elas_df: DataFrame de elasticidad por SKU.
            config: Configuración con guardrails y defaults.

        Returns:
            DataFrame act_df con columnas: sku, palanca_recomendada,
            precio_recomendado, cantidad_nueva, venta_nueva,
            posicionamiento_nuevo, margen_nuevo, delta_precio_pct,
            delta_unidades, delta_venta, delta_margen, impacto_venta,
            impacto_venta_pct_total_chiper, razon.
        """
        output_cols = [
            "sku",
            "palanca_recomendada",
            "precio_recomendado",
            "cantidad_nueva",
            "venta_nueva",
            "posicionamiento_nuevo",
            "margen_nuevo",
            "delta_precio_pct",
            "delta_unidades",
            "delta_venta",
            "delta_margen",
            "impacto_venta",
            "impacto_venta_pct_total_chiper",
            "razon",
        ]

        if base_df is None or base_df.empty:
            return pd.DataFrame(columns=output_cols)

        cfg = _coerce_config(config)
        guardrails = _get_guardrails(cfg)
        min_delta = guardrails["min_delta_precio_pct"]
        max_delta = guardrails["max_delta_precio_pct"]

        # Merge de dataframes
        d = base_df[["sku", "precio", "precio_competidor", "venta_neta", "cantidad", "margen"]].copy()

        if opp_df is not None and not opp_df.empty:
            d = d.merge(
                opp_df[["sku", "tipo_oportunidad", "senales", "posicionamiento_rol"]],
                on="sku",
                how="left",
            )
        else:
            d["tipo_oportunidad"] = "MANTENER"
            d["senales"] = "OK"
            d["posicionamiento_rol"] = np.nan

        if elas_df is not None and not elas_df.empty:
            d = d.merge(elas_df[["sku", "elasticidad"]], on="sku", how="left")
        else:
            d["elasticidad"] = cfg.get("default_elasticidad", -1.2)

        # Asegurar tipos numéricos
        for col in ["precio", "precio_competidor", "venta_neta", "cantidad", "margen", "elasticidad", "posicionamiento_rol"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")

        # Rellenar NaN
        d["elasticidad"] = d["elasticidad"].fillna(cfg.get("default_elasticidad", -1.2))
        d["tipo_oportunidad"] = d["tipo_oportunidad"].fillna("MANTENER")
        d["senales"] = d["senales"].fillna("OK")

        # Calcular total venta_neta para impacto_venta_pct_total_chiper
        total_venta_neta = d["venta_neta"].sum()

        # Calcular precio_recomendado según tipo_oportunidad
        d["precio_recomendado"] = d.apply(
            lambda row: self._calc_precio_recomendado(row, min_delta, max_delta), axis=1
        )

        # Calcular delta_precio_pct
        d["delta_precio_pct"] = np.where(
            (d["precio"].notna()) & (d["precio"] != 0),
            (d["precio_recomendado"] - d["precio"]) / d["precio"],
            0.0,
        )

        # Calcular nueva cantidad usando elasticidad: Q2 = Q1 * (P2/P1)^elasticidad
        d["ratio_precio"] = np.where(
            (d["precio"].notna()) & (d["precio"] > 0) & (d["precio_recomendado"] > 0),
            d["precio_recomendado"] / d["precio"],
            1.0,
        )
        d["cantidad_nueva"] = d["cantidad"] * np.power(d["ratio_precio"], d["elasticidad"])
        d["delta_unidades"] = d["cantidad_nueva"] - d["cantidad"]
        d["venta_nueva"] = d["cantidad_nueva"] * d["precio_recomendado"]
        d["delta_venta"] = d["venta_nueva"] - d["venta_neta"]

        # Calcular delta_margen (asumiendo margen = front + back por unidad)
        # margen_unitario = margen_total / cantidad
        d["margen_unitario"] = np.where(
            (d["cantidad"].notna()) & (d["cantidad"] != 0),
            d["margen"] / d["cantidad"],
            0.0,
        )
        # Nuevo margen = margen_unitario * cantidad_nueva + delta_precio * cantidad_nueva
        # Simplificación: el delta_margen incluye el efecto del cambio de precio y volumen
        d["margen_nuevo"] = d["cantidad_nueva"] * (d["margen_unitario"] + (d["precio_recomendado"] - d["precio"]))
        d["delta_margen"] = d["margen_nuevo"] - d["margen"]

        # Nuevo posicionamiento = precio_recomendado / precio_competidor
        d["posicionamiento_nuevo"] = np.where(
            (d["precio_competidor"].notna()) & (d["precio_competidor"] > 0),
            d["precio_recomendado"] / d["precio_competidor"],
            np.nan,
        )

        # Impacto venta
        d["impacto_venta"] = d["delta_venta"]
        d["impacto_venta_pct_total_chiper"] = np.where(
            total_venta_neta != 0,
            d["delta_venta"] / total_venta_neta,
            0.0,
        )

        # Palanca recomendada (V0: siempre PRICE)
        d["palanca_recomendada"] = "PRICE"

        # Razón
        d["razon"] = d.apply(
            lambda row: f"{row['tipo_oportunidad']} | {row['senales']}", axis=1
        )

        return d[output_cols]

    def _calc_precio_recomendado(self, row: pd.Series, min_delta: float, max_delta: float) -> float:
        """
        Calcula el precio recomendado para un SKU según su tipo de oportunidad.

        Lógica por tipo:
        - MEJORA_POS / MEJORA_POS_Y_MARGEN: Bajar precio hacia
          precio_competidor * posicionamiento_rol, limitado por guardrails.
        - MEJORA_MARGEN: Subir precio moderadamente (50% del max_delta).
        - MANTENER: Mantener precio actual.

        Args:
            row: Serie con datos del SKU (precio, precio_competidor,
                 posicionamiento_rol, tipo_oportunidad).
            min_delta: Límite inferior de cambio de precio (ej: -0.20).
            max_delta: Límite superior de cambio de precio (ej: 0.20).

        Returns:
            Precio recomendado redondeado a 2 decimales.
        """
        precio = row.get("precio")
        precio_comp = row.get("precio_competidor")
        pos_rol = row.get("posicionamiento_rol")
        tipo = row.get("tipo_oportunidad", "MANTENER")

        if pd.isna(precio) or precio == 0:
            return precio if pd.notna(precio) else 0.0

        if tipo in ("MEJORA_POS", "MEJORA_POS_Y_MARGEN"):
            # Objetivo: precio = precio_competidor * posicionamiento_rol
            if pd.notna(precio_comp) and pd.notna(pos_rol) and precio_comp > 0:
                precio_objetivo = precio_comp * pos_rol
                # Clamp al rango permitido
                precio_min = precio * (1 + min_delta)
                precio_max = precio * (1 + max_delta)
                precio_recomendado = max(precio_min, min(precio_objetivo, precio_max))
                return round(precio_recomendado, 2)
            else:
                # Si no hay datos de competidor, bajar al mínimo permitido
                return round(precio * (1 + min_delta), 2)

        elif tipo == "MEJORA_MARGEN":
            # Subir precio moderadamente (mitad del max_delta)
            incremento = max_delta * 0.5
            return round(precio * (1 + incremento), 2)

        else:  # MANTENER u otro
            return round(precio, 2)



