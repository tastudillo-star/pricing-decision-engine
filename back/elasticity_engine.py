"""
Back/elasticity_engine.py

OBJETIVO
- Asignar elasticidad por SKU (V0 simple y robusto) y dejar listo para usar en acciones.
- Produce elas_df por SKU.

RESPONSABILIDADES (V0)
- Definir elasticidad por SKU con estrategia robusta:
  - Preferencia: elasticidad por categoria y luego join por SKU
  - Fallback: default global si falta categoria o no hay mapping
- Mantener esto modular para evolucionar:
  V1: por categoría estimada
  V2: por SKU/segmento con shrinkage
  V3: cross-elasticities

ENTRADA
- base_df (por SKU)
- config (para escenarios / defaults)

SALIDA
- elas_df (por SKU) con:
  sku, elasticidad
"""

import pandas as pd
from typing import Any, Dict, Optional


def _coerce_config(config: Any) -> Dict[str, Any]:
    """
    Convierte el config de entrada a un dict normalizado.

    Args:
        config: Configuración en cualquier formato (dict, objeto, None).

    Returns:
        Dict con la configuración normalizada.
    """
    cfg: Dict[str, Any]
    if config is None:
        cfg = {}
    elif hasattr(config, "dict") and callable(getattr(config, "dict")):
        cfg = dict(config.dict())
    elif isinstance(config, dict):
        cfg = dict(config)
    else:
        cfg = dict(getattr(config, "__dict__", {}))
    return cfg


class ElasticityEngine:
    """
    Método principal esperado:
    - assign_elasticity(base_df, config) -> elas_df
    """

    def assign_elasticity(self, base_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Asigna elasticidad a cada SKU.

        V0: Asigna elasticidad constante (default_elasticidad del config)
        a todos los SKUs. Preparado para evolucionar a elasticidad
        por categoría o SKU específico.

        Args:
            base_df: DataFrame base con datos por SKU.
            config: Configuración con default_elasticidad (opcional).

        Returns:
            DataFrame elas_df con columnas: sku, elasticidad.
        """
        if base_df is None or base_df.empty:
            return pd.DataFrame(columns=["sku", "elasticidad"])

        cfg = _coerce_config(config)
        default_elas = float(cfg.get("default_elasticidad", -1.3))

        d = base_df.copy()
        if "sku" not in d.columns:
            d["sku"] = pd.NA

        elas_df = d[["sku"]].copy()
        elas_df["elasticidad"] = default_elas
        return elas_df

