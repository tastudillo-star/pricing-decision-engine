"""
Back/main_pipeline.py

OBJETIVO
- Orquestar todo el pipeline.
- Entregar un solo DataFrame master_df por SKU con todas las columnas canónicas.

REGLA
- main_pipeline es el único punto de entrada para el Front.
- Todo se calcula por SKU (1 fila por SKU).

FLUJO
1) base_df = BaseEngine.build_base_df(config)
2) opp_df  = OpportunityEngine.build_opportunities(base_df, config)
3) elas_df = ElasticityEngine.assign_elasticity(base_df, config)
4) act_df  = ActionEngine.generate_actions(base_df, opp_df, elas_df, config)
5) master_df = join por 'sku' (left joins) en orden:
   base_df ⟂ opp_df ⟂ elas_df ⟂ act_df
6) (Opcional) añadir columnas de objetivo/utility/selected_flag, sin romper contrato.
7) return master_df

ENTRADA
- PipelineConfig (o dict)

SALIDA
- master_df (DataFrame por SKU con TODO)
"""

import pandas as pd
from typing import Any, Dict, Union

from back.base_engine import BaseEngine
from back.opportunity_engine import OpportunityEngine
from back.elasticity_engine import ElasticityEngine
from back.action_engine import ActionEngine
from back.schemas import PipelineConfig, PipelineResult, MASTER_COLUMNS


def _coerce_config(config: Any) -> Dict[str, Any]:
    """
    Convierte PipelineConfig o dict a dict plano normalizado.

    Args:
        config: Configuración en cualquier formato (PipelineConfig, dict, None).

    Returns:
        Dict con la configuración normalizada para los engines.
    """
    if config is None:
        return {}
    if isinstance(config, PipelineConfig):
        return config.to_dict()
    if hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
        return config.to_dict()
    if hasattr(config, "dict") and callable(getattr(config, "dict")):
        return dict(config.dict())
    if isinstance(config, dict):
        return dict(config)
    return dict(getattr(config, "__dict__", {}))


class PricingPipeline:
    """
    Clase orquestadora del pipeline de pricing.

    Coordina la ejecución de todos los engines y construye
    el master_df final con todas las columnas canónicas.

    Attributes:
        base_engine: Instancia de BaseEngine para obtener datos base.
        opportunity_engine: Instancia de OpportunityEngine.
        elasticity_engine: Instancia de ElasticityEngine.
        action_engine: Instancia de ActionEngine.
    """

    def __init__(self):
        """Inicializa el pipeline con instancias de todos los engines."""
        self.base_engine = BaseEngine()
        self.opportunity_engine = OpportunityEngine()
        self.elasticity_engine = ElasticityEngine()
        self.action_engine = ActionEngine()

    def run(self, config: Union[PipelineConfig, Dict[str, Any]]) -> PipelineResult:
        """
        Ejecuta el pipeline completo y devuelve un PipelineResult.

        Parámetros:
            config: PipelineConfig o dict con parámetros del pipeline.

        Retorna:
            PipelineResult con master_df y DataFrames intermedios.
        """
        cfg = _coerce_config(config)

        # 1) Obtener base_df
        base_df = self.base_engine.build_base_df(cfg)

        # 2) Calcular oportunidades
        opp_df = self.opportunity_engine.build_opportunities(base_df, cfg)

        # 3) Asignar elasticidad
        elas_df = self.elasticity_engine.assign_elasticity(base_df, cfg)

        # 4) Generar acciones
        act_df = self.action_engine.generate_actions(base_df, opp_df, elas_df, cfg)

        # 5) Construir master_df (left joins por 'sku')
        master_df = self._build_master_df(base_df, opp_df, elas_df, act_df)

        # 6) Calcular métricas resumen
        total_skus = len(master_df)
        skus_con_accion = int((master_df.get("delta_precio_pct", pd.Series([0])) != 0).sum())
        delta_venta_total = float(master_df.get("delta_venta", pd.Series([0])).sum())
        delta_margen_total = float(master_df.get("delta_margen", pd.Series([0])).sum())

        return PipelineResult(
            master_df=master_df,
            base_df=base_df,
            opp_df=opp_df,
            elas_df=elas_df,
            act_df=act_df,
            total_skus=total_skus,
            skus_con_accion=skus_con_accion,
            delta_venta_total=delta_venta_total,
            delta_margen_total=delta_margen_total,
        )

    def _build_master_df(
        self,
        base_df: pd.DataFrame,
        opp_df: pd.DataFrame,
        elas_df: pd.DataFrame,
        act_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construye el master_df uniendo todos los DataFrames por SKU.

        Realiza left joins secuenciales: base_df <- opp_df <- elas_df <- act_df.

        Args:
            base_df: DataFrame base con datos por SKU.
            opp_df: DataFrame de oportunidades.
            elas_df: DataFrame de elasticidad.
            act_df: DataFrame de acciones.

        Returns:
            DataFrame master con todas las columnas canónicas.
        """
        if base_df is None or base_df.empty:
            return pd.DataFrame(columns=MASTER_COLUMNS)

        master = base_df.copy()

        # Join con opp_df
        if opp_df is not None and not opp_df.empty:
            opp_cols = [c for c in opp_df.columns]
            master = master.merge(opp_df[opp_cols], on="sku", how="left")

        # Join con elas_df
        if elas_df is not None and not elas_df.empty:
            elas_cols = [c for c in elas_df.columns]
            master = master.merge(elas_df[elas_cols], on="sku", how="left")

        # Join con act_df
        if act_df is not None and not act_df.empty:
            act_cols = [c for c in act_df.columns]
            master = master.merge(act_df[act_cols], on="sku", how="left")

        return master


def run_pipeline(config: Union[PipelineConfig, Dict[str, Any]]) -> PipelineResult:
    """
    Ejecuta el pipeline completo de pricing.

    Función helper para uso desde Streamlit u otros consumidores.
    Instancia PricingPipeline y ejecuta el método run().

    Args:
        config: Configuración del pipeline (PipelineConfig o dict).

    Returns:
        PipelineResult con master_df, DataFrames intermedios y métricas.
    """
    pipeline = PricingPipeline()
    return pipeline.run(config)

