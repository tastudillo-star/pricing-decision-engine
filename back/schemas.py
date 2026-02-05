"""
Back/schemas.py

OBJETIVO
- Definir contratos mínimos: nombres de columnas canónicas y estructuras de config/result.
- Evitar ambigüedad (moneda vs porcentaje, tipos, columnas obligatorias).
- Esto permite cambiar lógica interna sin romper UI.

ENTIDADES PRINCIPALES
1) PipelineConfig
- Contiene parámetros del pipeline: ventanas/fechas, objetivo, alpha, defaults de elasticidad,
  y guardrails.

2) Columnas canónicas del master_df (por SKU)
A) Base SKU (estado actual)
- id_sku, id_segmento, sku, nombre, macro, categoria, proveedor
- venta_neta, cantidad, peso_venta
- precio, precio_competidor
- posicionamiento
- margen

B) Oportunidad
- rol, bucket
- posicionamiento_rol, margen_rol
- tipo_oportunidad
- senales (string compacto "POS_BAJO|MARGEN_BAJO")

C) Elasticidad
- elasticidad

D) Acción
- palanca_recomendada
- precio_recomendado
- delta_precio_pct
- delta_unidades
- delta_venta
- delta_margen
- impacto_venta
- impacto_venta_pct_total_chiper
- razon

SALIDA ESPERADA
- Definiciones/constantes. No lógica de negocio.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


# =============================================================================
# COLUMNAS CANÓNICAS POR MÓDULO
# =============================================================================

# BaseEngine: base_df
BASE_COLUMNS: List[str] = [
    "id_sku",
    "id_segmento",
    "segmento",
    "sku",
    "nombre",
    "macro",
    "categoria",
    "proveedor",
    "venta_neta",
    "cantidad",
    "peso_venta",
    "precio",
    "precio_competidor",
    "posicionamiento",
    "margen",
]

# OpportunityEngine: opp_df
OPP_COLUMNS: List[str] = [
    "sku",
    "rol",
    "bucket",
    "posicionamiento_rol",
    "margen_rol",
    "tipo_oportunidad",
    "senales",
]

# ElasticityEngine: elas_df
ELAS_COLUMNS: List[str] = [
    "sku",
    "elasticidad",
]

# ActionEngine: act_df
ACT_COLUMNS: List[str] = [
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

# master_df: todas las columnas (sin duplicar 'sku')
MASTER_COLUMNS: List[str] = (
    BASE_COLUMNS
    + [c for c in OPP_COLUMNS if c != "sku"]
    + [c for c in ELAS_COLUMNS if c != "sku"]
    + [c for c in ACT_COLUMNS if c != "sku"]
)


# =============================================================================
# GUARDRAILS Y CONFIG
# =============================================================================

@dataclass
class Guardrails:
    """
    Parámetros de límites para el pipeline de pricing.

    Define los rangos permitidos para posicionamiento y cambios de precio.

    Attributes:
        min_posicionamiento: Límite inferior de posicionamiento válido.
        max_posicionamiento: Límite superior de posicionamiento válido.
        min_delta_precio_pct: Máxima reducción de precio permitida (negativo).
        max_delta_precio_pct: Máximo incremento de precio permitido.
    """
    min_posicionamiento: float = 0.5
    max_posicionamiento: float = 2.0
    min_delta_precio_pct: float = -0.20  # límite inferior permitido de cambio de precio
    max_delta_precio_pct: float = 0.20   # límite superior permitido de cambio de precio


@dataclass
class PipelineConfig:
    """
    Configuración canónica del pipeline de pricing.

    Contiene todos los parámetros necesarios para ejecutar el pipeline:
    ventanas temporales, competidor, objetivo de optimización,
    elasticidad por defecto y guardrails.

    Attributes:
        fecha_base: Fecha de referencia para el análisis.
        id_competidor: ID del competidor (1-3) o 4 para min de todos.
        ventana_chiper: Días de ventana para datos de Chiper.
        ventana_comp: Días de ventana para datos de competidor.
        excluir_dias_sin_venta_chiper: Si excluir días sin venta.
        objetivo: Tipo de optimización (MAX_VENTA, MAX_MARGEN, TRADE_OFF).
        alpha: Peso para TRADE_OFF (0-1, solo si objetivo=TRADE_OFF).
        default_elasticidad: Elasticidad por defecto si no hay específica.
        guardrails: Límites de posicionamiento y cambio de precio.
    """

    # Ventanas/fechas/competidor (usado por DataAccess y UI)
    fecha_base: date
    id_competidor: int = 4  # 1: Central, 2: Alvi, 3: La Oferta, 4: min(1–3)
    ventana_chiper: int = 30
    ventana_comp: Optional[int] = None  # si None, usar ventana_chiper
    excluir_dias_sin_venta_chiper: bool = True

    # Objetivo y trade-off
    objetivo: str = "MAX_VENTA"  # MAX_VENTA | MAX_MARGEN | TRADE_OFF
    alpha: Optional[float] = None       # solo aplica para TRADE_OFF

    # Elasticidad
    default_elasticidad: float = -0.1

    # Guardrails (clamps/filtros)
    guardrails: Guardrails = field(default_factory=Guardrails)

    def to_dict(self) -> dict:
        """
        Serializa la configuración a un diccionario plano.

        Returns:
            Dict compatible con los engines del pipeline.
        """
        return {
            "fecha_base": self.fecha_base,
            "id_competidor": self.id_competidor,
            "ventana_chiper": self.ventana_chiper,
            "ventana_comp": self.ventana_comp or self.ventana_chiper,
            "excluir_dias_sin_venta_chiper": self.excluir_dias_sin_venta_chiper,
            "objetivo": self.objetivo,
            "alpha": self.alpha,
            "default_elasticidad": self.default_elasticidad,
            "guardrails": self.guardrails,
        }


# =============================================================================
# RESULTADO DEL PIPELINE
# =============================================================================

@dataclass
class PipelineResult:
    """
    Resultado estructurado del pipeline de pricing.

    Contiene el master_df final, los DataFrames intermedios de cada
    etapa y métricas resumen del análisis.

    Attributes:
        master_df: DataFrame con todas las columnas canónicas por SKU.
        base_df: DataFrame base sin oportunidades ni acciones.
        opp_df: DataFrame de oportunidades detectadas.
        elas_df: DataFrame de elasticidad asignada.
        act_df: DataFrame de acciones recomendadas.
        total_skus: Número total de SKUs procesados.
        skus_con_accion: SKUs con cambio de precio recomendado.
        delta_venta_total: Impacto total en venta proyectado.
        delta_margen_total: Impacto total en margen proyectado.
    """
    master_df: "pd.DataFrame"  # DataFrame con todas las columnas canónicas
    base_df: "pd.DataFrame"    # DataFrame base (sin oportunidades/acciones)
    opp_df: "pd.DataFrame"     # DataFrame de oportunidades
    elas_df: "pd.DataFrame"    # DataFrame de elasticidad
    act_df: "pd.DataFrame"     # DataFrame de acciones

    # Métricas resumen
    total_skus: int = 0
    skus_con_accion: int = 0
    delta_venta_total: float = 0.0
    delta_margen_total: float = 0.0


