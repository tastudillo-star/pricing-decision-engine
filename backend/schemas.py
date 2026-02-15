# backend/schemas.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import pandas as pd


@dataclass(frozen=True)
class DFSchema:
    dtypes: Dict[str, str]


def _normalize_date_to_string(s: pd.Series) -> pd.Series:
    """
    Normaliza una columna de fecha a string consistente YYYY-MM-DD.
    - Si viene como datetime/date -> formatea.
    - Si viene como string mixto -> intenta parsear; si falla, deja NA o el valor original.
    """
    # Intento 1: parsear a datetime (robusto, no revienta)
    dt = pd.to_datetime(s, errors="coerce")

    # Si al menos algunos parsearon, usamos formato YYYY-MM-DD
    # y donde no pudo parsear, dejamos NA (string dtype soporta <NA>)
    out = dt.dt.strftime("%Y-%m-%d")

    # Asegura dtype pandas string
    return out.astype("string")


def coerce_df(df: pd.DataFrame, schema: DFSchema) -> pd.DataFrame:
    """
    Coerce robusto:
    - Numéricos -> to_numeric(errors="coerce") + dtype nullable
    - Strings -> astype("string")
    - Aplica solo a columnas presentes
    """
    if df is None:
        return df

    for col, dt in schema.dtypes.items():
        if col not in df.columns:
            continue

        # Caso especial: fecha como string normalizada
        if col == "fecha" and dt == "string":
            df[col] = _normalize_date_to_string(df[col])
            continue

        # Primero coerce numérico si aplica
        if dt in ("Int64", "Float64"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Luego aplica dtype (nullable)
        df[col] = df[col].astype(dt)

    return df

# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
# Básicos
SKU = DFSchema(
    dtypes={
        "id": "Int64",
        "sku": "string",
        "nombre": "string",
        "id_categoria": "Int64",
        "id_proveedor": "Int64",
        "id_segmento": "Int64",
    },
)

CATEGORIA = DFSchema(
    dtypes={
        "id": "Int64",
        "nombre": "string",
        "id_macro": "Int64",
    },
)

MACRO_CATEGORIA = DFSchema(
    dtypes={
        "id": "Int64",
        "nombre": "string",
    },
)

PROVEEDOR = DFSchema(
    dtypes={
        "id": "Int64",
        "nombre": "string",
    },
)

COMPETIDOR = DFSchema(
    dtypes={
        "id": "Int64",
        "nombre": "string",
    },
)

# Vetnas y precios
VENTAS = DFSchema(
    dtypes={
        "id": "Int64",
        "id_sku": "Int64",
        "fecha": "string",   # mantener como string (pandas string dtype)
        "cantidad": "Int64",
        "venta_neta": "Float64",
        "ganancia_neta": "Float64",
        "descuento_neto": "Float64",
        "iva": "Float64",
        "front": "Float64",
        "back": "Float64",
        "precio_neto": "Float64",
        "precio_bruto": "Float64",
        "r": "Float64",
        "precio_bruto_descuento": "Float64",
    },
)

PRECIO_COMPETIDOR = DFSchema(
    dtypes={
        "id": "Int64",
        "id_sku": "Int64",
        "id_competidor": "Int64",
        "fecha": "string",
        "precio_lleno": "Float64",
        "precio_descuento": "Float64",
    },
)

# Reglas de negocio y segmentos
SEGMENTO = DFSchema(
    dtypes={
        "id": "Int64",
        "nombre": "string",
    },
)

REGLA_NEGOCIO = DFSchema(
    dtypes={
        "id_segmento": "Int64",
        "posicionamiento_top": "Float64",
        "posicionamiento_fondo": "Float64",
        "margen": "Float64",
        "fecha": "string",
    },
)

REGLA_NEGOCIO_OVERRIDE = DFSchema(
    dtypes={
        "id_sku": "Int64",
        "posicionamiento_top": "Float64",
        "posicionamiento_fondo": "Float64",
        "margen": "Float64",
        "fecha": "string",
    },
)

# ---------------------------------------------------------------------
# Diccionario de schemas para coerción por nombre
# ---------------------------------------------------------------------
SCHEMAS: dict[str, DFSchema] = {
    "sku": SKU,
    "categoria": CATEGORIA,
    "macro_categoria": MACRO_CATEGORIA,
    "proveedor": PROVEEDOR,
    "competidor": COMPETIDOR,
    "ventas": VENTAS,
    "precio_competidor": PRECIO_COMPETIDOR,
    "segmento": SEGMENTO,
    "regla_negocio": REGLA_NEGOCIO,
    "regla_negocio_override": REGLA_NEGOCIO_OVERRIDE,
}

def coerce_by_name(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        raise KeyError(f"Schema no registrado: {schema_name}")
    return coerce_df(df, schema)
