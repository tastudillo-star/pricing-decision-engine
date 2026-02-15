from datetime import date, datetime
from typing import Optional

import pandas as pd

from utils.mysql_helper import execute_mysql_query
from backend.schemas import coerce_by_name

def _run_query(sql: str) -> pd.DataFrame:
    """
    Ejecuta una query SQL y devuelve el resultado como DataFrame.

    Args:
        sql: Query SQL a ejecutar.

    Returns:
        DataFrame con los resultados. DataFrame vacío si no hay datos.
    """
    try:
        df = execute_mysql_query(sql, fetch=True)
    except TypeError:
        df = execute_mysql_query(sql)
    if df is None:
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

def _ensure_date_str(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} debe ser una cadena en formato YYYY-MM-DD")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{name} debe tener el formato YYYY-MM-DD")
    return value

# =============================================================================
# FUNCIONES DE ACCESO A TABLAS
# =============================================================================

# Básicos
def get_sku() -> pd.DataFrame:
    """
    Obtiene el catálogo de SKUs.

    Returns:
        DataFrame con columnas: id, sku, nombre, id_categoria, id_proveedor, id_segmento.
    """
    sql = """
    SELECT
        id,
        sku,
        nombre,
        id_categoria,
        id_proveedor,
        id_segmento
    FROM sku
    """
    df = _run_query(sql)
    return coerce_by_name(df, "sku")

def get_categoria() -> pd.DataFrame:
    """
    Obtiene el catálogo de categorías.

    Returns:
        DataFrame con columnas: id, nombre, id_macro.
    """
    sql = """
    SELECT
        id,
        nombre,
        id_macro
    FROM categoria
    """
    df = _run_query(sql)
    return coerce_by_name(df, "categoria")

def get_macro_categoria() -> pd.DataFrame:
    """
    Obtiene el catálogo de macro categorías.

    Returns:
        DataFrame con columnas: id, nombre.
    """
    sql = """
    SELECT
        id,
        nombre
    FROM macro_categoria
    """
    df = _run_query(sql)
    return coerce_by_name(df, "macro_categoria")

def get_proveedor() -> pd.DataFrame:
    """
    Obtiene el catálogo de proveedores.

    Returns:
        DataFrame con columnas: id, nombre.
    """
    sql = """
    SELECT
        id,
        nombre
    FROM proveedor
    """
    df = _run_query(sql)
    return coerce_by_name(df, "proveedor")

def get_competidor() -> pd.DataFrame:
    """
    Obtiene los competidores.

    Returns:
        DataFrame con columnas: id, nombre.
    """
    sql = """
    SELECT
        id,
        nombre
    FROM competidor
    """
    df = _run_query(sql)
    return coerce_by_name(df, "competidor")


# Vetnas y precios
def get_ventas(
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None
) -> pd.DataFrame:
    """
    Obtiene las ventas históricas.

    Args:
        fecha_inicio: Fecha inicial del rango (inclusive). Si None, sin límite inferior.
        fecha_fin: Fecha final del rango (inclusive). Si None, sin límite superior.

    Returns:
        DataFrame con columnas: id, id_sku, fecha, cantidad, venta_neta, ganancia_neta,
        descuento_neto, iva, front, back, precio_neto, precio_bruto, r, precio_bruto_descuento.
    """

    # Construye cláusulas WHERE según los parámetros de fecha
    where_clauses = []
    fecha_inicio = _ensure_date_str(fecha_inicio, "fecha_inicio")
    fecha_fin = _ensure_date_str(fecha_fin, "fecha_fin")
    if fecha_inicio:
        where_clauses.append(f"DATE(fecha) >= '{fecha_inicio}'")
    if fecha_fin:
        where_clauses.append(f"DATE(fecha) <= '{fecha_fin}'")
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # Ejecuta la consulta SQL con clausulas WHERE dinámicas
    sql = f"""
    SELECT
        id, id_sku, 
        fecha, 
        cantidad, 
        venta_neta, 
        ganancia_neta,
        descuento_neto, 
        iva, 
        front, 
        back, 
        precio_neto, 
        precio_bruto, 
        r,
        precio_bruto_descuento
    FROM ventas_chiper
    {where_sql}
    """
    df =  _run_query(sql)
    return coerce_by_name(df, "ventas")

def get_precio_competidor(
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None,
        id_competidor: Optional[int] = None
) -> pd.DataFrame:
    """
    Obtiene los precios de un competidor.

    Args:
        fecha_inicio: Fecha inicial del rango (inclusive).
        fecha_fin: Fecha final del rango (inclusive).
        id_competidor: ID del competidor. Si None, obtiene precios de todos los competidores.

    Returns:
        DataFrame con columnas: id, id_sku, id_competidor, fecha, precio_lleno, precio_descuento.
    """

    # Construye cláusulas WHERE según los parámetros de fecha y competidor
    where_clauses = []
    fecha_inicio = _ensure_date_str(fecha_inicio, "fecha_inicio")
    fecha_fin = _ensure_date_str(fecha_fin, "fecha_fin")
    if fecha_inicio:
        where_clauses.append(f"DATE(fecha) >= '{fecha_inicio}'")
    if fecha_fin:
        where_clauses.append(f"DATE(fecha) <= '{fecha_fin}'")
    if id_competidor:
        where_clauses.append(f"id_competidor = {id_competidor}")
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # Ejecuta la consulta SQL con clausulas WHERE dinámicas
    sql = f"""
    SELECT
        id,
        id_sku,
        id_competidor,
        fecha,
        precio_lleno,
        precio_descuento
    FROM precio_competidor
    {where_sql}
    """
    df = _run_query(sql)
    return coerce_by_name(df, "precio_competidor")


# Reglas de negocio y segmentos
def get_segmento() -> pd.DataFrame:
    """
    Obtiene el catálogo de segmentos.

    Returns:
        DataFrame con columnas: id, nombre.
    """
    sql = """
    SELECT
        id,
        nombre
    FROM segmento
    """
    df = _run_query(sql)
    return coerce_by_name(df, "segmento")

def get_regla_negocio() -> pd.DataFrame:
    """
    Obtiene las reglas de negocio por segmento (última versión por id_segmento).

    Returns:
        DataFrame con columnas: id_segmento, posicionamiento_top,
        posicionamiento_fondo, margen, fecha.
    """
    sql = """
    SELECT rn.id_segmento, rn.posicionamiento_top, rn.posicionamiento_fondo, rn.margen, rn.fecha
    FROM regla_negocio rn
    INNER JOIN (
        SELECT id_segmento, MAX(fecha) AS max_fecha
        FROM regla_negocio
        GROUP BY id_segmento
    ) latest ON rn.id_segmento = latest.id_segmento AND rn.fecha = latest.max_fecha
    """
    df = _run_query(sql)
    return coerce_by_name(df, "regla_negocio")

def get_regla_negocio_override() -> pd.DataFrame:
    """
    Obtiene los overrides de reglas por SKU (última versión por id_sku).

    Returns:
        DataFrame con columnas: id_sku, posicionamiento_top,
        posicionamiento_fondo, margen, fecha.
    """
    sql = """
    SELECT ro.id_sku, ro.posicionamiento_top, ro.posicionamiento_fondo, ro.margen, ro.fecha
    FROM regla_negocio_override ro
    INNER JOIN (
        SELECT id_sku, MAX(fecha) AS max_fecha
        FROM regla_negocio_override
        GROUP BY id_sku
    ) latest ON ro.id_sku = latest.id_sku AND ro.fecha = latest.max_fecha
    """
    df = _run_query(sql)
    return coerce_by_name(df, "regla_negocio_override")

