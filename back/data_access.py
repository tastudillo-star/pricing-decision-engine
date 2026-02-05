"""
Back/data_access.py

OBJETIVO
- Capa de abstracción para acceder a las tablas de la base de datos como DataFrames.
- Cada engine accede a sus datos como desee, haciendo joins y filtros según necesite.

RESPONSABILIDADES
- Exponer funciones para obtener cada tabla como DataFrame.
- Permitir filtros opcionales por fecha cuando aplique.
- NO contiene lógica de negocio, solo acceso a datos.

TABLAS DISPONIBLES
- ventas_chiper: Ventas diarias de Chiper por SKU.
- precio_competidor: Precios de competidores por SKU y fecha.
- sku: Catálogo de SKUs con información básica.
- categoria: Catálogo de categorías.
- macro_categoria: Catálogo de macro categorías.
- proveedor: Catálogo de proveedores.
- segmento: Catálogo de segmentos.
- regla_negocio: Reglas de negocio por segmento.
- regla_negocio_override: Overrides de reglas por SKU.

USO
- Los engines importan las funciones que necesitan y hacen sus propios joins/filtros.
"""

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from utils.mySQLHelper import execute_mysql_query


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


# =============================================================================
# FUNCIONES DE ACCESO A TABLAS
# =============================================================================

def get_ventas_chiper(
    fecha_inicio: Optional[date] = None,
    fecha_fin: Optional[date] = None,
) -> pd.DataFrame:
    """
    Obtiene datos de ventas de Chiper.

    Args:
        fecha_inicio: Fecha inicial del rango (inclusive). Si None, sin límite inferior.
        fecha_fin: Fecha final del rango (inclusive). Si None, sin límite superior.

    Returns:
        DataFrame con columnas: id_sku, fecha, precio_bruto, venta_neta, front, back.
    """
    where_clauses = []
    if fecha_inicio:
        where_clauses.append(f"DATE(fecha) >= '{fecha_inicio.strftime('%Y-%m-%d')}'")
    if fecha_fin:
        where_clauses.append(f"DATE(fecha) <= '{fecha_fin.strftime('%Y-%m-%d')}'")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = f"""
    SELECT
        id_sku,
        DATE(fecha) AS fecha,
        precio_bruto,
        venta_neta,
        front,
        back
    FROM ventas_chiper
    {where_sql}
    """
    return _run_query(sql)


def get_precio_competidor(
    fecha_inicio: Optional[date] = None,
    fecha_fin: Optional[date] = None,
    id_competidor: Optional[int] = None,
) -> pd.DataFrame:
    """
    Obtiene datos de precios de competidores.

    Args:
        fecha_inicio: Fecha inicial del rango (inclusive).
        fecha_fin: Fecha final del rango (inclusive).
        id_competidor: ID del competidor (1, 2, 3) o None para todos.

    Returns:
        DataFrame con columnas: id_sku, id_competidor, fecha, precio_lleno, precio_descuento.
    """
    where_clauses = []
    if fecha_inicio:
        where_clauses.append(f"DATE(fecha) >= '{fecha_inicio.strftime('%Y-%m-%d')}'")
    if fecha_fin:
        where_clauses.append(f"DATE(fecha) <= '{fecha_fin.strftime('%Y-%m-%d')}'")
    if id_competidor and id_competidor in (1, 2, 3):
        where_clauses.append(f"id_competidor = {id_competidor}")
    elif id_competidor == 4:
        where_clauses.append("id_competidor IN (1, 2, 3)")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = f"""
    SELECT
        id_sku,
        id_competidor,
        DATE(fecha) AS fecha,
        precio_lleno,
        precio_descuento
    FROM precio_competidor
    {where_sql}
    """
    return _run_query(sql)


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
    return _run_query(sql)


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
    return _run_query(sql)


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
    return _run_query(sql)


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
    return _run_query(sql)


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
    return _run_query(sql)


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
    return _run_query(sql)


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
    return _run_query(sql)


# =============================================================================
# FUNCIONES HELPER PARA FECHAS
# =============================================================================

def get_fecha_rango(fecha_base: date, ventana_dias: int) -> tuple[date, date]:
    """
    Calcula el rango de fechas basado en una fecha base y ventana de días.

    Args:
        fecha_base: Fecha de referencia (fin del rango).
        ventana_dias: Número de días hacia atrás.

    Returns:
        Tupla (fecha_inicio, fecha_fin).
    """
    fecha_fin = fecha_base
    fecha_inicio = fecha_base - timedelta(days=ventana_dias - 1)
    return fecha_inicio, fecha_fin


def get_last_sunday() -> date:
    """
    Obtiene el último domingo (o hoy si es domingo).

    Returns:
        Fecha del último domingo.
    """
    today = date.today()
    days_since_sun = (today.weekday() - 6) % 7
    return today - timedelta(days=days_since_sun)
