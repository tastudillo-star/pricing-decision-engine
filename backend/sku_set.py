# sku_set.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd
from backend import data_layer as db

@dataclass
class SKUSet:
    name: str
    df_skus: Optional[pd.DataFrame] = None

    # caches
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None

    df_ventas: Optional[pd.DataFrame] = None
    df_reglas: Optional[pd.DataFrame] = None
    df_master: Optional[pd.DataFrame] = None

    posicionamiento: Optional[float] = None

    def __str__(self) -> str:
        return f"SKUSet(name={self.name})"

    @classmethod
    def from_ventas(
        cls,
        fecha_inicio: Optional[str] = None,
        fecha_fin: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "SKUSet":

        ## Obtenemos ventas
        df_ventas = db.get_ventas(fecha_inicio, fecha_fin)
        # ids presentes en ventas
        if df_ventas.empty or "id_sku" not in df_ventas.columns:
            ids_venta = pd.Series([], dtype="int64")
        else:
            ids_venta = df_ventas["id_sku"].dropna().astype(int).drop_duplicates()

        ## Obtenemos catálogo de SKUs y filtramos por skus presentes en ventas
        df_skus = db.get_sku()
        # Normaliza la clave del catálogo a "id_sku"
        if not df_skus.empty and "id" in df_skus.columns and "id_sku" not in df_skus.columns:
            df_skus = df_skus.rename(columns={"id": "id_sku"})
        # Deja solo SKUs que están en ventas
        if not df_skus.empty and "id_sku" in df_skus.columns:
            df_skus_filtrado = df_skus[df_skus["id_sku"].isin(ids_venta)].copy()
        else:
            # fallback: si no hay catálogo usable, deja DF mínimo
            df_skus_filtrado = pd.DataFrame({"id_sku": ids_venta.values})

        safe_name = name or f"SKUSet_{fecha_inicio or 'NA'}_{fecha_fin or 'NA'}"
        obj = cls(name=safe_name, df_skus=df_skus_filtrado)
        obj.df_ventas = df_ventas.copy()
        obj.fecha_inicio = fecha_inicio
        obj.fecha_fin = fecha_fin
        return obj

    def get_df_posicionamiento(self, competidor_id=None):
        df_competidor = db.get_precio_competidor(self.fecha_inicio, self.fecha_fin, competidor_id)
        df_competidor.drop(columns=['id'], inplace=True)
        ## aqui revisar ventas sino pedir [futuro]
        df_master = self.df_ventas.merge(
            df_competidor,
            on=['id_sku', 'fecha'],
            how='inner'
        )

        # Filtra el DataFrame para mantener solo filas cuyo `posicionamiento` sea
        # estrictamente mayor que 0.5 y estrictamente menor que 2.0.
        # - Se conserva únicamente `posicionamiento` numérico en el intervalo abierto (0.5, 2.0).
        # - Cualquier `NaN` / `pd.NA` en `posicionamiento` queda excluido porque las comparaciones no son True.
        # - Valores iguales a 0.5 o 2.0, valores negativos o cero también quedan fuera (la comparación es estricta).
        # - Si `posicionamiento` se generó usando denominadores reemplazados por `pd.NA` cuando eran 0, esas filas tendrán `NA` y se descartarán aquí.
        # - Como efecto práctico, la suma de `venta_neta` tras este filtro refleja solo las ventas de las filas dentro de ese rango.
        df_master["posicionamiento"] = (
                df_master[["precio_bruto", "precio_bruto_descuento"]].min(axis=1) / df_master[
            ["precio_lleno", "precio_descuento"]].min(axis=1).replace(0, pd.NA)
        )
        df_master = df_master[(df_master['posicionamiento'] > 0.5) & (df_master['posicionamiento'] < 2.0)]

        # Agrupamos fechas y ponderamos por venta
        def promedio_ponderado(g, valor_pesos):
            return (g * valor_pesos).sum() / valor_pesos.sum() if valor_pesos.sum() != 0 else pd.NA

        def agrupacion_posicionamiento(g):
            return pd.Series({
                'cantidad': g['cantidad'].sum(),
                'venta_neta': g['venta_neta'].sum(),
                'ganancia_neta': g['ganancia_neta'].sum(),
                'iva': promedio_ponderado(g['iva'], g['venta_neta']),
                'front': promedio_ponderado(g['front'], g['venta_neta']),
                'back': promedio_ponderado(g['back'], g['venta_neta']),
                'precio_neto': promedio_ponderado(g['precio_neto'], g['venta_neta']),
                'precio_bruto': promedio_ponderado(g['precio_bruto'], g['venta_neta']),
                'precio_bruto_descuento': promedio_ponderado(g['precio_bruto_descuento'], g['venta_neta']),
                'precio_lleno': promedio_ponderado(g['precio_lleno'], g['venta_neta']),
                'posicionamiento': promedio_ponderado(g['posicionamiento'], g['venta_neta']),
            })

        df_master = df_master.groupby("id_sku").apply(agrupacion_posicionamiento).reset_index()

        self.df_master = df_master.copy()
        del df_master, df_competidor
        return True

    def get_reglas(self, override_reglas=True):

        df_segmentos = db.get_segmento()
        df_reglas = db.get_regla_negocio()
        df_override = db.get_regla_negocio_override()

        df_reglas = df_reglas.merge(
            df_segmentos,
            left_on='id_segmento',
            right_on='id',
            how='left'
        )
        df_reglas.drop(columns='id', inplace=True)

        df_reglas = self.df_skus.merge(
            df_reglas,
            left_on='id_segmento',
            right_on='id_segmento',
            how='left',
            suffixes=('_sku', '_regla')
        )

        # Aplicar overrides de forma segura
        if override_reglas and not df_override.empty:
            df_reglas = df_reglas.merge(
                df_override,
                left_on='id_sku',
                right_on='id_sku',
                how='left',
                suffixes=('', '_override')
            )
            for col in ['posicionamiento_top', 'posicionamiento_fondo', 'margen']:
                override_col = f'{col}_override'
                if override_col in df_reglas.columns:
                    # usar override cuando exista, si no, mantener el original
                    df_reglas[col] = df_reglas[override_col].combine_first(df_reglas[col])
            # eliminar columnas de override restantes
            cols_to_drop = [c for c in df_reglas.columns if c.endswith('_override')]
            if cols_to_drop:
                df_reglas.drop(columns=cols_to_drop, inplace=True)

        # Seleccionar y retornar la salida esperada
        self.df_reglas = df_reglas[['id_sku', 'nombre_regla', 'posicionamiento_top', 'posicionamiento_fondo', 'margen']].copy()
        self.df_reglas.rename(columns={'margen': 'margen_segmento'}, inplace=True)

        # limpiar variables temporales
        del df_segmentos, df_reglas, df_override

        return self.df_reglas

    def get_posicionamiento(self, competidor_id=None) -> float:
        if self.df_ventas is None:
            raise ValueError("Debe cargar las ventas primero usando from_ventas()")
        if self.df_master is None:
            self.get_posicionamiento(competidor_id)

        # Calculamos posicionamiento promedio ponderado por venta_neta
        df_master = self.df_master
        self.posicionamiento = (df_master['posicionamiento'] * df_master['venta_neta']).sum() / df_master['venta_neta'].sum()

        return self.posicionamiento

