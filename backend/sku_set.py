# sku_set.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from backend import data_layer as db

@dataclass
class SKUSet:
    _name: str

    # caches
    _fecha_inicio: Optional[str] = None
    _fecha_fin: Optional[str] = None
    _id_competidor: Optional[int] = None

    _df_reglas: Optional[pd.DataFrame] = None
    _df_master: Optional[pd.DataFrame] = None

    # Almacenamiento interno para la property venta_neta.
    _venta_neta: Optional[float] = None
    _posicionamiento: Optional[float] = None
    _margen: Optional[float] = None

    _venta_neta_inicial: Optional[float] = None
    _rep_venta_neta: Optional[float] = None
    _rep_skus: Optional[float] = None

    def __str__(self) -> str:
        return f"SKUSet(name={self._name})"

    # HELPERS
    def _promedio_ponderado(self, g, valor_pesos):
        return (g * valor_pesos).sum() / valor_pesos.sum() if valor_pesos.sum() != 0 else pd.NA

    @property
    def venta_neta(self) -> Optional[float]:
        """Venta neta accesible como property.

        Comportamiento:
        - Si se ha seteado explícitamente `_venta_neta` (no-None), lo devuelve (cache).
        - Si no, y `self.df_master` está disponible, calcula y devuelve la suma de la
          columna 'venta_neta' de `df_master` (si existe).
        - Si no hay datos disponibles devuelve None.
        """
        if self._venta_neta is not None:
            return self._venta_neta

        # Intentar calcularlo a partir de df_master si existe
        if self._df_master is not None and 'venta_neta' in self._df_master.columns:
            total = self._df_master['venta_neta'].sum()
            # si la suma da 0.0 podría ser válido; devolvemos el float
            self._venta_neta = total
            return self._venta_neta

        # No hay información disponible
        return None

    @property
    def venta_neta_inicial(self) -> Optional[float]:
        if self._venta_neta_inicial is not None:
            return self._venta_neta_inicial

        return None

    @property
    def posicionamiento(self) -> Optional[float]:
        # Cache si ya fue calculado
        if self._posicionamiento is not None:
            return self._posicionamiento

        # Si no, intentar calcularlo
        if self._df_master is None or not {'posicionamiento', 'venta_neta'}.issubset(self._df_master.columns):
            raise ValueError("No se puede calcular posicionamiento: df_master no existe o no tiene las columnas necesarias")

        pos = pd.to_numeric(self._df_master["posicionamiento"], errors="coerce")
        ven = pd.to_numeric(self._df_master["venta_neta"], errors="coerce")
        mask = pos.notna() & ven.notna() & (ven > 0)
        if not mask.any():
            return None

        numerador = float((pos[mask] * ven[mask]).sum())
        denominador = float(ven[mask].sum())
        if denominador <= 0:
            return None

        self._posicionamiento = numerador / denominador
        return self._posicionamiento

    @property
    def margen(self) -> Optional[float]:
        # Cache si ya fue calculado
        if self._margen is not None:
            return self._margen

        # Si no, intentar calcularlo
        if self._df_master is None or not {'front', 'venta_neta'}.issubset(self._df_master.columns):
            raise ValueError("No se puede calcular margen: df_master no existe o no tiene las columnas necesarias")

        # Definición de margen
        margen = (pd.to_numeric(self._df_master["front"], errors="coerce") +
                  pd.to_numeric(self._df_master["back"], errors="coerce"))

        venta = pd.to_numeric(self._df_master["venta_neta"], errors="coerce")
        mask = margen.notna() & venta.notna() & (venta > 0)
        if not mask.any():
            return None
        self._margen = self._promedio_ponderado(margen[mask], venta[mask])
        return self._margen

    @property
    def rep_venta_neta(self) -> Optional[float]:
        if self._rep_venta_neta is not None:
            return self._rep_venta_neta

        return None

    @property
    def rep_skus(self) -> Optional[float]:
        if self._rep_skus is not None:
            return self._rep_skus

        return None

    @property
    def master(self):
        if self._df_master is None:
            raise ValueError(
                "No se puede calcular posicionamiento: df_master no existe o no tiene las columnas necesarias")
        return self._df_master


    @classmethod
    def build_master(
            cls,
            fecha_inicio: Optional[str] = None,
            fecha_fin: Optional[str] = None,
            id_competidor: Optional[int] = None,
            override_reglas: Optional[bool] = True,
            name: Optional[str] = None,
    ) -> "SKUSet":
        safe_name = name or f"SKUSet_{fecha_inicio or 'NA'}_{fecha_fin or 'NA'}"
        obj = cls(
            _name=safe_name,
            _fecha_inicio = fecha_inicio,
            _fecha_fin = fecha_fin,
            _id_competidor = id_competidor,
        )
        obj._build_master()
        return obj

    def _build_master(self) -> None:
        # Obtenemos toda la maestra:
        # Skus, ventas, precios competidor, reglas de negocio, segmentos.

        # Funciones helper para agrupación
        def agrupacion_posicionamiento(g):
            return pd.Series({
                'cantidad': g['cantidad'].sum(),
                'venta_neta': g['venta_neta'].sum(),
                'ganancia_neta': g['ganancia_neta'].sum(),
                'iva': self._promedio_ponderado(g['iva'], g['venta_neta']),
                'front': self._promedio_ponderado(g['front'], g['venta_neta']),
                'back': self._promedio_ponderado(g['back'], g['venta_neta']),
                'precio_neto': self._promedio_ponderado(g['precio_neto'], g['venta_neta']),
                'precio_bruto': self._promedio_ponderado(g['precio_bruto'], g['venta_neta']),
                'precio_bruto_descuento': self._promedio_ponderado(g['precio_bruto_descuento'], g['venta_neta']),
                'precio_lleno': self._promedio_ponderado(g['precio_lleno'], g['venta_neta']),
                'precio_descuento': self._promedio_ponderado(g['precio_descuento'], g['venta_neta']),
                'posicionamiento': self._promedio_ponderado(g['posicionamiento'], g['venta_neta']),
            })

        # Obtenemos df ventas
        df_ventas = self._get_ventas()
        # Obtenemos df posicionamiento competidor
        df_competidor = self._get_competidor()
        # Obtenemos reglas de negocio y segmentos (con overrides aplicados)
        df_reglas = self._get_reglas()

        venta_neta_inicial = df_ventas['venta_neta'].sum() if 'venta_neta' in df_ventas.columns else 0.0
        skus_iniciales = df_ventas['id_sku'].nunique() if 'id_sku' in df_ventas.columns else 0

        # Merge de ventas con precios competidor
        df_master = df_ventas.merge(
            df_competidor,
            on=['id_sku', 'fecha'],
            how='inner'
        )
        # Filtra el DataFrame para mantener solo filas cuyo `posicionamiento` sea
        df_master["posicionamiento"] = (
                df_master[["precio_bruto", "precio_bruto_descuento"]].min(axis=1) / df_master[
            ["precio_lleno", "precio_descuento"]].min(axis=1).replace(0, pd.NA)
        )
        df_master = df_master[(df_master['posicionamiento'] > 0.5) & (df_master['posicionamiento'] < 2.0)]

        # Agrupamos fechas y ponderamos por venta
        df_master = df_master.groupby("id_sku").apply(agrupacion_posicionamiento).reset_index()

        # Merge de ventas con reglas de negocio
        df_master = df_master.merge(
            df_reglas,
            on='id_sku',
            how='left'
        )

        # Agregar catálogos
        df_catalogos = self._get_catalogos()
        df_master = df_master.merge(
            df_catalogos,
            on="id_sku",
            how="left"
        )

        self._df_master = df_master
        self._venta_neta_inicial = venta_neta_inicial
        self._rep_venta_neta = self.venta_neta / venta_neta_inicial if self.venta_neta and self.venta_neta > 0 else None
        self._rep_skus = df_master['id_sku'].nunique() / skus_iniciales if df_master['id_sku'].nunique() > 0 else None

    def _get_ventas(self):
        fecha_inicio = self._fecha_inicio
        fecha_fin = self._fecha_fin
        df_ventas = db.get_ventas(fecha_inicio, fecha_fin)
        if df_ventas is None or df_ventas.empty:
            return pd.DataFrame()
        return df_ventas

    def _get_competidor(self):
        fecha_inicio = self._fecha_inicio
        fecha_fin = self._fecha_fin
        id_competidor = self._id_competidor

        df_competidor = db.get_precio_competidor(fecha_inicio, fecha_fin, id_competidor)

        # Si viene vacío o None, devolver DataFrame vacío sin romper.
        if df_competidor is None or df_competidor.empty:
            return pd.DataFrame()

        # Quitar columna id si existe (evita KeyError)
        if "id" in df_competidor.columns:
            df_competidor.drop(columns=["id"], inplace=True)

        # Si faltan columnas mínimas, no intentamos deduplicar (evita errores)
        req = {"id_sku", "fecha", "precio_lleno", "precio_descuento"}
        if not req.issubset(df_competidor.columns):
            return df_competidor

        # --- Elegir competidor con menor precio efectivo por (id_sku, fecha) ---
        # precio_efectivo = min(precio_lleno, precio_descuento)
        lleno = pd.to_numeric(df_competidor["precio_lleno"], errors="coerce")
        desc = pd.to_numeric(df_competidor["precio_descuento"], errors="coerce")
        df_competidor["_precio_eff"] = pd.concat([lleno, desc], axis=1).min(axis=1)

        # Ordenar por menor precio y quedarnos con la primera fila por (id_sku, fecha)
        df_competidor = df_competidor.sort_values(["id_sku", "fecha", "_precio_eff"], ascending=True)
        df_competidor = df_competidor.drop_duplicates(subset=["id_sku", "fecha"], keep="first")

        # Limpiar helper column
        df_competidor.drop(columns=["_precio_eff"], inplace=True)

        return df_competidor

    def _get_reglas(self):

        df_sku = db.get_sku()
        df_reglas = db.get_regla_negocio()
        df_segmentos = db.get_segmento()
        df_override = db.get_regla_negocio_override()

        df_segmentos.rename(columns={"nombre": "segmento"}, inplace=True)

        df_sku.rename(columns={"id": "id_sku"}, inplace=True)
        df_sku = df_sku[["id_sku", "id_segmento"]]

        df_reglas = df_reglas.merge(
            df_sku,
            on="id_segmento")

        df_reglas = df_reglas.merge(
            df_segmentos,
            left_on='id_segmento',
            right_on='id',
            how='left'
        )
        df_reglas.drop(columns='id', inplace=True)

        # Aplicar overrides de forma segura
        if not df_override.empty:
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
        df_reglas = df_reglas[['id_sku', 'segmento', 'posicionamiento_top', 'posicionamiento_fondo', 'margen']].copy()
        df_reglas.rename(columns={'margen': 'margen_segmento'}, inplace=True)

        # limpiar variables temporales
        del df_segmentos, df_override

        return df_reglas

    def _get_catalogos(self):
        df_sku = db.get_sku()
        df_cat = db.get_categoria()
        df_macro = db.get_macro_categoria()
        df_proveedor = db.get_proveedor()

        df_sku.rename(columns={"id": "id_sku"}, inplace=True)
        df_cat.rename(columns={"id": "id_categoria", "nombre": "categoria"}, inplace=True)
        df_macro.rename(columns={"id": "id_macro", "nombre": "macro"}, inplace=True)
        df_proveedor.rename(columns={"id": "id_proveedor", "nombre": "proveedor"}, inplace=True)

        df_sku = df_sku.merge(
            df_cat,
            on="id_categoria",
            how="left"
        )
        df_sku = df_sku.merge(
            df_macro,
            on="id_macro",
            how="left"
        )
        df_sku = df_sku.merge(
            df_proveedor,
            on="id_proveedor",
            how="left"
        )
        return df_sku[["id_sku", "nombre", "sku", "categoria", "macro", "proveedor"]]