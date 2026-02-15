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
    df_ventas: Optional[pd.DataFrame] = None
    df_posicionamiento: Optional[pd.DataFrame] = None
    df_reglas: Optional[pd.DataFrame] = None
    df_override: Optional[pd.DataFrame] = None

    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None

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

    def get_posicionamiento(self, competidor_id=None):
        df_competidor = db.get_precio_competidor(self.fecha_inicio, self.fecha_fin, competidor_id)
        df_competidor.drop(columns=['id'], inplace=True)
        ## aqui revisar ventas sino pedir [futuro]
        df_merged = self.df_ventas.merge(
            df_competidor,
            on=['id_sku', 'fecha'],
            how='inner'
        )
        # Guardrail por ratio contra precio_bruto (0.5x a 2.0x)
        ratio = df_merged["precio_descuento"] / df_merged["precio_bruto"]
        df_merged = df_merged[(ratio >= 0.5) & (ratio <= 2.0)]
        # Cáclulo de la columna posicionamiento usando precio_descuento
        df_merged['posicionamiento'] = df_merged['precio_bruto'].div(df_merged['precio_descuento'])
        self.df_posicionamiento = df_merged[['id_sku', 'fecha', 'posicionamiento', 'id_competidor']].copy()
        del df_competidor, df_merged

        return self.df_posicionamiento

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

    def get_master_pos(
        self,
        competidor_id=None
    ) -> float:
        if self.df_ventas is None:
            raise ValueError("Debe cargar las ventas primero usando from_ventas()")
        if self.df_posicionamiento is None:
            self.get_posicionamiento(competidor_id)

        # df_master IMPORTANTE
        df_master = self.df_ventas.merge(
            self.df_posicionamiento,
            on=['id_sku', 'fecha'],
            how='left'
        )
        # Filtro por ventas y pos válido
        df_master = df_master[df_master['venta_neta'].notna() & (df_master['venta_neta'] > 0)]
        df_master = df_master.dropna(subset=["posicionamiento", "id_competidor"])

        # KPIS
        df = df_master.copy()

        # 2) Venta por SKU-día (para no duplicarla por competidor)
        venta_sku_dia = (
            df.groupby(["fecha", "id_sku"], as_index=False)
            .agg(venta_neta=("venta_neta", "first"))  # si es duplicada idéntica por join; si no, usa "sum"
        )

        # 3) Posicionamiento máximo por SKU-día entre competidores
        pos_max_sku_dia = (
            df.groupby(["fecha", "id_sku"], as_index=False)
            .agg(pos_max=("posicionamiento", "max"))
        )

        # 4) Join y promedio ponderado total
        m = venta_sku_dia.merge(pos_max_sku_dia, on=["fecha", "id_sku"], how="inner")
        pos_total = (m["pos_max"] * m["venta_neta"]).sum() / m["venta_neta"].sum()

        return pos_total

    def get_master_pos_v2(
        self,
        competidor_id=None
    ) -> float:
        if self.df_ventas is None:
            raise ValueError("Debe cargar las ventas primero usando from_ventas()")
        if self.df_posicionamiento is None:
            self.get_posicionamiento(competidor_id)

        # df_master IMPORTANTE
        df_master = self.df_ventas.merge(
            self.df_posicionamiento,
            on=['id_sku', 'fecha'],
            how='left'
        )
        # Filtro por ventas y pos válido
        df_master = df_master[df_master['venta_neta'].notna() & (df_master['venta_neta'] > 0)]
        df_master = df_master.dropna(subset=["posicionamiento"])

        # KPIS
        df = df_master.copy()

        pos = (df['posicionamiento'] * df['venta_neta']).sum() / df['venta_neta'].sum()

        print(df)
        print(len(df))
        print(pos)

        venta = df['venta_neta'].sum()
        print(f"Venta total: {venta}")


        # 2) Venta por SKU-día (para no duplicarla por competidor)
        venta_sku_dia = (
            df.groupby(["fecha", "id_sku"], as_index=False)
            .agg(venta_neta=("venta_neta", "first"))  # si es duplicada idéntica por join; si no, usa "sum"
        )

        # 3) Posicionamiento máximo por SKU-día entre competidores
        pos_max_sku_dia = (
            df.groupby(["fecha", "id_sku"], as_index=False)
            .agg(pos_max=("posicionamiento", "mean"))
        )

        # 4) Join y promedio ponderado total
        m = venta_sku_dia.merge(pos_max_sku_dia, on=["fecha", "id_sku"], how="inner")

        pos_total = (m["pos_max"] * m["venta_neta"]).sum() / m["venta_neta"].sum()

        return pos_total


