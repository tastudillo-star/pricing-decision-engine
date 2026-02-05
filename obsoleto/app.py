"""
Front/app.py

OBJETIVO
- Interfaz Streamlit. No calcula lógica de negocio.
- Solicita parámetros al usuario y llama a Back/main_pipeline.py
- Renderiza el DataFrame master por SKU (1 fila por SKU con todo).

ENTRADAS (desde UI)
- Ventanas/fechas/competidor (según tu modelo actual)
- Objetivo: MAX_VENTA | MAX_MARGEN | TRADE_OFF
- alpha (solo si TRADE_OFF)
- Escenario / defaults de elasticidad
- Guardrails básicos (clamps de precio, filtros)

SALIDA
- Tabla principal: master_df (por SKU)
- Vistas opcionales: filtros, export, summary simple (agregado desde master_df)

CONTRATO
- import Back.main_pipeline.run_pipeline(config) -> PipelineResult
"""

import streamlit as st
from datetime import date, timedelta

from back.schemas import PipelineConfig
from back.main_pipeline import run_pipeline

# Streamlit UI aquí (sin lógica de negocio)
st.set_page_config(page_title="Pricing Decision Engine", layout="wide")
st.title("Pricing Decision Engine")

# =============================================================================
# SIDEBAR: Parámetros de entrada
# =============================================================================
st.sidebar.header("Parámetros")

today = date.today()
days_since_sun = (today.weekday() - 6) % 7
default_sunday = today - timedelta(days=days_since_sun)

fecha_base = st.sidebar.date_input("Fecha base", value=default_sunday)
ventana_chiper = st.sidebar.number_input("Ventana Chiper (días)", min_value=1, max_value=180, value=7, step=1)
ventana_comp = st.sidebar.number_input("Ventana Competidor (días)", min_value=1, max_value=180, value=7, step=1)
id_competidor = st.sidebar.selectbox(
    "Competidor",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Central Mayorista",
        2: "Alvi",
        3: "La Oferta",
        4: "Mejor precio (min 1–3)",
    }.get(x, x),
    index=3,
)
excluir_dias_sin_venta_chiper = st.sidebar.checkbox("Excluir días sin venta Chiper", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Configuración de Elasticidad")
default_elasticidad = st.sidebar.slider(
    "Elasticidad por defecto",
    min_value=-3.0,
    max_value=0.0,
    value=-0.2,
    step=0.1,
    help="Elasticidad precio-demanda por defecto (negativo = inelástica)"
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Ejecutar Pipeline", type="primary")

# =============================================================================
# EJECUCIÓN DEL PIPELINE
# =============================================================================
if run_btn:
    cfg = PipelineConfig(
        fecha_base=fecha_base,
        id_competidor=id_competidor,
        ventana_chiper=int(ventana_chiper),
        ventana_comp=int(ventana_comp),
        excluir_dias_sin_venta_chiper=excluir_dias_sin_venta_chiper,
        default_elasticidad=default_elasticidad,
    )

    with st.spinner("Ejecutando pipeline…"):
        try:
            result = run_pipeline(cfg)
        except Exception as e:
            st.error(f"Error en pipeline: {e}")
            st.stop()

    # =========================================================================
    # MÉTRICAS RESUMEN
    # =========================================================================
    st.success(f"Pipeline completado: {result.total_skus} SKUs procesados")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total SKUs", f"{result.total_skus:,}")
    with col2:
        st.metric("SKUs con acción", f"{result.skus_con_accion:,}")
    with col3:
        st.metric("Delta Venta Total", f"${result.delta_venta_total:,.0f}")
    with col4:
        st.metric("Delta Margen Total", f"${result.delta_margen_total:,.0f}")

    # =========================================================================
    # TABS: Vistas de datos
    # =========================================================================
    tab_resumen, tab_master, tab_base, tab_opp, tab_elas, tab_act = st.tabs([
        "Resumen Final",
        "Master DF",
        "Base DF",
        "Oportunidades",
        "Elasticidad",
        "Acciones",
    ])

    with tab_resumen:
        st.subheader("Tabla Resumen de Pricing")

        # Crear vista con columnas específicas renombradas
        df_resumen = result.master_df.copy()

        # Seleccionar y renombrar columnas
        cols_map = {
            "sku": "SKU",
            "nombre": "Nombre",
            "macro": "Macro",
            "categoria": "Categoría",
            "proveedor": "Proveedor",
            "segmento": "Segmento",
            "bucket": "Bucket",
            "rol": "Rol",
            "tipo_oportunidad": "Tipo Oportunidad",
            "precio": "Precio Actual",
            "precio_recomendado": "Precio Recomendado",
            "precio_competidor": "Precio Competidor",
            "cantidad": "Cantidad Actual",
            "cantidad_nueva": "Cantidad Nueva",
            "venta_neta": "Venta Actual",
            "venta_nueva": "Venta Nueva",
            "impacto_venta": "Impacto Venta",
            "posicionamiento": "Posicionamiento Actual",
            "posicionamiento_nuevo": "Posicionamiento Nuevo",
            "margen": "Margen Actual",
        }

        # Filtrar solo las columnas que existen
        cols_disponibles = [c for c in cols_map.keys() if c in df_resumen.columns]
        df_resumen = df_resumen[cols_disponibles].rename(columns=cols_map)

        # Convertir porcentajes de formato decimal a porcentaje (multiplicar por 100)
        percent_cols = ["Posicionamiento Actual", "Posicionamiento Nuevo", "Margen Actual"]
        for col in percent_cols:
            if col in df_resumen.columns:
                df_resumen[col] = df_resumen[col] * 100

        # Aplicar formato usando pandas Styler
        formatters = {}

        # Formato currency para precios y ventas
        currency_cols = ["Precio Actual", "Precio Recomendado", "Precio Competidor", "Venta Actual", "Venta Nueva", "Impacto Venta"]
        for col in currency_cols:
            if col in df_resumen.columns:
                formatters[col] = '${:,.0f}'

        # Formato unidades para cantidades (enteros sin decimales)
        units_cols = ["Cantidad Actual", "Cantidad Nueva"]
        for col in units_cols:
            if col in units_cols:
                formatters[col] = '{:,.0f}'

        # Formato porcentaje para posicionamiento y margen
        percent_cols = ["Posicionamiento Actual", "Posicionamiento Nuevo", "Margen Actual"]
        for col in percent_cols:
            if col in percent_cols:
                formatters[col] = '{:.2f}%'

        # Aplicar formato y mostrar dataframe
        styled_df = df_resumen.style.format(formatters)
        st.dataframe(styled_df, use_container_width=True)

        csv_resumen = df_resumen.to_csv(index=False)
        st.download_button("Descargar resumen.csv", data=csv_resumen, file_name="resumen_pricing.csv", mime="text/csv")

    with tab_master:
        st.subheader("Master DataFrame (todos los datos)")
        st.dataframe(result.master_df, use_container_width=True)
        csv_master = result.master_df.to_csv(index=False)
        st.download_button("Descargar master_df.csv", data=csv_master, file_name="master_df.csv", mime="text/csv")

    with tab_base:
        st.subheader("Base DataFrame (datos crudos)")
        st.dataframe(result.base_df, use_container_width=True)

    with tab_opp:
        st.subheader("Oportunidades por SKU")
        st.dataframe(result.opp_df, use_container_width=True)

    with tab_elas:
        st.subheader("Elasticidad por SKU")
        st.dataframe(result.elas_df, use_container_width=True)

    with tab_act:
        st.subheader("Acciones recomendadas por SKU")
        st.dataframe(result.act_df, use_container_width=True)
else:
    st.info("Configura los parámetros en el sidebar y haz clic en 'Ejecutar Pipeline' para comenzar.")
