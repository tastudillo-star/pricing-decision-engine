"""
Front/app.py

OBJETIVO
- Interfaz Streamlit. No calcula lógica de negocio.
- Solicita parámetros al usuario y llama a Back/main_pipeline.py
- Renderiza tabla pivote por Macro Categoría y Categoría.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from back.schemas import PipelineConfig
from back.main_pipeline import run_pipeline

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(page_title="Pricing Decision Engine", layout="wide")
st.title("Pricing Decision Engine")

# Constantes para cuadrantes (segmento actual basado en posicionamiento/margen)
POS_REF = 1.0  # 100% = precio igual al competidor
MARGEN_REF = 0.1772  # 17.72% margen referencia

QUADRANT_NAMES = {
    "Q1": "Contribuyente",   # pos >= 100%, margen >= ref
    "Q2": "Poderosa",        # pos < 100%, margen >= ref
    "Q3": "Magnética",       # pos < 100%, margen < ref
    "Q4": "Oportunista",     # pos >= 100%, margen < ref
}

# =============================================================================
# CONFIGURACIÓN DE TABLAS
# =============================================================================
TABLE_HEIGHT_PIVOT = 600  # Altura de la tabla pivote agrupable
TABLE_HEIGHT_SKU = 600    # Altura de la tabla de detalle SKU

def get_quadrant(pos: float, margen: float) -> str:
    """Determina el cuadrante basado en posicionamiento y margen."""
    if pd.isna(pos) or pd.isna(margen):
        return "N/A"
    if pos >= POS_REF and margen >= MARGEN_REF:
        return "Contribuyente"
    elif pos < POS_REF and margen >= MARGEN_REF:
        return "Poderosa"
    elif pos < POS_REF and margen < MARGEN_REF:
        return "Magnética"
    else:
        return "Oportunista"

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

run_btn = st.sidebar.button("Ejecutar Pipeline", type="primary")

# =============================================================================
# INICIALIZAR SESSION STATE
# =============================================================================
if 'pipeline_executed' not in st.session_state:
    st.session_state.pipeline_executed = False
    st.session_state.result = None

# =============================================================================
# FUNCIONES DE CÁLCULO PARA TABLA PIVOTE
# =============================================================================

def prepare_sku_table(df_skus: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara la tabla de SKUs para mostrar con las columnas requeridas.

    Columnas:
    - SKU
    - Nombre
    - Segmento
    - Bucket
    - Posicionamiento (actual, %)
    - Pos. Objetivo (posicionamiento objetivo según reglas)
    - Precio Chiper
    - Precio Competidor
    - Precio Final (editable, calculado desde pos_objetivo si no hay precio_recomendado)
    - Pos. Final (igual a pos_objetivo inicialmente)
    """
    df = df_skus.copy()

    result_data = []

    for _, row in df.iterrows():
        # SKU y nombre
        sku = str(row.get('sku', ''))
        nombre = str(row.get('nombre', ''))
        segmento = str(row.get('segmento', 'Sin segmento') or 'Sin segmento')
        bucket = str(row.get('bucket', 'N/A') or 'N/A')

        # Posicionamiento actual (convertir a porcentaje para display)
        pos_actual = row.get('posicionamiento', np.nan)
        if pd.notna(pos_actual):
            pos_actual = float(pos_actual) * 100  # Convertir a %

        # Posicionamiento objetivo (si existe)
        pos_objetivo_raw = row.get('posicionamiento_objetivo', row.get('posicionamiento_rol', np.nan))
        if pd.notna(pos_objetivo_raw):
            pos_objetivo = float(pos_objetivo_raw) * 100  # Convertir a %
        else:
            pos_objetivo = 100.0  # Default a 100%

        # Precios - usar pd.to_numeric para conversión robusta
        precio_chiper_raw = row.get('precio', np.nan)
        precio_competidor_raw = row.get('precio_competidor', np.nan)

        # Convertir a numérico de forma robusta (maneja strings, None, etc.)
        precio_competidor = pd.to_numeric(precio_competidor_raw, errors='coerce')
        precio_chiper = pd.to_numeric(precio_chiper_raw, errors='coerce')

        # Validar que sean positivos
        if pd.notna(precio_competidor) and precio_competidor <= 0:
            precio_competidor = np.nan

        if pd.notna(precio_chiper) and precio_chiper <= 0:
            precio_chiper = np.nan

        # Precio recomendado del pipeline (si existe) - no lo usamos pero lo dejamos por si acaso
        precio_recomendado_raw = row.get('precio_recomendado', np.nan)
        precio_recomendado = pd.to_numeric(precio_recomendado_raw, errors='coerce')
        if pd.notna(precio_recomendado) and precio_recomendado <= 0:
            precio_recomendado = np.nan

        # Determinar Precio Final y Pos. Final
        # PRIORIDAD:
        # 1. Si hay precio_competidor válido -> usar pos_objetivo para calcular precio_final
        # 2. Si NO hay precio_competidor -> precio_final = precio_chiper, pos_final = N/A

        if pd.notna(precio_competidor):
            # Tenemos precio competidor, podemos calcular posicionamiento
            # Pos. Final = Pos. Objetivo (el objetivo que queremos alcanzar)
            pos_final = pos_objetivo
            # Precio Final = Precio Competidor * (Pos. Objetivo / 100)
            precio_final = precio_competidor * (pos_objetivo / 100)
        else:
            # Sin precio competidor, no podemos calcular posicionamiento
            precio_final = precio_chiper if pd.notna(precio_chiper) else np.nan
            pos_final = np.nan

        result_data.append({
            'SKU': sku,
            'Nombre': nombre,
            'Segmento': segmento,
            'Bucket': bucket,
            'Posicionamiento': pos_actual,
            'Pos. Objetivo': pos_objetivo,
            'Precio Chiper': precio_chiper,
            'Precio Competidor': precio_competidor,
            'Precio Final': precio_final,
            'Pos. Final': pos_final,
        })

    return pd.DataFrame(result_data)

def compute_pivot_table(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la tabla pivote por Macro Categoría y Categoría.

    Columnas:
    - macro / categoria
    - segmento_actual: cuadrante basado en posicionamiento y margen ponderados
    - segmento_objetivo: segmento con mayor venta dentro del grupo
    - posicionamiento: posicionamiento ponderado por peso de venta
    - dispersion_pos: desviación estándar del posicionamiento
    """
    if master_df is None or master_df.empty:
        return pd.DataFrame()

    df = master_df.copy()

    # Asegurar columnas numéricas
    numeric_cols = ['posicionamiento', 'venta_neta', 'margen']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rellenar valores faltantes
    df['macro'] = df.get('macro', pd.Series(['Sin macro'] * len(df))).fillna('Sin macro')
    df['categoria'] = df.get('categoria', pd.Series(['Sin categoría'] * len(df))).fillna('Sin categoría')
    df['segmento'] = df.get('segmento', pd.Series(['Sin segmento'] * len(df))).fillna('Sin segmento')

    results = []

    # Calcular para cada Macro Categoría
    for macro in df['macro'].unique():
        macro_data = df[df['macro'] == macro]
        macro_row = compute_group_metrics(macro_data, macro, None)
        results.append(macro_row)

        # Calcular para cada Categoría dentro de la Macro
        for categoria in macro_data['categoria'].unique():
            cat_data = macro_data[macro_data['categoria'] == categoria]
            cat_row = compute_group_metrics(cat_data, macro, categoria)
            results.append(cat_row)

    return pd.DataFrame(results)

def compute_group_metrics(group_df: pd.DataFrame, macro: str, categoria: str = None) -> dict:
    """
    Calcula métricas para un grupo (macro o categoría).
    """
    venta_total = group_df['venta_neta'].sum() if 'venta_neta' in group_df.columns else 0

    # Posicionamiento ponderado por venta
    if venta_total > 0 and 'posicionamiento' in group_df.columns:
        pos_pond = (group_df['posicionamiento'] * group_df['venta_neta']).sum() / venta_total
    else:
        pos_pond = np.nan

    # Margen ponderado por venta
    if venta_total > 0 and 'margen' in group_df.columns:
        margen_pond = (group_df['margen'] * group_df['venta_neta']).sum() / venta_total
    else:
        margen_pond = np.nan

    # Segmento actual (cuadrante basado en pos/margen ponderados)
    segmento_actual = get_quadrant(pos_pond, margen_pond)

    # Segmento objetivo (segmento con mayor venta)
    if 'segmento' in group_df.columns and 'venta_neta' in group_df.columns:
        seg_ventas = group_df.groupby('segmento')['venta_neta'].sum()
        segmento_objetivo = seg_ventas.idxmax() if not seg_ventas.empty else 'N/A'
    else:
        segmento_objetivo = 'N/A'

    # Dispersión de posicionamiento (desviación estándar)
    if 'posicionamiento' in group_df.columns:
        dispersion = group_df['posicionamiento'].std(ddof=0)
    else:
        dispersion = np.nan

    return {
        'nivel': 'Macro' if categoria is None else 'Categoría',
        'macro_categoria': macro if categoria is None else f"  ↳ {categoria}",
        'segmento_actual': segmento_actual,
        'segmento_objetivo': segmento_objetivo,
        'posicionamiento': pos_pond,
        'dispersion_pos': dispersion,
        '_macro': macro,
        '_categoria': categoria,
    }

# =============================================================================
# EJECUCIÓN DEL PIPELINE (solo al hacer clic en el botón)
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
            st.session_state.result = result
            st.session_state.pipeline_executed = True
        except Exception as e:
            st.error(f"Error en pipeline: {e}")
            st.stop()

# =============================================================================
# MOSTRAR RESULTADOS (se ejecuta siempre que haya datos en session_state)
# =============================================================================
if st.session_state.pipeline_executed and st.session_state.result is not None:
    result = st.session_state.result

    # =========================================================================
    # MÉTRICAS RESUMEN
    # =========================================================================

    # =============================================================================
    # CREAR PESTAÑAS PARA ANÁLISIS
    # =============================================================================
    tab_pivot, tab_detalle = st.tabs(["Análisis por Categoría", "Detalle por SKU"])

    # =============================================================================
    # PESTAÑA 1: TABLA PIVOTE
    # =============================================================================
    with tab_pivot:
        st.subheader("Análisis por Macro Categoría / Categoría")

        df_pivot = compute_pivot_table(result.master_df)

        if not df_pivot.empty:
            # Preparar DataFrame para mostrar
            df_display = df_pivot[['macro_categoria', 'segmento_actual', 'segmento_objetivo',
                                   'posicionamiento', 'dispersion_pos']].copy()

            # Renombrar columnas
            df_display.columns = [
                'Macro Categoría / Categoría',
                'Segmento Actual',
                'Segmento Objetivo',
                'Posicionamiento',
                'Dispersión Pos.'
            ]

            # Formatear valores
            formatters = {
                'Posicionamiento': '{:.2%}',
                'Dispersión Pos.': '{:.4f}',
            }

            # Aplicar estilo según nivel (macro vs categoría)
            def style_row(row):
                if not row['Macro Categoría / Categoría'].startswith('  '):
                    return ['font-weight: bold; background-color: #f0f2f6'] * len(row)
                return [''] * len(row)

            styled_df = df_display.style\
                .format(formatters, na_rep='N/A')\
                .apply(style_row, axis=1)

            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=TABLE_HEIGHT_PIVOT)

            # Botón de descarga
            csv_pivot = df_display.to_csv(index=False)
            st.download_button(
                "Descargar tabla pivote (CSV)",
                data=csv_pivot,
                file_name="pivot_macro_categoria.csv",
                mime="text/csv"
            )
        else:
            st.warning("No hay datos suficientes para generar la tabla pivote.")

    # =============================================================================
    # PESTAÑA 2: DETALLE SKU
    # =============================================================================
    with tab_detalle:
        st.subheader("Detalle por SKU")

        # Filtros (sin reinicio porque session_state persiste los datos)
        col_f1, col_f2 = st.columns(2)

        df_for_filters = result.master_df.copy()
        df_for_filters['macro'] = df_for_filters.get('macro', pd.Series(['Sin macro'] * len(df_for_filters))).fillna('Sin macro')
        df_for_filters['categoria'] = df_for_filters.get('categoria', pd.Series(['Sin categoría'] * len(df_for_filters))).fillna('Sin categoría')

        macros_list = sorted(df_for_filters['macro'].unique().tolist())

        with col_f1:
            sel_macro = st.selectbox(
                "Seleccionar Macro Categoría",
                options=['Todas'] + macros_list,
                index=0,
                key="filter_macro"
            )

        # Filtrar categorías según macro seleccionada
        if sel_macro != 'Todas':
            cats_filtered = sorted(df_for_filters[df_for_filters['macro'] == sel_macro]['categoria'].unique().tolist())
        else:
            cats_filtered = sorted(df_for_filters['categoria'].unique().tolist())

        with col_f2:
            sel_categoria = st.selectbox(
                "Seleccionar Categoría",
                options=['Todas'] + cats_filtered,
                index=0,
                key="filter_categoria"
            )

        # Aplicar filtros al dataframe
        df_skus = result.master_df.copy()
        df_skus['macro'] = df_skus.get('macro', pd.Series(['Sin macro'] * len(df_skus))).fillna('Sin macro')
        df_skus['categoria'] = df_skus.get('categoria', pd.Series(['Sin categoría'] * len(df_skus))).fillna('Sin categoría')

        if sel_macro != 'Todas':
            df_skus = df_skus[df_skus['macro'] == sel_macro]
        if sel_categoria != 'Todas':
            df_skus = df_skus[df_skus['categoria'] == sel_categoria]

        if df_skus.empty:
            st.warning("No hay SKUs para la selección actual.")
        else:
            # Preparar tabla de SKUs desde datos originales
            df_sku_table = prepare_sku_table(df_skus)

            # CLAVE GLOBAL para guardar TODOS los cambios (independiente del filtro)
            edits_key = "sku_edits_global"

            # Inicializar diccionario global de ediciones si no existe
            if edits_key not in st.session_state:
                st.session_state[edits_key] = {}

            # Aplicar ediciones guardadas a la tabla actual
            edits_dict = st.session_state[edits_key]

            df_to_edit = df_sku_table.copy()
            for idx in df_to_edit.index:
                sku = df_to_edit.loc[idx, 'SKU']
                if sku in edits_dict:
                    # Restaurar valores editados previamente para este SKU
                    if 'Precio Final' in edits_dict[sku]:
                        df_to_edit.loc[idx, 'Precio Final'] = edits_dict[sku]['Precio Final']
                    if 'Pos. Final' in edits_dict[sku]:
                        df_to_edit.loc[idx, 'Pos. Final'] = edits_dict[sku]['Pos. Final']

            # =================================================================
            # KPIs DINÁMICOS (se actualizan con cambios de precio/posición)
            # =================================================================
            # Obtener venta_neta para ponderación
            df_skus_venta = df_skus[['sku', 'venta_neta', 'segmento', 'margen']].copy()
            df_skus_venta['sku'] = df_skus_venta['sku'].astype(str)
            df_skus_venta['venta_neta'] = pd.to_numeric(df_skus_venta['venta_neta'], errors='coerce').fillna(0)
            df_skus_venta['margen'] = pd.to_numeric(df_skus_venta['margen'], errors='coerce').fillna(0)

            # Crear mapeo SKU -> venta_neta y margen
            venta_map = dict(zip(df_skus_venta['sku'], df_skus_venta['venta_neta']))
            margen_map = dict(zip(df_skus_venta['sku'], df_skus_venta['margen']))
            segmento_map = dict(zip(df_skus_venta['sku'], df_skus_venta['segmento']))

            # Agregar venta_neta a df_to_edit para cálculos
            df_to_edit['venta_neta'] = df_to_edit['SKU'].map(venta_map).fillna(0)
            df_to_edit['margen'] = df_to_edit['SKU'].map(margen_map).fillna(0)
            df_to_edit['segmento_sku'] = df_to_edit['SKU'].map(segmento_map).fillna('Sin segmento')

            # Calcular KPIs ponderados por venta
            venta_total = df_to_edit['venta_neta'].sum()

            if venta_total > 0:
                # Posicionamiento Inicial ponderado (convertir de % a decimal para cálculo)
                # Filtrar solo filas con posicionamiento válido
                df_valid_inicial = df_to_edit[df_to_edit['Posicionamiento'].notna() & (df_to_edit['venta_neta'] > 0)].copy()
                venta_valid_inicial = df_valid_inicial['venta_neta'].sum()

                if venta_valid_inicial > 0:
                    pos_inicial_decimal = df_valid_inicial['Posicionamiento'] / 100
                    pos_inicial_ponderado = (pos_inicial_decimal * df_valid_inicial['venta_neta']).sum() / venta_valid_inicial

                    # Dispersión de Posicionamiento Inicial (desviación estándar ponderada)
                    if len(df_valid_inicial) > 1:
                        mean_pos_inicial = pos_inicial_ponderado
                        variance_inicial = ((pos_inicial_decimal - mean_pos_inicial) ** 2 * df_valid_inicial['venta_neta']).sum() / venta_valid_inicial
                        dispersion_inicial = np.sqrt(variance_inicial)
                    else:
                        dispersion_inicial = 0.0
                else:
                    pos_inicial_ponderado = np.nan
                    dispersion_inicial = np.nan

                # Posicionamiento Final ponderado (convertir de % a decimal para cálculo)
                # Filtrar solo filas con pos final válido
                df_valid_final = df_to_edit[df_to_edit['Pos. Final'].notna() & (df_to_edit['venta_neta'] > 0)].copy()
                venta_valid_final = df_valid_final['venta_neta'].sum()

                if venta_valid_final > 0:
                    pos_final_decimal = df_valid_final['Pos. Final'] / 100
                    pos_final_ponderado = (pos_final_decimal * df_valid_final['venta_neta']).sum() / venta_valid_final

                    # Dispersión de Posicionamiento Final (desviación estándar ponderada)
                    if len(df_valid_final) > 1:
                        mean_pos = pos_final_ponderado
                        variance = ((pos_final_decimal - mean_pos) ** 2 * df_valid_final['venta_neta']).sum() / venta_valid_final
                        dispersion_final = np.sqrt(variance)
                    else:
                        dispersion_final = 0.0
                else:
                    pos_final_ponderado = np.nan
                    dispersion_final = np.nan

                # Margen ponderado para calcular segmento final
                margen_ponderado = (df_to_edit['margen'] * df_to_edit['venta_neta']).sum() / venta_total

                # Segmento Final (cuadrante basado en pos/margen ponderados finales)
                segmento_final = get_quadrant(pos_final_ponderado, margen_ponderado)

                # Segmento Objetivo (segmento con mayor venta)
                seg_ventas = df_to_edit.groupby('segmento_sku')['venta_neta'].sum()
                segmento_objetivo = seg_ventas.idxmax() if not seg_ventas.empty else 'N/A'
            else:
                pos_final_ponderado = np.nan
                dispersion_inicial = np.nan
                dispersion_final = np.nan
                segmento_final = 'N/A'
                segmento_objetivo = 'N/A'

            # Quitar columnas auxiliares antes de mostrar
            df_to_edit_display = df_to_edit.drop(columns=['venta_neta', 'margen', 'segmento_sku'], errors='ignore')

            # Mostrar tabla editable
            st.markdown("**Tabla de SKUs** — Edita *Precio Final* o *Pos. Final* y el otro se recalcula automáticamente")

            edited_df = st.data_editor(
                df_to_edit_display,
                column_config={
                    "SKU": st.column_config.TextColumn("SKU", disabled=True, width="small"),
                    "Nombre": st.column_config.TextColumn("Nombre", disabled=True, width="large"),
                    "Segmento": st.column_config.TextColumn("Segmento", disabled=True, width="small"),
                    "Bucket": st.column_config.TextColumn("Bucket", disabled=True, width="small"),
                    "Posicionamiento": st.column_config.NumberColumn(
                        "Posicionamiento",
                        disabled=True,
                        format="%.2f%%",
                        width="small"
                    ),
                    "Pos. Objetivo": st.column_config.NumberColumn(
                        "Pos. Objetivo",
                        disabled=True,
                        format="%.2f%%",
                        width="small"
                    ),
                    "Precio Chiper": st.column_config.NumberColumn(
                        "Precio Chiper",
                        disabled=True,
                        format="$%.0f",
                        width="small"
                    ),
                    "Precio Competidor": st.column_config.NumberColumn(
                        "Precio Competidor",
                        disabled=True,
                        format="$%.0f",
                        width="small"
                    ),
                    "Precio Final": st.column_config.NumberColumn(
                        "Precio Final",
                        min_value=0,
                        format="$%.0f",
                        width="small",
                        required=True
                    ),
                    "Pos. Final": st.column_config.NumberColumn(
                        "Pos. Final",
                        min_value=0,
                        max_value=500,
                        format="%.2f%%",
                        width="small",
                        required=True
                    ),
                },
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                height=TABLE_HEIGHT_SKU,
            )

            # Detectar cambios y recalcular
            if edited_df is not None:
                df_recalculado = edited_df.copy()
                hubo_cambios = False

                # Comparar con df_to_edit_display (que ya tiene las ediciones previas aplicadas)
                for idx in df_recalculado.index:
                    sku = df_recalculado.loc[idx, 'SKU']
                    precio_comp = df_recalculado.loc[idx, 'Precio Competidor']

                    # IGNORAR filas sin precio competidor válido
                    if pd.isna(precio_comp) or precio_comp <= 0:
                        continue

                    precio_actual = df_recalculado.loc[idx, 'Precio Final']
                    precio_anterior = df_to_edit_display.loc[idx, 'Precio Final']

                    pos_actual = df_recalculado.loc[idx, 'Pos. Final']
                    pos_anterior = df_to_edit_display.loc[idx, 'Pos. Final']

                    # IGNORAR si los valores actuales no son válidos
                    if pd.isna(precio_actual) or pd.isna(pos_actual):
                        continue

                    # IGNORAR si los valores anteriores no son válidos
                    if pd.isna(precio_anterior) or pd.isna(pos_anterior):
                        continue

                    # Verificar si cambió el precio
                    precio_cambio = abs(precio_actual - precio_anterior) > 0.01

                    # Verificar si cambió el posicionamiento
                    pos_cambio = abs(pos_actual - pos_anterior) > 0.01

                    if precio_cambio:
                        # Recalcular Pos. Final desde Precio Final
                        df_recalculado.loc[idx, 'Pos. Final'] = (precio_actual / precio_comp) * 100

                        # Guardar edición en diccionario global
                        if sku not in edits_dict:
                            edits_dict[sku] = {}
                        edits_dict[sku]['Precio Final'] = precio_actual
                        edits_dict[sku]['Pos. Final'] = df_recalculado.loc[idx, 'Pos. Final']
                        hubo_cambios = True

                    elif pos_cambio:
                        # Recalcular Precio Final desde Pos. Final
                        df_recalculado.loc[idx, 'Precio Final'] = (pos_actual / 100) * precio_comp

                        # Guardar edición en diccionario global
                        if sku not in edits_dict:
                            edits_dict[sku] = {}
                        edits_dict[sku]['Precio Final'] = df_recalculado.loc[idx, 'Precio Final']
                        edits_dict[sku]['Pos. Final'] = pos_actual
                        hubo_cambios = True

                # Guardar y forzar rerun solo si hubo cambios
                if hubo_cambios:
                    st.session_state[edits_key] = edits_dict
                    st.rerun()

            # =================================================================
            # KPIs DINÁMICOS (debajo de la tabla, se actualizan con cambios)
            # =================================================================
            st.markdown("---")
            kpi_cols = st.columns(5)

            with kpi_cols[0]:
                st.metric(
                    label="Segmento Objetivo",
                    value=segmento_objetivo,
                    help="Segmento con mayor peso de venta en la selección"
                )

            with kpi_cols[1]:
                st.metric(
                    label="Segmento Final",
                    value=segmento_final,
                    help="Cuadrante basado en posicionamiento y margen ponderados finales"
                )

            with kpi_cols[2]:
                pos_display = f"{pos_final_ponderado:.2%}" if pd.notna(pos_final_ponderado) else "N/A"
                st.metric(
                    label="Posicionamiento Final",
                    value=pos_display,
                    help="Posicionamiento ponderado por peso de venta"
                )

            with kpi_cols[3]:
                disp_inicial_display = f"{dispersion_inicial:.4f}" if pd.notna(dispersion_inicial) else "N/A"
                st.metric(
                    label="Dispersión Inicial",
                    value=disp_inicial_display,
                    help="Desviación estándar del posicionamiento inicial ponderada"
                )

            with kpi_cols[4]:
                disp_display = f"{dispersion_final:.4f}" if pd.notna(dispersion_final) else "N/A"
                st.metric(
                    label="Dispersión Final",
                    value=disp_display,
                    help="Desviación estándar del posicionamiento final ponderada"
                )

            st.markdown("---")

            # Detectar cambios para mostrar resumen
            if edited_df is not None:
                df_recalculado = edited_df.copy()

                # Mostrar resumen de TODOS los cambios (comparar con datos originales por SKU)
                original_dict = df_sku_table.set_index('SKU').to_dict('index')

                cambios_list = []
                for idx in df_recalculado.index:
                    sku = df_recalculado.loc[idx, 'SKU']
                    precio_comp = df_recalculado.loc[idx, 'Precio Competidor']

                    # IGNORAR filas sin precio competidor válido
                    if pd.isna(precio_comp) or precio_comp <= 0:
                        continue

                    if sku not in original_dict:
                        continue

                    original = original_dict[sku]

                    precio_orig = original.get('Precio Final')
                    pos_orig = original.get('Pos. Final')
                    precio_now = df_recalculado.loc[idx, 'Precio Final']
                    pos_now = df_recalculado.loc[idx, 'Pos. Final']

                    # IGNORAR si algún valor no es válido
                    if pd.isna(precio_orig) or pd.isna(precio_now):
                        continue
                    if pd.isna(pos_orig) or pd.isna(pos_now):
                        continue

                    # Comparar con tolerancia
                    precio_diff = abs(precio_now - precio_orig) > 0.01
                    pos_diff = abs(pos_now - pos_orig) > 0.01

                    if precio_diff or pos_diff:
                        cambios_list.append(idx)

                if cambios_list:
                    cambios = df_recalculado.loc[cambios_list].copy()
                    st.success(f"{len(cambios)} SKU(s) modificado(s) en esta vista — Valores recalculados automáticamente")

                    with st.expander("Ver detalle de cambios"):
                        cambios_display = cambios[['SKU', 'Nombre', 'Precio Chiper', 'Precio Final',
                                                    'Precio Competidor', 'Posicionamiento', 'Pos. Final']].copy()

                        cambios_display['Δ Precio'] = cambios_display['Precio Final'] - cambios_display['Precio Chiper']
                        cambios_display['Δ Precio %'] = np.where(
                            cambios_display['Precio Chiper'] > 0,
                            (cambios_display['Precio Final'] / cambios_display['Precio Chiper'] - 1) * 100,
                            np.nan
                        )
                        cambios_display['Δ Pos.'] = cambios_display['Pos. Final'] - cambios_display['Posicionamiento']

                        cols_order = ['SKU', 'Nombre', 'Precio Chiper', 'Precio Final', 'Δ Precio', 'Δ Precio %',
                                      'Precio Competidor', 'Posicionamiento', 'Pos. Final', 'Δ Pos.']
                        cambios_display = cambios_display[[c for c in cols_order if c in cambios_display.columns]]

                        st.dataframe(
                            cambios_display.style.format({
                                'Precio Chiper': '${:,.0f}',
                                'Precio Final': '${:,.0f}',
                                'Precio Competidor': '${:,.0f}',
                                'Posicionamiento': '{:.2f}%',
                                'Pos. Final': '{:.2f}%',
                                'Δ Precio': '${:+,.0f}',
                                'Δ Precio %': '{:+.2f}%',
                                'Δ Pos.': '{:+.2f}%',
                            }, na_rep='N/A'),
                            use_container_width=True,
                            hide_index=True
                        )

                # Mostrar total de ediciones globales
                total_edits = len(st.session_state[edits_key])
                if total_edits > 0:
                    st.info(f"Total de SKUs editados (todas las categorías): {total_edits}")

            # Botón de descarga
            df_download = edited_df if edited_df is not None else df_to_edit_display
            csv_skus = df_download.to_csv(index=False)
            st.download_button(
                "Descargar tabla SKUs (CSV)",
                data=csv_skus,
                file_name=f"skus_{sel_macro}_{sel_categoria}.csv",
                mime="text/csv"
            )

            # Botón para limpiar todas las ediciones
            if st.session_state.get(edits_key):
                if st.button("Limpiar todas las ediciones", type="secondary"):
                    st.session_state[edits_key] = {}
                    st.rerun()

else:
    st.info("Configura los parámetros en el sidebar y haz clic en 'Ejecutar Pipeline' para comenzar.")

