import streamlit as st
from datetime import date
from backend.sku_set import SKUSet
from utils.auth import Auth




#=============================================================================
# CONFIGURACIÓN DE PÁGINA Y CONTROLES
#=============================================================================
st.header("Test usuario admin")

auth = Auth()
auth.require_page(category='admin', strict=True)