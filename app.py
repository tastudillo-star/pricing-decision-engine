# app.py

import pandas as pd
import streamlit as st
from utils.auth import Auth

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("HOLAAAA")

st.set_page_config(
    page_title="Pricing Chiper – BI",
    page_icon="https://chiper.cl/wp-content/uploads/2023/06/cropped-favicon-192x192.png",
    layout="centered",
)


auth = Auth()
auth.require_page()



st.title("Pricing Chiper – BI")
st.caption("Acceso rápido a los módulos principales")
st.caption("version 1.0")