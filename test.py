import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from backend.sku_set import SKUSet

FECHA_INICIO = "2026-02-02"
FECHA_FIN = "2026-02-08"
ID_COMPETIDOR = 1

print('Hola mundo')
skuset = SKUSet.from_ventas(
    fecha_inicio=str(FECHA_INICIO),
    fecha_fin=str(FECHA_FIN),
)
skuset.get_df_posicionamiento(ID_COMPETIDOR)
skuset.get_posicionamiento()
skuset.get_reglas()

print(skuset.df_reglas)
print(skuset.posicionamiento)