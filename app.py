# app.py


from backend import data_layer as db


#print(db.get_ventas('2026-01-01', '2026-01-31'))
print(db.get_precio_competidor('2026-01-01', '2026-01-31'))

