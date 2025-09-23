# -*- coding: utf-8 -*-
# Módulo aislado para manejo de múltiples herbicidas postemergentes
# Incluye: definición, UI en Streamlit y simulación básica

import streamlit as st
import pandas as pd
from datetime import date

# ===============================
# Clase herbicida postemergente
# ===============================
class HerbicidaPost:
    def __init__(self, nombre, fecha, dias_res, eficiencia, estados):
        self.nombre = nombre
        self.fecha = fecha
        self.dias_res = dias_res
        self.eficiencia = eficiencia
        self.estados = estados  # lista ["S1", "S2", ...]

    def to_dict(self):
        return {
            "Nombre": self.nombre,
            "Fecha aplicación": self.fecha,
            "Residualidad (días)": self.dias_res,
            "Eficiencia (%)": self.eficiencia,
            "Estados sensibles": ", ".join(self.estados),
        }

# ===============================
# Inicializar lista en session_state
# ===============================
if "herbicidas_post" not in st.session_state:
    st.session_state["herbicidas_post"] = []

# ===============================
# Sidebar para agregar herbicidas
# ===============================
st.sidebar.header("➕ Agregar herbicida postemergente")

with st.sidebar.form("form_herbicida"):
    nombre = st.text_input("Nombre del herbicida", "Herbicida X")
    fecha = st.date_input("Fecha de aplicación", value=date.today())
    dias_res = st.slider("Días de residualidad", 0, 60, 15)
    eficiencia = st.slider("Eficiencia (%)", 0, 100, 80)
    estados = st.multiselect("Estados sensibles", ["S1", "S2", "S3", "S4"], ["S1", "S2"])
    submitted = st.form_submit_button("Agregar")

if submitted:
    nuevo = HerbicidaPost(nombre, fecha, dias_res, eficiencia, estados)
    st.session_state["herbicidas_post"].append(nuevo)
    st.success(f"✔ {nombre} agregado a la lista")

# ===============================
# Mostrar herbicidas cargados
# ===============================
if st.session_state["herbicidas_post"]:
    st.subheader("📋 Lista de herbicidas cargados")
    df = pd.DataFrame([h.to_dict() for h in st.session_state["herbicidas_post"]])
    st.dataframe(df)

# ===============================
# Simulación muy simple (demo)
# ===============================
st.subheader("🔬 Simulación (demo)")

for h in st.session_state["herbicidas_post"]:
    st.write(f"→ {h.nombre}: aplicado el {h.fecha}, "
             f"eficacia {h.eficiencia}% sobre {h.estados}, "
             f"residualidad {h.dias_res} días")

