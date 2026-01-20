import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ---------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------
st.set_page_config(
    page_title="EDA Telco Customer Churn",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# CLASE POO
# ---------------------------
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def clasificar_variables(self):
        numericas = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categoricas = self.df.select_dtypes(include=["object"]).columns.tolist()
        return numericas, categoricas

    def valores_nulos(self):
        return self.df.isnull().sum()

    def estadisticas_descriptivas(self):
        return self.df.describe()

    def media(self, col):
        return self.df[col].mean()

    def mediana(self, col):
        return self.df[col].median()

    def moda(self, col):
        return self.df[col].mode()[0]

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("📌 Menú Principal")
menu = st.sidebar.radio(
    "Seleccione una opción",
    ["🏠 Home", "📂 Carga de Datos", "🔍 EDA", "📌 Conclusiones"]
)

# ---------------------------
# HOME
# ---------------------------
if menu == "🏠 Home":
    st.title("📊 Análisis Exploratorio de Datos – Telco Customer Churn")

    st.markdown("""
    ### 🎯 Objetivo del proyecto
    Desarrollar una aplicación interactiva en **Streamlit** para realizar un **Análisis Exploratorio
    de Datos (EDA)** del dataset **TelcoCustomerChurn**, con el fin de identificar patrones asociados
    a la fuga de clientes (**Churn**).

    ### 👤 Autor
    - **Nombre:** Frank Bellido  
    - **Curso:** Especialización en Python for Analytics  
    - **Docente:** Carlos Carrillo Villavicencio  
    - **Año:** 2026  

    ### 📊 Dataset
    Información demográfica, servicios contratados, facturación, antigüedad y estado de churn
    de clientes de una empresa de telecomunicaciones.

    ### 🛠 Tecnologías utilizadas
    - 🐍 Python  
    - 📊 Pandas  
    - 🔢 NumPy  
    - 📈 Matplotlib  
    - 🎨 Seaborn  
    - 🚀 Streamlit  
    """)

# ---------------------------
# CARGA DE DATOS
# ---------------------------
elif menu == "📂 Carga de Datos":
    st.title("📂 Carga del Dataset")

    archivo = st.file_uploader(
        "📁 Seleccione el archivo TelcoCustomerChurn.csv",
        type="csv"
    )

    if archivo is not None:
        df = pd.read_csv(archivo)
        st.session_state["df"] = df

        st.success("✅ Archivo cargado correctamente")

        st.subheader("👀 Vista previa del dataset")
        st.dataframe(df.head())

        st.subheader("📐 Dimensiones del dataset")
        st.write(f"🔹 Filas: {df.shape[0]}")
        st.write(f"🔹 Columnas: {df.shape[1]}")
    else:
        st.warning("⚠️ Debe cargar el dataset para continuar")

# ---------------------------
# EDA
# ---------------------------
elif menu == "🔍 EDA":
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")

    if "df" not in st.session_state:
        st.warning("⚠️ Primero debe cargar el dataset")
    else:
        df = st.session_state["df"]
        analyzer = DataAnalyzer(df)
        numericas, categoricas = analyzer.clasificar_variables()

        tabs = st.tabs([
            "ℹ️ Info general",
            "📋 Clasificación",
            "📈 Estadísticas",
            "❓ Valores faltantes",
            "📊 Distribución numérica",
            "🏷️ Variables categóricas",
            "📉 Num vs Churn",
            "📊 Cat vs Churn",
            "⚙️ Análisis dinámico",
            "🧠 Hallazgos"
        ])

        # ÍTEM 1
        with tabs[0]:
            st.subheader("ℹ️ Información general del dataset")
            buffer = StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

        # ÍTEM 2
        with tabs[1]:
            st.subheader("📋 Clasificación de variables")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔢 Variables numéricas")
                st.write(numericas)

            with col2:
                st.markdown("### 🏷️ Variables categóricas")
                st.write(categoricas)

        # ÍTEM 3
        with tabs[2]:
            st.subheader("📈 Estadísticas descriptivas")
            st.dataframe(analyzer.estadisticas_descriptivas())

            var = st.selectbox("🔢 Seleccione una variable numérica", numericas)
            st.write(f"📌 **Media:** {analyzer.media(var):.2f}")
            st.write(f"📌 **Mediana:** {analyzer.mediana(var):.2f}")
            st.write(f"📌 **Moda:** {analyzer.moda(var)}")

        # ÍTEM 4
        with tabs[3]:
            st.subheader("❓ Análisis de valores faltantes")
            nulos = analyzer.valores_nulos()
            st.dataframe(nulos)

            if nulos.sum() > 0:
                st.bar_chart(nulos[nulos > 0])
            else:
                st.success("✅ No existen valores faltantes")

        # ÍTEM 5
        with tabs[4]:
            st.subheader("📊 Distribución de variables numéricas")
            var = st.selectbox("🔢 Variable", numericas)
            bins = st.slider("📦 Número de bins", 5, 50, 30)

            fig, ax = plt.subplots()
            sns.histplot(df[var], bins=bins, ax=ax)
            st.pyplot(fig)

        # ÍTEM 6
        with tabs[5]:
            st.subheader("🏷️ Análisis de variables categóricas")
            var = st.selectbox("🏷️ Variable categórica", categoricas)

            fig, ax = plt.subplots()
            sns.countplot(data=df, x=var, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        # ÍTEM 7
        with tabs[6]:
            st.subheader("📉 Variable numérica vs Churn")
            var = st.selectbox("🔢 Variable numérica", numericas)

            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="Churn", y=var, ax=ax)
            st.pyplot(fig)

        # ÍTEM 8
        with tabs[7]:
            st.subheader("📊 Variable categórica vs Churn")
            var = st.selectbox("🏷️ Variable categórica", categoricas)

            fig, ax = plt.subplots()
            sns.countplot(data=df, x=var, hue="Churn", ax=ax)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        # ÍTEM 9
        with tabs[8]:
            st.subheader("⚙️ Análisis dinámico")
            columnas = st.multiselect(
                "🔢 Seleccione variables numéricas",
                numericas
            )

            mostrar = st.checkbox("📋 Mostrar estadísticas")

            if columnas and mostrar:
                st.dataframe(df[columnas].describe())

        # ÍTEM 10
        with tabs[9]:
            st.subheader("🧠 Hallazgos clave")
            churn_prop = df["Churn"].value_counts(normalize=True)
            st.bar_chart(churn_prop)

            st.markdown("""
            **🔎 Insights principales:**
            - 📉 El churn es mayor en clientes con baja antigüedad.
            - 💰 Cargos mensuales altos se asocian a mayor abandono.
            - 📄 Contratos mensuales presentan mayor churn.
            - ➕ Servicios adicionales reducen la fuga.
            - 📊 El EDA apoya decisiones estratégicas de retención.
            """)

# ---------------------------
# CONCLUSIONES
# ---------------------------
elif menu == "📌 Conclusiones":
    st.title("📌 Conclusiones finales")

    st.markdown("""
    ✅ Los primeros meses del cliente son críticos para la retención.  
    ✅ Cargos elevados influyen negativamente en la permanencia.  
    ✅ El tipo de contrato es una variable clave en el churn.  
    ✅ Servicios adicionales reducen la probabilidad de abandono.  
    ✅ El análisis exploratorio es fundamental para la toma de decisiones.
    """)
