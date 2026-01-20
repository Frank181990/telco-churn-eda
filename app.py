import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

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

def clasificar_variables(df):
    numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categoricas = df.select_dtypes(include=["object"]).columns.tolist()
    return numericas, categoricas


st.title("Proyecto EDA - Telco Customer Churn")

menu = st.sidebar.radio(
    "Menú",
    ["Home", "Carga de Datos", "EDA", "Conclusiones"]
)

if menu == "Home":
    st.title("Análisis Exploratorio de Datos - Telco Customer Churn")

    st.markdown("""
    ### 📌 Objetivo del proyecto
    Desarrollar una aplicación interactiva en Streamlit para realizar un Análisis Exploratorio
    de Datos (EDA) sobre el comportamiento de clientes de una empresa de telecomunicaciones,
    identificando patrones asociados a la fuga de clientes (Churn).

    ### 👤 Autor
    **Nombre:** Frank Bellido
    **Curso:** Especialización en Python for Analytics  
    **Docente:** Carlos Carrillo Villavicencio  
    **Año:** 2026

    ### 📊 Dataset
    El dataset **TelcoCustomerChurn.csv** contiene información sobre clientes, servicios
    contratados, facturación, tiempo de permanencia y estado de churn.

    ### 🛠 Tecnologías utilizadas
    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - Streamlit
    """)


elif menu == "Carga de Datos":
    st.subheader("Carga del Dataset")

    archivo = st.file_uploader(
        "Seleccione el archivo TelcoCustomerChurn.csv",
        type="csv"
    )

    if archivo is not None:
        df = pd.read_csv(archivo)
        st.session_state["df"] = df

        st.success("Archivo cargado correctamente")

        st.subheader("Vista previa del dataset")
        st.dataframe(df.head())

        st.subheader("Dimensiones del dataset")
        st.write(f"Filas: {df.shape[0]}")
        st.write(f"Columnas: {df.shape[1]}")
    else:
        st.warning("Por favor, cargue un archivo CSV para continuar")


elif menu == "EDA":
    st.subheader("Análisis Exploratorio de Datos")

    if "df" not in st.session_state:
        st.warning("Primero debe cargar el dataset")
    else:
        df = st.session_state["df"]
        analyzer = DataAnalyzer(df)
        numericas, categoricas = analyzer.clasificar_variables()

        tabs = st.tabs([
            "📄 Info general",
            "📊 Estadísticas",
            "🔍 Valores faltantes",
            "📈 Univariado",
            "🔄 Bivariado",
            "🧠 Hallazgos"
        ])

        # TAB 1 — INFO GENERAL
        with tabs[0]:
            st.subheader("Información general del dataset")

            st.write("Dimensiones:")
            st.write(df.shape)

            st.dataframe(df.head())

            buffer = StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

            col1, col2 = st.columns(2)
            with col1:
                st.write("Variables numéricas")
                st.write(numericas)
            with col2:
                st.write("Variables categóricas")
                st.write(categoricas)

        # TAB 2 — ESTADÍSTICAS
        with tabs[1]:
            st.subheader("Estadísticas descriptivas")
            st.dataframe(analyzer.estadisticas_descriptivas())

        # TAB 3 — VALORES FALTANTES
        with tabs[2]:
            st.subheader("Valores faltantes")
            nulos = analyzer.valores_nulos()
            st.dataframe(nulos)

            nulos_filtrados = nulos[nulos > 0]
            if len(nulos_filtrados) > 0:
                st.bar_chart(nulos_filtrados)
            else:
                st.success("No se encontraron valores faltantes")

        # TAB 4 — UNIVARIADO
        with tabs[3]:
            st.subheader("Análisis univariado")

            var_num = st.selectbox("Variable numérica", numericas)
            fig, ax = plt.subplots()
            sns.histplot(df[var_num], bins=30, ax=ax)
            st.pyplot(fig)

            var_cat = st.selectbox("Variable categórica", categoricas)
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=var_cat, ax=ax)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        # TAB 5 — BIVARIADO
        with tabs[4]:
            st.subheader("Análisis bivariado")

            var_num = st.selectbox("Numérica vs Churn", numericas)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="Churn", y=var_num, ax=ax)
            st.pyplot(fig)

            var_cat = st.selectbox("Categórica vs Churn", categoricas)
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=var_cat, hue="Churn", ax=ax)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        # TAB 6 — HALLAZGOS
        with tabs[5]:
            st.markdown("""
            ### Hallazgos clave

            1. El churn se concentra en clientes con baja antigüedad.
            2. Cargos mensuales altos se asocian a mayor abandono.
            3. El tipo de contrato influye en la retención.
            4. Servicios adicionales reducen el churn.
            5. El EDA permite identificar patrones críticos de negocio.
            """)


elif menu == "Conclusiones":
    st.title("Conclusiones finales del análisis")

    st.markdown("""
### Conclusiones principales

1. El churn se presenta principalmente en clientes con baja antigüedad, lo que indica que los primeros meses son críticos para la retención.
2. Los clientes con cargos mensuales más elevados tienden a abandonar el servicio con mayor frecuencia, lo que sugiere una posible percepción negativa del valor del servicio.
3. La mayoría de los clientes no pertenecen al grupo de adultos mayores, por lo que las estrategias de retención deben enfocarse en el segmento general de clientes.
4. El churn representa una proporción significativa del total de clientes, lo cual puede generar impactos económicos relevantes considerando el alto costo de adquisición de nuevos clientes.
5. El análisis exploratorio e interactivo permite identificar patrones clave que pueden apoyar la toma de decisiones estratégicas orientadas a la mejora de la retención de clientes.
""")






