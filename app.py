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

        st.write("Dimensiones del dataset:")
        st.write(df.shape)

        st.subheader("Primeras filas")
        st.dataframe(df.head())

        st.subheader("Información general del dataset")

        buffer = StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue()

        st.text(info_text)

        st.subheader("Valores nulos por columna")
        st.dataframe(df.isnull().sum())

        st.subheader("Clasificación de variables")

        analyzer = DataAnalyzer(df)
        numericas, categoricas = analyzer.clasificar_variables()

        col1, col2 = st.columns(2)

        with col1:
            st.write("Variables numéricas")
            st.write(f"Total: {len(numericas)}")
            st.write(numericas)

        with col2:
            st.write("Variables categóricas")
            st.write(f"Total: {len(categoricas)}")
            st.write(categoricas)
        
        st.subheader("Estadísticas descriptivas de variables numéricas")

        st.dataframe(analyzer.estadisticas_descriptivas())

        st.subheader("Análisis de valores faltantes")

        nulos = analyzer.valores_nulos()
        st.dataframe(nulos)


        st.subheader("Visualización de valores faltantes")

        nulos_filtrados = nulos[nulos > 0]

        if len(nulos_filtrados) == 0:
            st.write("No se encontraron valores faltantes en el dataset.")
        else:
            st.bar_chart(nulos_filtrados)

        st.subheader("Distribución de variables numéricas")

        st.write("Distribución de tenure")

        st.bar_chart(
            df["tenure"].value_counts().sort_index()
        )

        st.write("Distribución de MonthlyCharges")

        st.bar_chart(
            df["MonthlyCharges"].value_counts().sort_index()
        )

        st.write("Distribución de SeniorCitizen")

        st.bar_chart(
            df["SeniorCitizen"].value_counts()
        )

        st.subheader("Análisis de la variable Churn")

        churn_counts = df["Churn"].value_counts()

        st.write("Conteo de clientes según Churn:")
        st.write(churn_counts)

        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Churn", ax=ax)
        ax.set_title("Distribución de Churn")

        st.pyplot(fig)

        st.subheader("Análisis bivariado: Tenure vs Churn")

        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Churn", y="tenure", ax=ax)
        ax.set_title("Tenure según estado de Churn")

        st.pyplot(fig)

        st.write("Estadísticas descriptivas de tenure según Churn:")
        st.write(df.groupby("Churn")["tenure"].describe())

        st.subheader("Análisis bivariado: MonthlyCharges vs Churn")

        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
        ax.set_title("Cargos mensuales según estado de Churn")

        st.pyplot(fig)

        st.write("Estadísticas descriptivas de MonthlyCharges según Churn:")
        st.write(df.groupby("Churn")["MonthlyCharges"].describe())

        st.subheader("Análisis dinámico según variable seleccionada")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()


        selected_var = st.selectbox(
        "Selecciona una variable numérica para analizar vs Churn:",
        numeric_cols
        )

        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Churn", y=selected_var, ax=ax)
        ax.set_title(f"{selected_var} según estado de Churn")

        st.pyplot(fig)

        st.subheader("Análisis de variables categóricas")

        cat_var = st.selectbox(
        "Selecciona una variable categórica",
        categoricas
            )

        conteo = df[cat_var].value_counts()

        st.write(f"Conteo de {cat_var}:")
        st.dataframe(conteo)

        fig, ax = plt.subplots()
        sns.countplot(data=df, x=cat_var, ax=ax)
        ax.set_title(f"Distribución de {cat_var}")
        ax.tick_params(axis='x', rotation=45)

        st.pyplot(fig)

        st.subheader("Análisis bivariado: Variable categórica vs Churn")

        cat_var2 = st.selectbox(
        "Selecciona una variable categórica para comparar con Churn",
        categoricas
        )

        fig, ax = plt.subplots()
        sns.countplot(data=df, x=cat_var2, hue="Churn", ax=ax)
        ax.set_title(f"{cat_var2} vs Churn")
        ax.tick_params(axis='x', rotation=45)

        st.pyplot(fig)

        st.subheader("Hallazgos clave del análisis exploratorio")

        st.markdown("""
**Principales hallazgos:**

1. La mayoría de los clientes que presentan churn tienen una baja antigüedad, lo que indica que el abandono ocurre principalmente en los primeros meses.
2. Los clientes con cargos mensuales más elevados muestran una mayor tendencia a abandonar el servicio.
3. La base de clientes está compuesta mayoritariamente por usuarios que no pertenecen al grupo SeniorCitizen.
4. El churn representa una proporción relevante del total de clientes, lo que puede impactar negativamente en los ingresos de la empresa.
5. El análisis interactivo confirma que distintas variables numéricas presentan comportamientos diferenciados según el estado de churn.
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






