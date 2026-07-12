"""Interfaz Streamlit en español conectada exclusivamente a FastAPI."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("EDUFEEL_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="EduFeel IA", page_icon="🎓", layout="wide")


def api_request(method: str, endpoint: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{endpoint}", timeout=120, **kwargs)
        if response.ok:
            return response.json()
        detail = response.json().get("detail", response.text)
        st.error(f"La API respondió con un error: {detail}")
    except requests.RequestException:
        st.error("La API no está disponible. Inicia FastAPI en http://localhost:8000.")
    return None


def show_metrics(metrics: dict) -> None:
    columns = st.columns(4)
    for column, key, label in zip(columns, ("accuracy", "precision", "recall", "f1_score"), ("Accuracy", "Precisión", "Recall", "F1-score")):
        column.metric(label, f"{metrics[key]:.2%}")
    st.write(f"Entrenamiento: {metrics['training_samples']} · Prueba: {metrics['test_samples']}")
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
    st.subheader("Matriz de confusión")
    st.dataframe(matrix, use_container_width=True)
    report = pd.DataFrame(metrics["classification_report"]).transpose()
    st.subheader("Reporte por clase")
    st.dataframe(report, use_container_width=True)


st.title("🎓 EduFeel IA")
st.caption("Sistema inteligente para analizar opiniones estudiantiles")
page = st.sidebar.radio("Navegación", ["Inicio", "Analizar comentario", "Entrenar modelo", "Métricas", "Historial", "Información del proyecto"])

if page == "Inicio":
    st.header("Inicio")
    st.write("EduFeel IA clasifica comentarios educativos como positivos, neutrales o negativos mediante TF-IDF y aprendizaje automático.")
    health = api_request("GET", "/api/health")
    if health:
        st.success(f"API conectada · Modelo {'disponible' if health['model_trained'] else 'sin entrenar'}")

elif page == "Analizar comentario":
    st.header("Analizar comentario")
    comment = st.text_area("Comentario estudiantil", placeholder="Escribe aquí una opinión sobre la clase…", height=140)
    if st.button("Analizar", type="primary"):
        if not comment.strip():
            st.warning("El comentario no puede estar vacío.")
        else:
            result = api_request("POST", "/api/predict", json={"comentario": comment})
            if result:
                st.success(f"Clasificación: {result['clasificacion'].upper()}")
                st.metric("Confianza", f"{result['confianza']:.2%}")
                st.bar_chart(pd.Series(result["probabilidades"], name="Probabilidad"))
                st.caption(f"Fecha: {result['fecha']}")

elif page == "Entrenar modelo":
    st.header("Entrenar modelo")
    st.write("Carga comentarios etiquetados nuevos. Se acumularán con los anteriores y el modelo se reentrenará con el conjunto completo.")
    uploaded_file = st.file_uploader(
        "Dataset nuevo (CSV)", type=["csv"],
        help="Columnas: comentario, clasificacion. Categorías: positivo, neutral o negativo.",
    )
    algorithm_label = st.selectbox("Algoritmo", ["Regresión logística", "Naive Bayes"])
    col1, col2 = st.columns(2)
    test_size = col1.slider("Proporción de prueba", 0.1, 0.5, 0.2, 0.05)
    max_features = col2.number_input("Máximo de características", 100, 20000, 5000, 100)
    ngram_min = col1.selectbox("N-grama mínimo", [1, 2], index=0)
    ngram_max = col2.selectbox("N-grama máximo", [1, 2, 3], index=1)
    max_iter = col1.number_input("Iteraciones máximas", 100, 5000, 1000, 100)
    alpha = col2.number_input("Alpha (Naive Bayes)", 0.01, 10.0, 1.0, 0.1)
    button_label = "Agregar datos y entrenar" if uploaded_file else "Entrenar con datos acumulados"
    if st.button(button_label, type="primary"):
        if ngram_min > ngram_max:
            st.warning("El n-grama mínimo no puede ser mayor al máximo.")
        else:
            payload = {
                "algorithm": "logistic_regression" if algorithm_label.startswith("Regresión") else "naive_bayes",
                "test_size": test_size, "max_features": max_features, "ngram_min": ngram_min,
                "ngram_max": ngram_max, "max_iter": max_iter, "alpha": alpha,
            }
            with st.spinner("Entrenando…"):
                if uploaded_file:
                    response = api_request(
                        "POST", "/api/train/upload",
                        data={key: str(value) for key, value in payload.items()},
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    )
                else:
                    response = api_request("POST", "/api/train", json=payload)
            if response:
                st.success(response["message"])
                if "dataset" in response:
                    dataset = response["dataset"]
                    st.info(
                        f"Dataset acumulado: {dataset['total_samples']} comentarios · "
                        f"Nuevos: {dataset['added_samples']} · Duplicados omitidos: {dataset['duplicates_ignored']}"
                    )
                    st.write("Distribución por clase:", dataset["class_distribution"])
                show_metrics(response["metrics"])

elif page == "Métricas":
    st.header("Métricas del modelo")
    data = api_request("GET", "/api/metrics")
    if data:
        show_metrics(data)

elif page == "Historial":
    st.header("Historial de predicciones")
    limit = st.slider("Cantidad", 10, 200, 50, 10)
    data = api_request("GET", f"/api/history?limit={limit}")
    if data is not None:
        if data:
            frame = pd.DataFrame(data)
            st.dataframe(frame.drop(columns=["probabilidades"]), use_container_width=True, hide_index=True)
        else:
            st.info("Aún no existen predicciones.")

else:
    st.header("Información del proyecto")
    st.markdown("""
    **EduFeel IA** demuestra un flujo completo de clasificación de texto: limpieza, TF-IDF,
    entrenamiento supervisado, evaluación, API REST, interfaz web y persistencia SQLite.

    El modelo es educativo y sus resultados dependen de la representatividad del dataset.
    """)
    info = api_request("GET", "/api/model-info")
    if info:
        st.subheader("Modelo actual")
        st.json(info)
