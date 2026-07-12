# Guion de exposición

## 1. Apertura (1 minuto)

Presentar el problema: muchas opiniones educativas son difíciles de revisar oportunamente. Introducir EduFeel IA como apoyo para agruparlas en positivo, neutral y negativo.

## 2. Datos y modelo (2 minutos)

Mostrar el CSV balanceado y explicar limpieza, validaciones y conservación del español. Describir TF-IDF y la elección entre Regresión Logística y Naive Bayes. Aclarar que la prueba está separada del entrenamiento.

## 3. Demostración (3 minutos)

Entrenar desde Streamlit, abrir métricas y explicar la matriz de confusión. Analizar un ejemplo de cada clase. Mostrar confianza, probabilidades e historial persistente. Abrir `/docs` para enseñar la API.

## 4. Arquitectura y calidad (1 minuto)

Explicar la separación Streamlit–FastAPI–modelo–SQLite y mencionar la suite automatizada de limpieza, modelos, API y base de datos.

## 5. Límites y cierre (1 minuto)

Recalcar que el dataset es demostrativo, que la revisión humana es necesaria y proponer datos reales anonimizados, calibración y monitoreo como siguiente fase.

