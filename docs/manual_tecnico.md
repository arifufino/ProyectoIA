# Manual técnico

## Componentes

`training/data.py` limpia y valida los CSV: exige `comentario` y `clasificacion`, elimina vacíos y duplicados, normaliza a minúsculas y conserva caracteres españoles y la negación “no”. Cada una de las tres clases debe tener al menos diez casos tras la limpieza.

`training/model.py` separa los datos de forma estratificada y crea un `Pipeline` con `TfidfVectorizer` y Regresión Logística o `MultinomialNB`. Guarda con Joblib el pipeline completo, configuración, clases, fecha, volumen del dataset y métricas del conjunto de prueba.

`backend/main.py` presenta seis endpoints REST. Las rutas del modelo, base y dataset pueden sobrescribirse con `EDUFEEL_MODEL_PATH`, `EDUFEEL_DB_PATH` y `EDUFEEL_DATASET_PATH`. `frontend/app.py` consume dichos endpoints con HTTP. `database/repository.py` usa consultas parametrizadas y JSON para probabilidades.

## Flujo de entrenamiento

1. Leer el CSV como UTF-8.
2. Validar columnas, categorías, vacíos, duplicados y tamaño.
3. Dividir de forma estratificada con semilla 42.
4. Ajustar TF-IDF y clasificador solo con entrenamiento.
5. Evaluar en prueba: accuracy, precisión, recall, F1, reporte por clase y matriz.
6. Persistir un único artefacto Joblib.

## Operación y seguridad

Los errores de entrada retornan HTTP 422; un modelo ausente retorna 503. SQLite se inicializa al arrancar y también de forma defensiva en el repositorio. CORS está abierto para facilitar la demostración local; en producción debe restringirse. Joblib solo debe cargar artefactos confiables.

## Desarrollo y pruebas

Instale `requirements.txt`, ejecute `pytest` y luego inicie Uvicorn y Streamlit. La documentación OpenAPI está en `/docs`. Los tests usan rutas temporales mediante variables de entorno y no contaminan datos reales.

