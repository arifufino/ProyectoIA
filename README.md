# EduFeel IA

Sistema inteligente que analiza opiniones estudiantiles en español y las clasifica como **positivas**, **neutrales** o **negativas**. Incluye entrenamiento configurable, métricas reales, API REST, interfaz web e historial persistente.

## Objetivos

- Facilitar el análisis inicial de la percepción estudiantil.
- Comparar Regresión Logística y Naive Bayes sobre representaciones TF-IDF.
- Exponer un flujo completo y reproducible, desde los datos hasta una aplicación web.

## Tecnologías

Python, Pandas, Scikit-learn, TF-IDF, FastAPI, Streamlit, SQLite, Joblib y Pytest.

## Arquitectura

```text
Streamlit ──HTTP──> FastAPI ──> Pipeline TF-IDF + clasificador
                        ├──────> artefacto Joblib
                        └──────> historial SQLite
Entrenamiento <──── CSV validado y limpio
```

El frontend nunca accede directamente al modelo o la base: toda operación pasa por FastAPI. El artefacto Joblib guarda pipeline, configuración, fecha y métricas.

## Instalación

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecución

En una terminal:

```powershell
uvicorn backend.main:app --reload
```

En otra terminal:

```powershell
streamlit run frontend/app.py
```

API: `http://localhost:8000`; documentación interactiva: `http://localhost:8000/docs`; interfaz: `http://localhost:8501`.

## Entrenamiento

Puede entrenarse desde la sección **Entrenar modelo** o con la API. Se configuran algoritmo, `test_size`, `max_features`, rango de n-gramas, `max_iter` y `alpha`. El CSV predeterminado contiene 150 comentarios balanceados. Para regenerarlo:

```powershell
python scripts/generate_dataset.py
```

## Uso de la API

| Método | Ruta | Propósito |
|---|---|---|
| GET | `/api/health` | Estado del servicio y modelo |
| POST | `/api/train` | Entrenar y obtener métricas |
| POST | `/api/predict` | Clasificar y guardar un comentario |
| GET | `/api/metrics` | Consultar evaluación actual |
| GET | `/api/history` | Historial SQLite |
| GET | `/api/model-info` | Configuración y fecha del modelo |

Ejemplos PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/train `
  -ContentType 'application/json' -Body '{"algorithm":"logistic_regression","test_size":0.2}'

Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/predict `
  -ContentType 'application/json' -Body '{"comentario":"La explicación fue muy clara"}'
```

La predicción devuelve `clasificacion`, `confianza`, `probabilidades`, `fecha` e `id` del historial.

## Pruebas

```powershell
pytest
```

Las pruebas cubren limpieza, esquema/categorías, suficiencia de datos, ambos modelos, predicción, errores, todos los flujos principales de API y SQLite.

## Estructura del proyecto

```text
backend/        API y contratos
frontend/       aplicación Streamlit
training/       limpieza, validación y modelo
data/           dataset educativo CSV
saved_models/   artefactos generados (ignorados por Git)
database/       historial SQLite (ignorado por Git)
scripts/        generación reproducible del dataset
tests/          suite Pytest
docs/           manuales, informe y guiones
```

## Limitaciones

El dataset es pequeño, sintético y de demostración. No representa todos los dialectos, contextos, ironías o comentarios mixtos. La confianza es una probabilidad del modelo, no una garantía, y el sistema no debe automatizar decisiones sensibles sobre estudiantes o docentes.

## Trabajos futuros

- Incorporar datos reales anonimizados y revisados éticamente.
- Validación cruzada, calibración y monitoreo de deriva.
- Autenticación, roles, exportación y paneles por periodo.
- Explicabilidad y revisión humana de casos inciertos.


Consulte [el manual de usuario](docs/manual_usuario.md) y [el manual técnico](docs/manual_tecnico.md) para más detalles.

