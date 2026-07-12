# Manual de usuario

## Puesta en marcha

Tras instalar dependencias, abra dos PowerShell: en el primero ejecute `uvicorn backend.main:app --reload`; en el segundo, `streamlit run frontend/app.py`. Abra la dirección mostrada por Streamlit.

## Secciones

- **Inicio:** confirma si la API y el modelo están disponibles.
- **Analizar comentario:** escriba una opinión, pulse **Analizar** y consulte clase, confianza y probabilidades.
- **Entrenar modelo:** seleccione algoritmo y parámetros. El entrenamiento reemplaza el modelo local actual.
- **Métricas:** muestra resultados reales del conjunto reservado para prueba.
- **Historial:** presenta predicciones persistidas, primero las más recientes.
- **Información del proyecto:** resume alcance y configuración actual.

## Solución de problemas

- “API no disponible”: inicie Uvicorn y confirme el puerto 8000.
- “Modelo aún no entrenado”: use **Entrenar modelo** antes de analizar.
- “Comentario vacío”: escriba contenido visible.
- Error de CSV: use UTF-8, los encabezados exactos y solo `positivo`, `neutral` o `negativo`.

No use el resultado como único criterio para evaluar personas. Revise manualmente opiniones ambiguas o de baja confianza.

