# Entrenamiento con datos acumulativos

La pestaña **Entrenar modelo** permite cargar archivos CSV nuevos con las columnas `comentario` y `clasificacion`. Las categorías admitidas son `positivo`, `neutral` y `negativo`.

Al pulsar **Agregar datos y entrenar**, la API:

1. Valida y limpia el archivo recibido.
2. Combina los ejemplos con el dataset acumulado existente o, en el primer uso, con los 150 ejemplos base.
3. Elimina comentarios duplicados.
4. Guarda el conjunto ampliado en `data/dataset_acumulado.csv`.
5. Entrena un modelo nuevo con todos los datos disponibles y reemplaza el artefacto Joblib anterior.
6. Muestra cuántos ejemplos se recibieron, agregaron u omitieron y la nueva distribución por clase.

El archivo acumulado se excluye de Git porque contiene datos aportados durante el uso. Puede cambiarse su ubicación con `EDUFEEL_ACCUMULATED_DATASET_PATH`.

También se puede consumir directamente el endpoint multipart `POST /api/train/upload`. Los parámetros de configuración se envían como campos de formulario y el CSV en el campo `file`.

Este flujo es aprendizaje incremental a nivel de datos y reentrenamiento: no modifica el clasificador después de cada predicción sin etiqueta, pues una predicción no aporta una respuesta correcta con la cual aprender.
