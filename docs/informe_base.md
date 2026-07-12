# Informe base — EduFeel IA

## Problema y propuesta

Revisar manualmente grandes cantidades de opiniones estudiantiles consume tiempo y dificulta detectar tendencias. EduFeel IA propone una clasificación inicial en tres polaridades para apoyar —no reemplazar— la revisión humana.

## Metodología

Se construyó un conjunto balanceado de 150 frases educativas en español. La preparación elimina registros vacíos y duplicados, normaliza el texto y controla esquema, etiquetas y suficiencia. Una partición estratificada reproducible reserva 20 % para evaluación predeterminada.

TF-IDF convierte palabras y bigramas en características. Se implementaron Regresión Logística, adecuada como línea base lineal interpretable, y Naive Bayes multinomial, rápido y habitual en texto. Las métricas se calculan solo sobre el conjunto de prueba e incluyen valores ponderados y resultados por clase.

## Producto

FastAPI centraliza entrenamiento, inferencia, métricas e historial. Streamlit ofrece seis vistas en español. Cada inferencia registra texto, clase, confianza, distribución probabilística y fecha UTC en SQLite. Joblib conserva el pipeline y metadatos.

## Ética, límites y conclusión

Al ser sintéticos y limitados, los datos pueden producir sesgos y generalización deficiente. Deben anonimizarse datos reales, obtener consentimiento según corresponda y evitar decisiones automatizadas que afecten a personas. El proyecto demuestra una arquitectura funcional y extensible, pero requiere validación con datos representativos antes de un uso institucional.

