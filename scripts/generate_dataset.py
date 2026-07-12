"""Genera el dataset demostrativo balanceado de EduFeel IA."""

from pathlib import Path
import csv

subjects = ["matemáticas", "lengua", "ciencias", "historia", "inglés", "programación", "física", "química", "arte", "estadística"]
templates = {
    "positivo": [
        "La clase de {s} fue excelente y aprendí mucho.",
        "La profesora explicó {s} con claridad y buenos ejemplos.",
        "Me gustaron las actividades de {s}; fueron útiles y dinámicas.",
        "El material de {s} estuvo muy bien organizado y fue fácil de comprender.",
        "Ahora entiendo mejor {s} gracias al apoyo del docente.",
    ],
    "neutral": [
        "La clase de {s} se realizó según el horario establecido.",
        "Hoy revisamos el contenido de {s} y completamos las actividades.",
        "El docente presentó el tema de {s} y dejó una tarea.",
        "La sesión de {s} tuvo explicación y ejercicios en clase.",
        "Se utilizó el material de {s} disponible en la plataforma.",
    ],
    "negativo": [
        "La clase de {s} fue confusa y no logré aprender el tema.",
        "La explicación de {s} fue poco clara y demasiado rápida.",
        "No entendí los ejercicios de {s} y faltó apoyo del docente.",
        "El material de {s} estaba desordenado y fue difícil de usar.",
        "La actividad de {s} fue frustrante y las instrucciones eran incorrectas.",
    ],
}

output = Path(__file__).resolve().parents[1] / "data" / "comentarios.csv"
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["comentario", "clasificacion"])
    for label, variants in templates.items():
        for subject in subjects:
            for template in variants:
                writer.writerow([template.format(s=subject), label])
print(f"Dataset generado: {output} (150 registros)")
