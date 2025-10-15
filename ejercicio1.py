import pandas as pd
import math

# Datos de ejemplo
datos = pd.DataFrame({
    "ID": range(1, 11),
    "Edad": [24, 38, 29, 45, 52, 33, 41, 27, 36, 31],
    "Uso_datos": [2.5, 6.0, 3.0, 8.0, 7.5, 4.0, 5.5, 2.0, 6.5, 3.5],
    "Linea_fija": ["No", "Sí", "No", "Sí", "Sí", "No", "Sí", "No", "Sí", "No"],
    "Acepto_oferta": ["No", "Sí", "No", "Sí", "Sí", "No", "Sí", "No", "Sí", "No"]
})

# Agrupar atributos en rangos
def grupo_edad(edad):
    if edad <= 30:
        return "Joven"
    elif 31 <= edad <= 50:
        return "Adulto"
    else:
        return "Mayor"

def grupo_uso(uso):
    if uso <= 3:
        return "Bajo"
    elif 3.1 <= uso <= 6:
        return "Medio"
    else:
        return "Alto"

datos["Edad_grupo"] = datos["Edad"].apply(grupo_edad)
datos["Uso_grupo"] = datos["Uso_datos"].apply(grupo_uso)

# Función para calcular entropía
def entropia(clase):
    total = len(clase)
    valores = clase.value_counts()
    ent = -sum((v/total) * math.log2(v/total) for v in valores)
    return ent

# Entropía del conjunto original
H_total = entropia(datos["Acepto_oferta"])
print(f"Entropía total del conjunto: {H_total:.3f}")

# Función para calcular ganancia de información
def ganancia_info(df, atributo, objetivo="Acepto_oferta"):
    total = len(df)
    H_total = entropia(df[objetivo])
    valores = df[atributo].unique()
    H_cond = 0
    for val in valores:
        sub = df[df[atributo] == val]
        H_sub = entropia(sub[objetivo])
        H_cond += (len(sub)/total) * H_sub
    return H_total - H_cond

# Ganancia de información para cada atributo
atributos = ["Edad_grupo", "Linea_fija", "Uso_grupo"]
for attr in atributos:
    gi = ganancia_info(datos, attr)
    print(f"Ganancia de información ({attr}): {gi:.3f}")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# Simulamos un dataset de ejemplo (simplificado)
import numpy as np
np.random.seed(42)

datos_est = pd.DataFrame({
    "Edad": np.random.randint(18, 30, 20),
    "Genero": np.random.choice(["M", "F"], 20),
    "Carrera": np.random.choice(["IA", "Medicina", "Administracion"], 20),
    "Promedio": np.random.uniform(4, 10, 20),
    "Materias_aprobadas": np.random.randint(0, 6, 20),
    "Materias_desaprobadas": np.random.randint(0, 3, 20),
    "Asistencia": np.random.randint(50, 100, 20),
    "Situacion_laboral": np.random.choice(["Trabaja", "No trabaja"], 20),
    "Distancia": np.random.randint(1, 50, 20),
    "Tutorias": np.random.choice(["Sí", "No"], 20),
    "Estado_final": np.random.choice(["Abandono", "Continua"], 20)
})

# Codificar variables categóricas
datos_est_enc = pd.get_dummies(datos_est.drop("Estado_final", axis=1))
y = datos_est["Estado_final"]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datos_est_enc, y, test_size=0.3, random_state=42)

# Entrenar árbol de decisión
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Evaluación
score = clf.score(X_test, y_test)
print(f"\nPrecisión del árbol de decisión: {score:.2f}")

# Mostrar árbol generado
arbol_texto = export_text(clf, feature_names=list(X_train.columns))
print("\nÁrbol de decisión generado:\n")
print(arbol_texto)
#Interpretación
print("Interpretación:")
print("""Interpretación:
Las variables que aparecen en los primeros niveles del árbol son las más importantes para predecir el abandono.
Se pueden tomar decisiones para la institución, como reforzar tutorías, monitorear asistencia o acompañamiento personalizado según los factores más relevantes.""")
