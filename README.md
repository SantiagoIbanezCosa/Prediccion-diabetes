escripción del Proyecto

Este proyecto realiza un análisis y modelado predictivo utilizando datos del archivo diabetes.csv.
El objetivo es predecir la variable Outcome mediante diversas técnicas de preprocesamiento, exploración y modelado de datos.

Características

Preprocesamiento de Datos

Imputación avanzada de valores faltantes con IterativeImputer.

Normalización y escalado con PowerTransformer.

Visualización de correlaciones con un mapa de calor.

Modelado

Se implementaron los siguientes algoritmos:

Regresión Lineal.

Árbol de Decisión con ajuste de hiperparámetros.

XGBoost con búsqueda aleatoria de hiperparámetros.

Random Forest.

Ensamblado de modelos (Stacking).

Evaluación

Las métricas de rendimiento utilizadas incluyen:

MAE (Error Absoluto Medio).

MSE (Error Cuadrático Medio).

RMSE (Raíz del Error Cuadrático Medio).

R² (Coeficiente de Determinación).

Requisitos

Librerías Necesarias

Instala las siguientes librerías antes de ejecutar el script:

pip install pandas seaborn matplotlib scikit-learn xgboost numpy

Archivos Necesarios

diabetes.csv: archivo de datos con características y valores objetivo.

Instrucciones de Uso

Clonar el Repositorio:

git clone https://github.com/tu_usuario/tu_repositorio.git

Colocar el Archivo de Datos:

Asegúrate de que diabetes.csv esté en el mismo directorio que el script.

Ejecutar el Script:

python script_name.py

Analizar Resultados:

Revisa las métricas de salida y las gráficas generadas.

Visualizaciones

El script incluye la generación de un mapa de calor para analizar correlaciones entre las variables.

Resultados

Este proyecto compara varios modelos predictivos y selecciona el de mejor rendimiento según métricas clave.
Los resultados también incluyen las mejores configuraciones de hiperparámetros.

Notas

Si el nombre de la columna objetivo no es Outcome, ajusta el script para reflejar el nombre correcto.

Asegúrate de que los datos sean consistentes tras el preprocesamiento.


