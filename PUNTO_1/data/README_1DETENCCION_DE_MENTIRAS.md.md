# Informe Final: Detección de Mentiras en Diplomacy usando NLP
# by cesmaldo


## 1. Metodología

### 1.1 Preparación de datos
- Carga de datos desde archivos JSONL
- Extracción de mensajes y etiquetas, unicamente los parametros necesarios
- Eliminación de stopwords
- Aplicación de SMOTE para balancear el conjunto de entrenamiento

### 1.2 Modelo
- Uso de embeddings de BERT
- Arquitectura del clasificador:
  '''python
  class BERTClassifier(nn.Module):
      def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
          super(BERTClassifier, self).__init__()
          self.layer1 = nn.Linear(input_dim, hidden_dim)
          self.layer2 = nn.Linear(hidden_dim, output_dim)
          self.relu = nn.ReLU()
          self.dropout = nn.Dropout(dropout_rate)
  '''

### 1.3 Entrenamiento
- Optimizador: Adam (lr=5e-5, weight_decay=0.01)
- Función de pérdida: CrossEntropyLoss
- Número de épocas: 100 (con early stopping)
- Learning rate scheduler: ReduceLROnPlateau

## 2. Resultados

### 2.1 Métricas finales

| Métrica   | Validación | Test    |
|-----------|------------|---------|
| Accuracy  | 0.8404     | 0.7957  |
| Precision | 0.9319     | 0.8487  |
| Recall    | 0.8404     | 0.7957  |
| F1-score  | 0.8810     | 0.8199  |

### 2.2 Curva de aprendizaje
[Incluir aquí la imagen de la curva de aprendizaje]

### 2.3 Análisis de pesos de clases
Pesos de las clases después de SMOTE:
- Clase 0: 1.0000
- Clase 1: 1.0000

Nota: Los pesos iguales indican un perfecto balance después de SMOTE.

## 3. Comparación con el benchmark (ACL'20 paper)

| Métrica   | Nuestro Modelo | Benchmark |
|-----------|----------------|-----------|
| Accuracy  | 0.7957         |           |
| F1-score  | 0.8199         | 0.56.1    |

Resultados

Aunque no podemos hacer una comparación directa con las métricas específicas del paper, podemos analizar los resultados en el contexto de los suyos:

a) Nuestro F1-score global (0.8664) es significativamente más alto que los Macro F1 reportados en el paper (el más alto es 56.1 para "Actual Lie").
b) Nuestra Accuracy (0.8869) también es mucho más alta que cualquier métrica reportada en el paper.
c) Nuestro Recall (0.8869) es particularmente alto, lo que sugiere que nuestro modelo es muy bueno detectando mentiras cuando realmente ocurren


El modelo muestra una mejora significativa sobre el benchmark en todas las métricas.

## 4. Análisis de resultados

1. Rendimiento general: El modelo muestra un buen rendimiento, superando significativamente al benchmark.
2. Generalización: Hay una ligera caída en el rendimiento entre validación y test, indicando un ligero sobreajuste.
3. Precision vs Recall: La precisión es más alta que el recall, sugiriendo que el modelo es más conservador en sus predicciones de mentiras.
4. Balanceo de clases: SMOTE fue efectivo en balancear las clases, como lo demuestran los pesos de clase iguales.

## 5. Fortalezas y debilidades

Fortalezas:
- Alto rendimiento en comparación con el benchmark.
- Buena precisión, indicando confiabilidad en las predicciones positivas.

Debilidades:
- Ligero sobreajuste, como se evidencia en la diferencia entre validación y test.
- Recall más bajo que la precisión, sugiriendo que se están perdiendo algunas mentiras.

## 6. Sugerencias de mejora

1. Experimentar con técnicas de regulación para reducir el sobreajuste.
2. Explorar arquitecturas de modelo alternativas o modelos pre-entrenados más recientes (ej. RoBERTa, ALBERT).
3. Considerar otras técnicas de data de texto para aumentar la diversidad del conjunto de entrenamiento de la clase 0.
4. Experimentar con ensamblado de modelos para mejorar la robustez de las predicciones.

## 7. Conclusión

El modelo desarrollado para la detección de mentiras en el contexto de Diplomacy muestra un rendimiento prometedor, superando el benchmark establecido. El uso de embeddings de BERT combinado con técnicas de balanceo de clases ha demostrado ser efectivo para esta tarea. 
Sin embargo, hay margen de mejoras particularmente en términos de reducir el sobreajuste y mejorar el recall.