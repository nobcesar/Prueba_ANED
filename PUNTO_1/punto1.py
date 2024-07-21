# PNL DIPLOMACY
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import time
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tqdm import tqdm
import os 
import sys

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"El archivo {file_path} no se encuentra.")
    except json.JSONDecodeError:
        print(f"Error al decodificar JSON en el archivo {file_path}.")
    except Exception as e:
        print(f"Se produjo un error: {e}")
    return data

def get_script_directory():
    # Obtiene la ruta completa del archivo .py que se está ejecutando
    script_path = os.path.abspath(sys.argv[0])
    # Obtiene el directorio del archivo .py
    script_directory = os.path.dirname(script_path)
    return script_directory

# Obtener el directorio del script y mostrarlo
current_script_directory = get_script_directory()
print(f"La ruta del directorio del archivo .py es: {current_script_directory}")

#Cargando los .json
train_data = load_jsonl(current_script_directory + r'\data\train.jsonl')
test_data = load_jsonl(current_script_directory + r'\data\test.jsonl')
validation_data = load_jsonl(current_script_directory + r'\data\validation.jsonl')


#función para extraer columna mensaje y etiquetas que son mis parametros de entrada
def extract_messages_and_labels(data):
    messages = []
    labels = []
    for game in data:
        for message, label in zip(game['messages'], game['sender_labels']):
            messages.append(message)
            labels.append(int(label))
    return messages, labels

#parametros de entrada
#Extrae mensajes y etiquetas
train_messages, train_labels = extract_messages_and_labels(train_data)
val_messages, val_labels = extract_messages_and_labels(validation_data)
test_messages, test_labels = extract_messages_and_labels(test_data)


#Se crean los DataFrames
train_df = pd.DataFrame({'message': train_messages, 'label': train_labels})
val_df = pd.DataFrame({'message': val_messages, 'label': val_labels})
test_df = pd.DataFrame({'message': test_messages, 'label': test_labels})


# EDA - Análisis Exploratorio de Datos
# Imprimir información sobre los datos
print(f"Tamaño conjunto de entrenamiento: {len(train_df)}")
print(f"Tamaño conjunto de prueba: {len(test_df)}")
print(f"Tamaño conjunto de validación: {len(val_df)}")
print(f"Distribución de etiquetas en el conjunto de entrenamiento:")
print(train_df['label'].value_counts(normalize=True))

# Gráfico de barras de la distribución de etiquetas
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=train_df)
plt.title('Distribución de etiquetas en el conjunto de entrenamiento')
plt.xlabel('Etiqueta')
plt.ylabel('Recuento')
plt.show()

#Longitud promedio de los mensajes
train_df['message_length'] = train_df['message'].apply(len)
test_df['message_length'] = test_df['message'].apply(len)
val_df['message_length'] = val_df['message'].apply(len)
print(f"Longitud promedio de los mensajes en el conjunto de train: {train_df['message_length'].mean():.2f}")
print(f"Longitud promedio de los mensajes en el conjunto test: {test_df['message_length'].mean():.2f}")
print(f"Longitud promedio de los mensajes en el conjunto de val: {val_df['message_length'].mean():.2f}")

# Gráfico de distribución de longitud de mensajes
plt.figure(figsize=(8, 6))
sns.histplot(data=train_df, x='message_length', bins=30, kde=True)
plt.title('Distribución de longitud de mensajes en el conjunto de entrenamiento')
plt.xlabel('Longitud del mensaje')
plt.ylabel('Frecuencia')
plt.show()

#Descargando las palabras stopwords
nltk.download('stopwords')

# Obtener las palabras de parada en inglés
stop_words = set(stopwords.words('english'))

# Función para eliminar las palabras de parada de un mensaje
def remove_stopwords(message):
    words = message.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)



#Longitud promedio de los mensajes
train_df['message_length'] = train_df['message'].apply(len)
test_df['message_length'] = test_df['message'].apply(len)
val_df['message_length'] = val_df['message'].apply(len)
print(f"Longitud promedio de los mensajes en el conjunto de train: {train_df['message_length'].mean():.2f}")
print(f"Longitud promedio de los mensajes en el conjunto test: {test_df['message_length'].mean():.2f}")
print(f"Longitud promedio de los mensajes en el conjunto de val: {val_df['message_length'].mean():.2f}")


# Eliminar las palabras de parada de los mensajes en train_df
train_df['message_filtered'] = train_df['message'].apply(remove_stopwords)

# Eliminar las palabras de parada de los mensajes en test_df
test_df['message_filtered'] = test_df['message'].apply(remove_stopwords)

# Eliminar las palabras de parada de los mensajes en val_df
val_df['message_filtered'] = val_df['message'].apply(remove_stopwords)

#Longitud promedio de los mensajes
train_df['message_length'] = train_df['message_filtered'].apply(len)
test_df['message_length'] = test_df['message_filtered'].apply(len)
val_df['message_length'] = val_df['message_filtered'].apply(len)

print("\nLongitud promedio de los mensajes despues de eliminar las palabras stopwords\n")
print(f"Longitud promedio de los mensajes en el conjunto de train: {train_df['message_length'].mean():.2f}")
print(f"Longitud promedio de los mensajes en el conjunto test: {test_df['message_length'].mean():.2f}")
print(f"Longitud promedio de los mensajes en el conjunto de val: {val_df['message_length'].mean():.2f}")

# Verificar si CUDA está disponible y seleccionar el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print("Usando GPU:", torch.cuda.get_device_name(0))
else:
    print("Usando CPU")

# Mover el modelo a CPU si no hay suficiente memoria en GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Cargar el tokenizador y el modelo pre-entrenado de BERT
#tokenizer = BertTokenizer.from_pretrained(ruta_raiz + r'\bert-master')
#bert_model = BertModel.from_pretrained(ruta_raiz + r'\bert-master')


#Cargar el tokenizador y el modelo pre-entrenado de BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)

# Mover el modelo a CPU si no hay suficiente memoria en GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

#procesa los datos en lotes más pequeños para evitar problemas de memoria
def get_embeddings(df, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df['message_filtered'].iloc[i:i+batch_size].tolist()
        encodings = tokenizer(batch, truncation=True, padding=True, return_tensors='pt')
        
        # Mover los tensores al dispositivo adecuado
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

# Obtener embeddings para cada conjunto de datos
print("Procesando embeddings del conjunto de train...")
train_embeddings = get_embeddings(train_df)

print("Procesando embeddings del conjunto de test...")
test_embeddings = get_embeddings(test_df)

print("Procesando embeddings del conjunto de validación...")
val_embeddings = get_embeddings(val_df)

# Aplicar SMOTE para sobremuestreo en el conjunto de entrenamiento
print("Aplicando SMOTE...")
smote = SMOTE(random_state=42)
train_embeddings_resampled, train_labels_resampled = smote.fit_resample(train_embeddings, train_df['label'])

print("Procesamiento completado.")


#TSNE es una técnica de reducción de dimensionalidad que se utiliza para visualizar datos de alta dimensión en un espacio de menor dimensión
def plot_embeddings(embeddings, labels, title):
    # Reducir dimensionalidad con t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Crear un DataFrame para facilitar la visualización
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels
    })
    
    # Crear el gráfico
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='deep')
    plt.title(title)
    plt.show()

# Visualizar embeddings originales
plot_embeddings(train_embeddings, train_df['label'], "Distribución de embeddings originales")

# Visualizar embeddings después de SMOTE
plot_embeddings(train_embeddings_resampled, train_labels_resampled, "Distribución de embeddings después de SMOTE")

# Analizar la distribución de clases
print("Distribución de clases original:")
print(pd.Series(train_df['label']).value_counts(normalize=True))

print("\nDistribución de clases después de SMOTE:")
print(pd.Series(train_labels_resampled).value_counts(normalize=True))


# Convertir los embeddings y etiquetas a tensores de PyTorch
# Datos de entrenamiento (ya balanceados con SMOTE)
train_embeddings = torch.FloatTensor(train_embeddings_resampled)
train_labels = torch.LongTensor(train_labels_resampled)

# Datos de validación
val_embeddings = torch.FloatTensor(val_embeddings)  #messages embeddings
val_labels = torch.LongTensor(val_labels)

# Datos de prueba
test_embeddings = torch.FloatTensor(test_embeddings)
test_labels = torch.LongTensor(test_labels)

# Crear datasets
# Un TensorDataset combina los embeddings con sus etiquetas correspondientes
train_dataset = TensorDataset(train_embeddings, train_labels)
val_dataset = TensorDataset(val_embeddings, val_labels)
test_dataset = TensorDataset(test_embeddings, test_labels)

# Crear dataloaders
# Los DataLoaders permiten cargar los datos en lotes (batches) durante el entrenamiento y evaluación
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)


# Definir el modelo
class BERTClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BERTClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

# Inicializar el modelo
input_dim = train_embeddings.shape[1]  # Dimensión de los embeddings de BERT
hidden_dim = 128 #256 # numero de neuronas en la capa oculta
output_dim = 2  # Binario: mentira o no mentira
model = BERTClassifier(input_dim, hidden_dim, output_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)





# Usar las etiquetas después de SMOTE para los pesos
train_labels_np = train_labels_resampled.cpu().numpy() if isinstance(train_labels_resampled, torch.Tensor) else np.array(train_labels_resampled)
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_np), y=train_labels_np)
class_weights = torch.FloatTensor(class_weights).to(device)


# Ajustar los pesos manualmente
adjustment_factor = 1.5  # Dar 50% más de peso a la clase 1
class_weights[1] *= adjustment_factor

# Normalizar los pesos para que sumen 1
class_weights /= class_weights.sum()

# Actualizar el criterio con los pesos calculados
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5 , weight_decay=0.01) #lr es la tasa de aprendizaje y weight_decay es la regularización L2



print("Pesos personalizados de las clases:")
for i, weight in enumerate(class_weights):
    print(f"Clase {i}: {weight.item():.4f}")

# print(criterion)

# Función de entrenamiento
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for embeddings, labels in tqdm(dataloader):
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Función de evaluación
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for embeddings, labels in tqdm(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    accuracy = sum(1 for x,y in zip(all_predictions, all_labels) if x == y) / len(all_labels)
    
    return avg_loss, accuracy, precision, recall, f1

# Función de early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# Entrenamiento con early stopping opcional y medición de tiempo
num_epochs = 100 # Número máximo de épocas
use_early_stopping = True  # Cambia a False para desactivar early stopping
early_stopping = EarlyStopping(patience=8, min_delta=0.001) if use_early_stopping else None

start_time = time.time()
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f'Epoch {epoch + 1}/{num_epochs}')
    train_loss = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    print(f'Training loss: {train_loss:.4f}')
    
    val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)

    val_losses.append(val_loss)
    print(f'Validation loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation Precision: {val_precision:.4f}')
    print(f'Validation Recall: {val_recall:.4f}')
    print(f'Validation F1-score: {val_f1:.4f}')
    
    epoch_time = time.time() - epoch_start_time
    print(f'Epoch time: {epoch_time:.2f} seconds')

    if use_early_stopping:
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

total_time = time.time() - start_time
print(f'Total training time: {total_time:.2f} seconds')

#Evaluación final en el conjunto de prueba
test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)
print('Test Results:')
print(f'Test loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F1-score: {test_f1:.4f}')

# Guardar el modelo
torch.save(model.state_dict(), 'bert_embedding_classifier_smote_false.pt')





# Asumimos que tienes estas listas con los valores de pérdida para cada época
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, 'b', label='Pérdida de entrenamiento')
plt.plot(epochs, val_losses, 'r', label='Pérdida de validación')
plt.title('Curva de Aprendizaje')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# Guardar la figura
plt.savefig('curva_aprendizaje.png')

# Mostrar la figura
plt.show()



#CALCULO DEL MACRO F1 SCORE Y LIE F1 SCORE

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for embeddings, labels in tqdm(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    accuracy = sum(1 for x,y in zip(all_predictions, all_labels) if x == y) / len(all_labels)
    
    # Calcular el Macro F1 score
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    
    # Calcular el Lie F1 score (asumiendo que la clase de mentiras es 1)
    lie_f1 = f1_score(all_labels, all_predictions, pos_label=1)
    
    return avg_loss, accuracy, precision, recall, f1, macro_f1, lie_f1

val_loss, val_accuracy, val_precision, val_recall, val_f1, val_macro_f1, val_lie_f1 = evaluate(model, val_loader, criterion, device)
print(f'Validation Macro F1-score: {val_macro_f1:.4f}')
print(f'Validation Lie F1-score: {val_lie_f1:.4f}')

test_loss, test_accuracy, test_precision, test_recall, test_f1, test_macro_f1, test_lie_f1 = evaluate(model, test_loader, criterion, device)
print(f'Test Macro F1-score: {test_macro_f1:.4f}')
print(f'Test Lie F1-score: {test_lie_f1:.4f}')
