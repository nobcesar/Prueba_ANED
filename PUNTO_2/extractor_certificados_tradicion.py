import json
import re
import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from unidecode import unidecode
from fuzzywuzzy import fuzz

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError:
        log_message(f"Error al cargar el archivo JSON: {file_path}")
        return None
    except Exception as e:
        log_message(f"Ocurrió un error inesperado al cargar el archivo: {file_path}\nError: {e}")
        return None

def normalize_text(text):
    return unidecode(text).lower()

def fuzzy_search(text, patterns, threshold=80):
    for pattern in patterns:
        normalized_text = normalize_text(text)
        normalized_pattern = normalize_text(pattern)
        ratio = fuzz.partial_ratio(normalized_text, normalized_pattern)
        if ratio >= threshold:
            return True
    return False

def extract_with_geometry(blocks, block_type, left_range, top_range, patterns, fuzzy=False):
    for block in blocks:
        if isinstance(block, dict) and block.get('BlockType') == block_type:
            geometry = block.get('Geometry')
            if geometry:
                left = geometry.get('BoundingBox', {}).get('Left', 0)
                top = geometry.get('BoundingBox', {}).get('Top', 0)
                if left_range[0] <= left <= left_range[1] and top_range[0] <= top <= top_range[1]:
                    if fuzzy:
                        if fuzzy_search(block.get('Text', ''), patterns):
                            return block.get('Text', '')
                    else:
                        for pattern in patterns:
                            match = re.search(pattern, normalize_text(block.get('Text', '')))
                            if match:
                                return match.group(1) if match.groups() else match.group(0)
    return ""

def extract_estado_folio(blocks):
    for i, block in enumerate(blocks):
        if isinstance(block, dict) and block.get('BlockType') == 'LINE':
            normalized_text = normalize_text(block.get('Text', ''))
            if 'estado del folio:' in normalized_text:
                for next_block in blocks[i+1:]:
                    if isinstance(next_block, dict) and next_block.get('BlockType') in ['LINE', 'WORD']:
                        return next_block.get('Text', '').strip()
    
    # Si no se encuentra en LINE, buscar en WORD
    for block in blocks:
        if isinstance(block, dict) and block.get('BlockType') == 'WORD':
            normalized_text = normalize_text(block.get('Text', ''))
            if normalized_text in ['activo', 'inactivo']:
                return block.get('Text', '').strip()
    
    return ""

def extract_from_circulo_registral(text, field):
    patterns = {
        'Departamento': r'depto:\s*([\w\s]+?)(?=\s+municipio|$)',
        'Municipio': r'municipio:\s*([\w\s]+?)(?=\s+vereda|$)',
        'Vereda': r'vereda:\s*([\w\s]+)$'
    }
    normalized_text = normalize_text(text)
    match = re.search(patterns[field], normalized_text)
    return match.group(1).strip() if match else ""

def parse_date(date_str):
    date_match = re.search(r'(\d{1,2} de [A-Za-zá-ú]+ de \d{4})', date_str)
    if date_match:
        date_str = date_match.group(1)
        meses = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
        }
        day, month, year = date_str.split(' de ')
        month_num = meses[month.lower()]
        return f"{year}-{month_num}-{day.zfill(2)}"
    return ""

def process_json_file(file_path):
    data = load_json(file_path)
    if data is None:
        return None

    blocks = data.get('Blocks', [])

    nro_matricula = extract_with_geometry(blocks, 'LINE', (0.68, 0.70), (0.10, 0.12), [r'nro\s*matricula:?\s*(\d{3}[-\s]?\d+)', r'nro matricuula'], fuzzy=True)
    if nro_matricula:
        nro_matricula = re.sub(r'nro\s*matricula:?\s*', '', normalize_text(nro_matricula), flags=re.IGNORECASE)
        nro_matricula = re.sub(r'\s+', '', nro_matricula)
        if '-' not in nro_matricula and len(nro_matricula) == 9:
            nro_matricula = f"{nro_matricula[:3]}-{nro_matricula[3:]}"
    
    results = {
        "Nro_Matricula": nro_matricula.strip(),
        "Fecha_Impresion": "",
        "Departamento": "",
        "Municipio": "",
        "Vereda": "",
        "ESTADO DEL FOLIO": extract_estado_folio(blocks)
    }

    fecha_str = extract_with_geometry(blocks, 'LINE', (0.27, 0.29), (0.13, 0.16), [r'impreso el (.+)'])
    if fecha_str:
        results["Fecha_Impresion"] = parse_date(fecha_str)

    circulo_registral = extract_with_geometry(blocks, 'LINE', (0.02, 0.04), (0.22, 0.24), [r'circulo registral:.*'])
    if circulo_registral:
        results["Departamento"] = extract_from_circulo_registral(circulo_registral, 'Departamento')
        results["Municipio"] = extract_from_circulo_registral(circulo_registral, 'Municipio')
        results["Vereda"] = extract_from_circulo_registral(circulo_registral, 'Vereda')

    if not results["Departamento"]:
        results["Departamento"] = extract_with_geometry(blocks, 'WORD', (0.30, 0.33), (0.22, 0.24), [r'depto:\s*([\w\s]+)'])
    if not results["Municipio"]:
        results["Municipio"] = extract_with_geometry(blocks, 'WORD', (0.43, 0.46), (0.22, 0.24), [r'municipio:\s*([\w\s]+)'])
    if not results["Vereda"]:
        results["Vereda"] = extract_with_geometry(blocks, 'WORD', (0.58, 0.64), (0.22, 0.24), [r'vereda:\s*([\w\s]+)'])

    log_message(f"\nResultados para el archivo: {file_path}")
    for key, value in results.items():
        log_message(f"{key}: {value}")

    return results

def process_all_json_files(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            result = process_json_file(file_path)
            if result:
                results.append(result)
            log_message("=" * 30)
    return results

def save_results_to_csv(results, output_file):
    fieldnames = ["Nro_Matricula", "Fecha_Impresion", "Departamento", "Municipio", "Vereda", "ESTADO DEL FOLIO"]
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    log_message(f"Resultados guardados en {output_file}")

def log_message(message):
    print(message)
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_directory, f'log_{current_date}.txt')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()}: {message}\n")

def run_extraction():
    json_directory = filedialog.askdirectory(title="Seleccione la carpeta con los archivos JSON")
    if not json_directory:
        return
    
    results = process_all_json_files(json_directory)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_file = os.path.join(results_directory, f'resultados_{current_date}.csv')
    save_results_to_csv(results, output_file)
    
    messagebox.showinfo("Proceso Completado", f"Se han procesado {len(results)} archivos.\nLos resultados se han guardado en {output_file}")


import sys
def get_script_directory():
    # Obtiene la ruta completa del archivo .py que se está ejecutando
    script_path = os.path.abspath(sys.argv[0])
    # Obtiene el directorio del archivo .py
    script_directory = os.path.dirname(script_path)
    return script_directory

# Obtener el directorio del script y mostrarlo
current_script_directory = get_script_directory()
print(f"La ruta del directorio del archivo .py es: {current_script_directory}")


results_directory = os.path.join(current_script_directory, 'results')
logs_directory = os.path.join(current_script_directory, 'logs')
print('Directorio de logs:', logs_directory)

os.makedirs(results_directory, exist_ok=True)

os.makedirs(logs_directory, exist_ok=True)
print(f"¿Existe la carpeta de logs?: {os.path.exists(logs_directory)}")

# Interfaz gráfica
root = tk.Tk()
root.title("Certificados de tradicción")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

tk.Label(frame, text="Extractor de Información de Certificados", font=("Helvetica", 16)).pack(pady=10)
tk.Button(frame, text="Seleccionar Carpeta y Procesar .json", command=run_extraction).pack(pady=20)

print(f"Directorio de resultados: {results_directory}")
print(f"Directorio de logs: {logs_directory}")


root.mainloop()