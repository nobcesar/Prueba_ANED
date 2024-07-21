# Procesador de Archivos RUAF

Este script procesa el archivo en formato csv llamado RUAF.csv y los convierte en un formato estructurado de Excel.

## Requisitos

- Python 3.7+
- pandas
- chardet
- os 
- sys

## Puedes instalar las dependencias:
pip install pandas chardet

## Uso

1. Coloca tu archivo RUAF.csv en el directorio donde esta el script pero dentro de la carpeta data.
2. Ejecuta el script con: estructura_RUF.py
3. El script generará un archivo 'resultado_estructurado_RUAF.xlsx' dentro de la carpeta result.

## Estructura del Código

- `read_file`: Lee el archivo csv y maneja la codificación.
- `process_ruaf_content`: Procesa el contenido del archivo RUAF.
- `create_dataframe`: Convierte los registros procesados en un DataFrame.
- `ensure_all_columns`: Asegura que todas las columnas requeridas estén presentes y en el orden correcto.
- `save_to_excel`: Guarda el DataFrame resultante en un archivo Excel.
- `process_ruaf_file`: Función principal que coordina todo el proceso.

## Notas

- El script asume que el archivo de entrada se llama 'RUAF.csv'. Si tu archivo tiene un nombre diferente, modifica la variable `input_file` en la sección `if __name__ == "__main__":` del script.
- Asegúrate de que el archivo RUAF.csv esté en el formato correcto, con los separadores y estructura esperados.

## Autor
Cesar Augusto Maldonado Parra