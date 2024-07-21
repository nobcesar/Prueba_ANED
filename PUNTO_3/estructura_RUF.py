import pandas as pd
import chardet , os , sys
from IPython import get_ipython

def get_notebook_dir():
    try:
        #Esta línea obtiene el path completo del script en ejecución
        notebook_path = os.path.abspath(__file__) #sys.argv[0]
        # Esta línea obtiene el directorio del script
        notebook_dir = os.path.dirname(notebook_path)
        return notebook_dir
    except Exception as e:
        print(f"Error al obtener la ruta del directorio del notebook: {e}")
        return None
# Obtener el directorio del script y mostrarlo
current_script_directory = get_notebook_dir()

def read_file(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    detected = chardet.detect(raw_data)
    encoding = detected['encoding']
    
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        return content
    except UnicodeDecodeError:
        print(f"No se pudo decodificar el archivo con {encoding}. Probando con 'latin-1'.")
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()
        return content

def process_ruaf_content(content):
    records = content.split('¬-*-¬DATA')
    processed_records = []

   # Mapeo de campos
    field_mapping = {
        'INFORMACIÓN BASICA': {
            'Primer Nombre': 'primer_nombre',
            'Segundo Nombre': 'segundo_nombre',
            'Primer Apellido': 'primer_apellido',
            'Segundo Apellido': 'segundo_apellido',
            'Sexo': 'sexo'
        },
        'AFILIACIÓN A SALUD': {
            'Administradora': 'admon_salud',
            'Régimen': 'regimen_salud',
            'Fecha Afiliacion': 'f_afiliacion_salud',
            'Fecha de Afiliacion': 'f_afiliacion_salud',
            'Estado de Afiliación': 'est_afiliacion_salud',
            'Tipo de Afiliado': 'tipo_afiliado_salud',
            'Departamento -> Municipio': 'dep_municipio'
        },
        'AFILIACIÓN A PENSIONES': {
            'Régimen': 'regimen_pension',
            'Administradora': 'admon_pension',
            'Fecha de Afiliación': 'f_afiliacion_pension',
            'Estado de Afiliación': 'est_afiliacion_pension'
        },
        'AFILIACIÓN A RIESGOS LABORALES': {
            'Administradora': 'admon_rl',
            'Fecha de Afiliación': 'f_afiliacion_rl',
            'Estado de Afiliación': 'est_afiliacion_rl',
            'Actividad Economica': 'act_economica_rl',
            'Municipio Labora': 'municipio_laboral_rl'
        },
        'AFILIACIÓN A COMPENSACIÓN FAMILIAR': {
            'Administradora CF': 'admon_cf',
            'Fecha de Afiliación': 'f_afiliacion_cf',
            'Estado de Afiliación': 'est_afiliacion_cf',
            'Tipo de Miembro de la Población Cubierta': 'tipo_miembro_pobl_cf',
            'Tipo de Afiliado': 'tipo_afiliado_cf',
            'Municipio Labora': 'municipio_laboral_cf'
        },
        'AFILIACIÓN A CESANTIAS': {
            'Administradora': 'admon_cesantias',
            'Fecha de Afiliación': 'f_afiliacion_cesantias',
            'Estado de Afiliación': 'est_afiliacion_cesantias',
            'Régimen': 'regimen_cesantias',
            'Municipio Labora': 'municipio_laboral_cesantias'
        },
        'PENSIONADOS': {
            'Entidad que reconoce la pensión': 'entidad_pension',
            'Fecha Resolución': 'f_resolucion_pension',
            'Estado': 'est_pension',
            'Modalidad': 'modalidad_pension',
            'Número Resoluciòn Pension PG': 'num_resolucion_pension',
            'Tipo de Pensión': 'tipo_pension',
            'Tipo de Pensionado': 'tipo_pensionado'
        },
        'VINCULACIÓN A PROGRAMAS DE ASISTENCIA SOCIAL': {
            'Administradora': 'admon_vpas',
            'Fecha de Vinculación': 'f_vinculacion_vpas',
            'Estado de la Vinculación': 'est_vinculacion_vpas',
            'Estado del Beneficio': 'est_beneficio_vpas',
            'Fecha Ultimo Beneficio': 'f_ult_beneficio_vpas',
            'Programa': 'programa_vpas',
            'Ubicación de Entrega del Beneficio': 'ubicacion_entrega_vpas'
        }
    }

    for record in records:
        if not record.strip():
            continue

        processed_record = {
            'key_fep': '',
            'origen': 'RUAF',
            'f_consulta': '',
            'num_id': '',
            'tipo_id_num': '',
            'tipo_id_str': '',
            'marca_sin_informacion': 1
        }

        lines = record.strip().split('\n')

        for line in lines:
            parts = line.split(';')
            
            if '¬-*-¬ID' in parts[0]:
                processed_record['key_fep'] = parts[1]
                processed_record['f_consulta'] = parts[2]
            
            elif '|' in parts[0] and len(parts) > 4:
                id_parts = parts[0].split('|')
                processed_record['tipo_id_num'] = id_parts[0]
                processed_record['tipo_id_str'] = id_parts[1]
                processed_record['num_id'] = parts[1]
                
                section = parts[2]
                field = parts[3]
                value = parts[4]
                
                if section in field_mapping and field in field_mapping[section]:
                    field_name = field_mapping[section][field]
                    processed_record[field_name] = value
                    processed_record['marca_sin_informacion'] = 0

        processed_records.append(processed_record)

    return processed_records

def create_dataframe(processed_records):
    return pd.DataFrame(processed_records)

def ensure_all_columns(df):
    all_columns = [
        'key_fep', 'origen', 'f_consulta', 'num_id', 'tipo_id_num', 'tipo_id_str',
        'primer_nombre', 'segundo_nombre', 'primer_apellido', 'segundo_apellido', 'sexo',
        'admon_salud', 'regimen_salud', 'f_afiliacion_salud', 'est_afiliacion_salud', 'tipo_afiliado_salud', 'dep_municipio',
        'regimen_pension', 'admon_pension', 'f_afiliacion_pension', 'est_afiliacion_pension',
        'admon_rl', 'f_afiliacion_rl', 'est_afiliacion_rl', 'act_economica_rl', 'municipio_laboral_rl',
        'admon_cf', 'f_afiliacion_cf', 'est_afiliacion_cf', 'tipo_miembro_pobl_cf', 'tipo_afiliado_cf', 'municipio_laboral_cf',
        'admon_cesantias', 'f_afiliacion_cesantias', 'est_afiliacion_cesantias', 'regimen_cesantias', 'municipio_laboral_cesantias',
        'entidad_pension', 'f_resolucion_pension', 'est_pension', 'modalidad_pension', 'num_resolucion_pension', 'tipo_pension', 'tipo_pensionado',
        'admon_vpas', 'f_vinculacion_vpas', 'est_vinculacion_vpas', 'est_beneficio_vpas', 'f_ult_beneficio_vpas', 'programa_vpas', 'ubicacion_entrega_vpas', 'marca_sin_informacion'
    ]
    
    for col in all_columns:
        if col not in df.columns:
            df[col] = pd.NA
    
    return df[all_columns]

def save_to_excel(df, output_file):
    df.to_excel(output_file, index=False)
    print(f"Archivo guardado como {output_file}")

def process_ruaf_file(input_file, output_file):
    content = read_file(input_file)
    processed_records = process_ruaf_content(content)
    df = create_dataframe(processed_records)
    df = ensure_all_columns(df)
    save_to_excel(df, output_file)


if __name__ == "__main__":

    # Obtener el directorio del script y mostrarlo
    ruta_notebook_jupyter = get_notebook_dir()   
    print(f"Directorio del script: {ruta_notebook_jupyter}")

    input_file = ruta_notebook_jupyter + r'\data\RUAF.csv'
    output_file =ruta_notebook_jupyter +  r'\result\resultado_estructurado_RUAF.xlsx'
    process_ruaf_file(input_file, output_file)