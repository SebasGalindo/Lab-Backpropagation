# file for store functions that are used in differents files of the proyect

import os # Import the os module to interact with the operating system
import sys # Import the sys module to interact with the Python interpreter
import random # Import the random module to generate random numbers
import math # Import the math module to perform mathematical operations
import json # Import the json module to work with JSON files
import shutil # Import the shutil module to perform file operations
import webbrowser # Import the webbrowser module to open web pages
from tkinter import filedialog # Import the filedialog module to open file dialogs

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    params:
        relative_path: relative path to the resource
    return: absolute path to the resource
    """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def open_link(event = None, link = None):
    """
    Open the Link in the default browser.
    
    Parameters:
    event (tk.Event): Event object.
    """
    webbrowser.open(link)    

def download_json(filename, extension = ".json", data = None, directory = "Data"):
    """
    Save a JSON file in a new location.
    
    Parameters:
    filename (str): Name of the file.
    extension (str): Extension of the file.
    """
    file_path = filedialog.asksaveasfilename(defaultextension=extension, filetypes=[("JSON files", f"*{extension}")], initialfile=f"{filename}{extension}")
    if data:    
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
            print(f"JSON guardado en: {file_path}")
    elif file_path:
        json_path = get_resource_path(f"{directory}/{filename}{extension}")
        shutil.copy(json_path, file_path)  # Copiar el archivo a la nueva ubicación
        print(f"JSON guardado en: {file_path}")
    
def load_json(filename = None, extension = ".json", directory = "Data"):
    """
    Load a JSON file.
    
    Parameters:
    filename (str): Name of the file.
    extension (str): Extension of the file.
    
    Returns:
    data_json (dict): Dictionary with the JSON data.
    """
    data_json = {}
    if filename is None:
        file = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file:
            file_name = os.path.basename(file)
            with open(file, 'r') as file:
                content = json.load(file)
                data_json = content
                print(f"Contenido del JSON cargado", data_json)
            return data_json, file_name
        else:
            return None, None
        
    json_path = get_resource_path(f"{directory}/{filename}{extension}")

    with open(json_path, 'r', encoding='utf-8') as file:
        content = json.load(file)
        data_json = content
        print(f"Contenido del JSON cargado")

    return data_json

def save_json(data, filename = "resultado", extension = ".json"):
    """
    Save a JSON file.
    
    Parameters:
    data (dict): Dictionary with the JSON data.
    filename (str): Name of the file.
    extension (str): Extension of the file.
    """
    json_path = get_resource_path(f"Data/{filename}{extension}")
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)
        print(f"JSON guardado en: {json_path}")

def secondCase_Data():
    # List initialization
    entradas = []
    salidas = []

    # Generate 100 rows with random values for a, b, c
    for _ in range(100):
        # Random values generation
        a = random.uniform(0, 2 * math.pi)  # a between [0, 2*PI]
        b = random.uniform(0, 2 * math.pi)  # b between [0, 2*PI]
        c = random.uniform(-1, 1)           # c between [-1, 1]
        
        # Output calculation using the random values generated above for a, b, c
        salida = math.sin(a) + math.cos(b) + c
        
        # Add the values to the lists
        entradas.append([a, b, c])
        salidas.append([salida])

    # Create the json data
    data = {
        "inputs": entradas,
        "outputs": salidas
    }


    return data

def add_or_replace_secon_case(second_case_json):

    file_path = get_resource_path("Data/cases.json")
    # Open the file in write mode
    cases_json = {}

    # Check if the file exists
    if os.path.exists(file_path):
        # Read the file
        with open(file_path, "r") as file:
            # Load the data in the file
            cases_json = json.load(file)

    # Add the second case to the dictionary
    cases_json["case_2"] = second_case_json

    # Reeplace the file with the new data
    with open(file_path, "w") as file:
        # Write the new data
        json.dump(cases_json, file, indent=4)
    

if __name__ == "__main__":
    # Generate the data for the second case
    data = secondCase_Data()
    # Save the data in a JSON file
    save_json(data, "testeo_datos_caso_2") 