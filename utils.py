import os
import sys
import random
import math
import json

# file for store functions that are used in differents files of the proyect
def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    params:
        relative_path: relative path to the resource
    return: absolute path to the resource
    """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

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
        salidas.append(salida)

    # Create the json data
    data = {
        "entradas": entradas,
        "salidas": salidas
    }


    return data

def add_or_replace_secon_case(second_case_json):

    file_path = get_resource_path("cases.json")
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