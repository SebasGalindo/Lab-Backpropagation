import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import END
import tkinter as tk
from PIL import Image
import json
import threading

from utils import get_resource_path, open_link, download_json, load_json, save_json
from backpropagation import backpropagation_training, test_neural_network, set_stop_training, get_graph_json, get_weights_json
from graphics import plot_neural_network_with_labels, plot_error_total_by_epoch

ctk.set_appearance_mode("dark")
# Global variables
main_window, content_frame = None, None
download_weights_btn, download_training_data_btn, results_btn = None, None, None
# Data variables
data_train_json, data_test_json, weights_test_json = None, None, None
num_excercise = 0

labels_error = []
epoch_label, total_error_label = None, None

def main_windowSetup():
    main_window = ctk.CTk()
    main_window.title("Inteligencia Artificial - Backpropagation")
    main_window.geometry("1200x700")
    main_window.resizable(False, True)
    icon_path = get_resource_path("Resources/brand_logo.ico")
    main_window.iconbitmap(icon_path)
    main_window.lift()
    main_window.grid_columnconfigure(1, weight=0) 
    main_window.grid_columnconfigure(2, weight=1) 
    for i in range(12):
        main_window.grid_rowconfigure(i, weight=1)
    return main_window

def sidebar_creation(master_window):
    sidebar = ctk.CTkFrame(master=master_window, width=200, fg_color="#11371A", corner_radius=0)
    sidebar.grid(row=0, column=0, sticky="nsew", rowspan=12)

    # region Seccion de Elementos de Presentación

    # UdeC logo creation
    logo_path = get_resource_path("Resources/logo_UdeC.png")
    logo_UdeC = Image.open(logo_path)
    logo_UdeC = ctk.CTkImage(dark_image=logo_UdeC, size=(60, 90))

    # title creation
    title_lbl = ctk.CTkLabel(master=sidebar, text="  Backpropagation", image=logo_UdeC, font=("Arial", 18), compound="left", cursor="hand2", text_color="#ffffff")
    title_lbl.bind("<Button-1>", initial_frame)
    title_lbl.grid(row=0, column=0, pady=10, sticky="n")

    # horizontal separator
    ctk.CTkFrame(master=sidebar, height=2, fg_color="#0B2310").grid(row=1, column=0, sticky="ew", pady=5)

    # Authors section 
    authors_lbl = ctk.CTkLabel(master=sidebar, text="Autores: \nJohn Sebastián Galindo Hernández \nMiguel Ángel Moreno Beltrán", font=("Arial", 16), text_color="#ffffff")
    authors_lbl.grid(row=2, column=0, pady=10, padx=5, sticky="n")

    # endregion

    # horizontal separator
    ctk.CTkFrame(master=sidebar, height=2, fg_color="#0B2310").grid(row=3, column=0, sticky="ew", pady=5)

    # Buttons section
    excersice_one_btn = ctk.CTkButton(master=sidebar, text="Ejercicio Uno", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    excersice_one_btn.grid(row=4, column=0, pady=10, sticky="n")
    excersice_one_btn.configure(command= lambda: excersice_frame_creation(master_window, 1))

    excersice_two_btn = ctk.CTkButton(master=sidebar, text="Ejercicio Dos", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    excersice_two_btn.grid(row=5, column=0, pady=10, sticky="n")
    excersice_two_btn.configure(command= lambda: excersice_frame_creation(master_window, 2))

    excersice_three_btn = ctk.CTkButton(master=sidebar, text="Ejercicio Tres", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    excersice_three_btn.grid(row=6, column=0, pady=10, sticky="n")
    excersice_three_btn.configure(command= lambda: excersice_frame_creation(master_window, 3))

    excersice_four_btn = ctk.CTkButton(master=sidebar, text="Ejercicio Cuatro", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    excersice_four_btn.grid(row=7, column=0, pady=10, sticky="n")
    excersice_four_btn.configure(command= lambda: excersice_frame_creation(master_window, 4))

    train_btn = ctk.CTkButton(master=sidebar, text="Realizar Nuevo Entrenamiento", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    train_btn.grid(row=8, column=0, pady=10, sticky="n")
    train_btn.configure(command= lambda: excersice_frame_creation(master=master_window))

    test_solution_btn = ctk.CTkButton(master=sidebar, text="Probar Soluciones", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    test_solution_btn.grid(row=9, column=0, pady=10, sticky="n")
    test_solution_btn.configure(command = lambda: test_frame_creation(master= master_window, number_excercise = 0))

    return sidebar

def grid_setup(frame):
    for i in range(12): 
        frame.grid_rowconfigure(i, weight=1)
        frame.grid_columnconfigure(i, weight=1)
    return frame

def principal_frame_creation(master_window):

    principal_frame = ctk.CTkScrollableFrame(master=master_window, corner_radius=0, fg_color="#11371A",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    principal_frame.grid(row=0, column=2, sticky="nsew", rowspan=12)
    principal_frame = grid_setup(principal_frame)

    return principal_frame

def explanation_frame_creation(master_window, num_excercise):
    explanation_frame = ctk.CTkScrollableFrame(master=master_window, corner_radius=8, fg_color="#154721", scrollbar_button_color="#154721", scrollbar_button_hover_color="#154721", height=400)
    explanation_frame.grid(row=0, column=0, sticky="nsew", columnspan=12, rowspan=6, padx=10, pady=5)
    explanation_frame = grid_setup(explanation_frame)

    # Get the explanation.json file
    explanation_json = load_json("explanations")
    explanation_json = explanation_json[f"case_{num_excercise}"]

    # Put the title, description, type excersice, inputs text and outputs text, number of patterns and the requiremets
    # Title
    title_txt = "Ejercicio Número: " + str(num_excercise)
    title = ctk.CTkLabel(master=explanation_frame, text=title_txt, font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=5, sticky="nsew", columnspan=12)

    # Description
    description_txt = explanation_json["description"]
    description = ctk.CTkLabel(master=explanation_frame, text=description_txt, font=("Arial", 16), justify="left", wraplength=860)
    description.grid(row=1, column=0, pady=5, sticky="w", columnspan=12)

    # Type of excersice
    type_txt = "Tipo de Ejercicio:"
    type_lbl = ctk.CTkLabel(master=explanation_frame, text=type_txt, font=("Arial", 16, "bold"), justify="left", wraplength=900, text_color="#fbe122")
    type_lbl.grid(row=2, column=0, pady=5, sticky="w", columnspan=2)

    type_txt2 = explanation_json["type"]
    type_lbl2 = ctk.CTkLabel(master=explanation_frame, text=type_txt2, font=("Arial", 16), justify="left", wraplength=900)
    type_lbl2.grid(row=2, column=2, pady=5, sticky="w", columnspan=10)

    # Inputs text
    inputs_txt = "Entradas: "
    inputs_lbl = ctk.CTkLabel(master=explanation_frame, text=inputs_txt, font=("Arial", 16, "bold"), justify="left", wraplength=800, text_color="#fbe122")
    inputs_lbl.grid(row=3, column=0, pady=5, sticky="w", columnspan=12)

    inputs_txt2 = explanation_json["inputs"]
    inputs_lbl2 = ctk.CTkLabel(master=explanation_frame, text=inputs_txt2, font=("Arial", 16), justify="left", wraplength=860)
    inputs_lbl2.grid(row=4, column=0, pady=5, sticky="w", columnspan=12)

    # Outputs text
    outputs_txt = "Salidas: "
    outputs_lbl = ctk.CTkLabel(master=explanation_frame, text=outputs_txt, font=("Arial", 16, "bold"), justify="left", wraplength=900, text_color="#fbe122")
    outputs_lbl.grid(row=5, column=0, pady=5, sticky="w", columnspan=12)

    outputs_txt2 = explanation_json["outputs"]
    outputs_lbl2 = ctk.CTkLabel(master=explanation_frame, text=outputs_txt2, font=("Arial", 16), justify="left", wraplength=860)
    outputs_lbl2.grid(row=6, column=0, pady=5, sticky="w", columnspan=12)

    # Number of patterns
    patterns_txt = "Número de Patrones: "
    patterns_lbl = ctk.CTkLabel(master=explanation_frame, text=patterns_txt, font=("Arial", 16, "bold"), justify="left", wraplength=900, text_color="#fbe122")
    patterns_lbl.grid(row=7, column=0, pady=5, sticky="w", columnspan=2)

    patterns_txt2 = explanation_json["number_of_patterns"]
    patterns_lbl2 = ctk.CTkLabel(master=explanation_frame, text=patterns_txt2, font=("Arial", 16), justify="left", wraplength=900)
    patterns_lbl2.grid(row=7, column=2, pady=5, sticky="w", columnspan=10)

    # Requirements
    requirements_txt = "Requerimientos: "
    requirements_lbl = ctk.CTkLabel(master=explanation_frame, text=requirements_txt, font=("Arial", 16, "bold"), justify="left", wraplength=900, text_color="#fbe122")
    requirements_lbl.grid(row=8, column=0, pady=5, sticky="w", columnspan=12)

    requirements_txt2 = explanation_json["requirements"]
    requirements_lbl2 = ctk.CTkLabel(master=explanation_frame, text=requirements_txt2, font=("Arial", 16), justify="left", wraplength=860)
    requirements_lbl2.grid(row=9, column=0, pady=5, sticky="w", columnspan=12)

    # Image of the example
    example_path = get_resource_path(explanation_json["example"])
    example_img = Image.open(example_path)
    example_img = ctk.CTkImage(dark_image=example_img, size=(860, 400))

    example_lbl = ctk.CTkLabel(master=explanation_frame, image=example_img, compound="center", text="")
    example_lbl.grid(row=10, column=0, pady=5, sticky="nsew", columnspan=12)

    return explanation_frame

def GUI_creation():

    global main_window, content_frame

    # Build the main window
    main_window = main_windowSetup()

    # Sidebar creation
    sidebar = sidebar_creation(main_window)

    # Vertical separator
    ctk.CTkFrame(master=main_window, width=2, fg_color="#0B2310").grid(row=0, column=1, sticky="ns")

    content_frame = initial_frame(main_window)

    # Run the main window
    main_window.mainloop()

def initial_frame(event=None, master=None):

    principal_frame = principal_frame_creation(master)

    # Create the section title
    title = ctk.CTkLabel(master=principal_frame, text="Sección de Explicación", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=20, sticky="nsew", columnspan=2, padx=15)

    # App Explanation section
    explanation_txt = ("Esta aplicación permite realizar entrenamiento con una red neuronal artifical con el modelo Backpropagation, "
                        "Existen 3 botones para ver los resutados de los ejercicios propuestos en clase."
                        "Por otro lado se cuenta con la funcionalidad de realizar un nuevo entrenamiento y probar soluciones con el modelo entrenado.")
    explanation_lbl = ctk.CTkLabel(master=principal_frame, text=explanation_txt, font=("Arial", 16), justify="left", wraplength=890, text_color="#ffffff")
    explanation_lbl.grid(row=1, column=0, pady=5, sticky="nsew", columnspan=2, padx=15)

    # Training backpropagation explanation section
    # Create the section title
    title2 = ctk.CTkLabel(master=principal_frame, text="Sección de entrenamiento", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title2.grid(row=2, column=0, pady=20, sticky="nsew", columnspan=2, padx=15)
    # explanation text
    explanation2_txt = ("La sección de entrenamiento permite realizar un nuevo entrenamiento con el modelo de backpropagation, "
                        "para ello, se deberá cargar un archivo de datos en formato JSON, "
                        "el cual deberá contener los datos de entrenamiento (Tanto las Entradas como las Salidas). "
                        "Se Aconseja descargar la plantilla de ejemplo para cargar los datos correctamente."
                        "O usar una de las opciones relacionadas con los ejercicios propuestos. \n"
                        "Una vez tenga los datos en formato JSON, deberá cargarlos con el botón 'Cargar Datos'."
                        "Una vez cargados los datos, deberá ingresar el valor de α (tasa de aprendizaje), θ (bias), β (Betha) y la ρ (precisión) deseada para el entrenamiento."
                        "el valor de α y la precisión pueden cambiar dependiendo el problema a resolver pero se recomiendan vaores bajos."
                        "Finalmente, deberá presionar el botón 'Iniciar Entrenamiento' para comenzar el proceso."
                        "Durante el entrenamiento se mostrará cada mil iteraciones el estado del error de cada patrón ingresado."
                        "Una vez finalizado el entrenamiento, se mostrará un gráfico con los errores por época y los pesos por época."
                        )
    explanation2_lbl = ctk.CTkLabel(master=principal_frame, text=explanation2_txt, font=("Arial", 16), justify="left", wraplength=890)
    explanation2_lbl.grid(row=3, column=0, pady=5, sticky="nsew", columnspan=2, padx=15)

    # Test explanation section
    # Create the section title
    title4 = ctk.CTkLabel(master=principal_frame, text="Sección de Pruebas", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title4.grid(row=4, column=0, pady=20, sticky="nsew", columnspan=2, padx=15)
    # explanation text
    explanation4_txt = ("La sección de Pruebas permite realizar pruebas con el modelo entrenado, "
                        "para ello, se deberá cargar dos archivos en formato JSON, "
                        "uno con las entradas y otro con los datos usados en el entrenamiento."
                        "La opcion por defecto es usar el boton que permite cargar el último entrenamiento realizado y que está alojaddo en la app."
                        "Una vez cargados los datos, se deberá presionar el botón 'Probar Soluciones' para comenzar el proceso."
                        "Finalmente, se mostrara una la lista de entradas, salidas deseadas, salidas obtenidas y error obtenido."
                        )
    explanation4_lbl = ctk.CTkLabel(master=principal_frame, text=explanation4_txt, font=("Arial", 16), justify="left", wraplength=890)
    explanation4_lbl.grid(row=5, column=0, pady=5, sticky="nsew", columnspan=2, padx=15)
    
    # Github logo 
    github_path = get_resource_path("Resources/github_PNG.png")
    logo_github_img = Image.open(github_path)
    logo_github = ctk.CTkImage(dark_image=logo_github_img, size=(216, 80))

    # link to the Github Project
    github_link = ctk.CTkLabel(master=principal_frame, text="Codigo del proyecto", font=("Arial", 16, "bold"), text_color="#fbe122", cursor="hand2", image=logo_github, compound="right")
    github_url = "Github.com"
    github_link.bind("<Button-1>", lambda e: open_link(link = github_url))
    github_link.grid(row=6, column=0, pady=20, sticky="nsew", padx=15)

    # Documentation logo
    doc_path = get_resource_path("Resources/doc_logo.png")
    logo_doc_img = Image.open(doc_path)
    logo_doc = ctk.CTkImage(dark_image=logo_doc_img, size=(80, 80))

    # link to the Documentation
    doc_link = ctk.CTkLabel(master=principal_frame, text="IEEE del proyecto", font=("Arial", 16, "bold"), text_color="#fbe122", cursor="hand2", image=logo_doc, compound="right")
    doc_url = "onedrive.live.com"
    doc_link.bind("<Button-1>", lambda e: open_link(link = doc_url))
    doc_link.grid(row=6, column=1, pady=20, sticky="nsew", padx=15)

def excersice_frame_creation(master=None, num_excercise=0):

    global content_frame

    print(f"Excersice en excersice frame creation {num_excercise}")

    content_frame = ctk.CTkScrollableFrame(master=master, corner_radius=0, fg_color="#11371A", scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    content_frame.grid(row=0, column=2, sticky="nsew", columnspan=12, rowspan=12)
    content_frame = grid_setup(content_frame)
    # Add the Explication Section
    row = 0
    if num_excercise != 0:
        explanation_frame = explanation_frame_creation(content_frame, num_excercise)
        row = 6
        # Buttons for train and test the model
        # Train button

        train_btn = ctk.CTkButton(master=content_frame, text="Entrenamiento", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
        train_btn.grid(row=row, column=0, pady=10, sticky="n")
        train_btn.configure(command= lambda: train_frame_creation(master=content_frame, num_excercise=num_excercise))

        test_btn = ctk.CTkButton(master=content_frame, text="Pruebas", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
        test_btn.grid(row=row, column=1, pady=10, sticky="n")
        test_btn.configure(command= lambda: test_frame_creation(master=master, number_excercise=num_excercise))

    else:
        train_frame_creation(master=content_frame, num_excercise=num_excercise, row=row)

def train_frame_creation(master = None, num_excercise = 0, row = 8):

    global data_train_json, download_weights_btn, download_training_data_btn, results_btn

    print(f"Excersice number in train frame creation {num_excercise}")

    train_frame = None
    train_frame = ctk.CTkFrame(master=master, corner_radius=0, fg_color="#11371A")
    train_frame.grid(row=row, column=0, sticky="nsew", columnspan=12, rowspan=4, padx=10, pady=5)
    train_frame = grid_setup(train_frame)

    # region Inputs, Comboboxes and Checkboxes for data training
    # Checkbox for use the last training data
    
    last_training = ctk.CTkCheckBox(master=train_frame, text="Usar datos del entrenamiento por defecto", font=("Arial", 16), text_color="#fbe122")
    last_training.grid(row=0, column=0, pady=10, sticky="w", columnspan=4)

    # Checkbox for momentun
    momentum = ctk.CTkCheckBox(master=train_frame, text="Momentum", font=("Arial", 16), text_color="#fbe122")
    momentum.grid(row=0, column=4, pady=10, sticky="w", columnspan=2)

    # Inputs for quantity of neurons in the hidden layer
    # Hidden Layer
    hidden_layer_txt = "Cantidad de Neuronas en Capa Oculta (H):"
    hidden_layer_lbl = ctk.CTkLabel(master=train_frame, text=hidden_layer_txt, font=("Arial", 16, "bold"), text_color="#fbe122")
    hidden_layer_lbl.grid(row=1, column=0, pady=10, sticky="w", columnspan=3)

    hidden_layer_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="3", placeholder_text_color= "#8d918e")
    hidden_layer_entry.grid(row=1, column=3, pady=10, sticky="w", columnspan=2)

    # Input for max epochs
    max_epochs_txt = "Número Máximo de Épocas:"
    max_epochs_lbl = ctk.CTkLabel(master=train_frame, text=max_epochs_txt, font=("Arial", 16, "bold"), text_color="#fbe122")
    max_epochs_lbl.grid(row=1, column=5, pady=10, sticky="w", columnspan=2)

    max_epochs_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="1000", placeholder_text_color= "#8d918e")
    max_epochs_entry.grid(row=1, column=7, pady=10, sticky="w", columnspan=2)

    # Add Bias, Alpha, Betha and Precision inputs
    # Bias
    bias_txt = "Bias (θ):"
    bias_lbl = ctk.CTkLabel(master=train_frame, text=bias_txt, font=("Arial", 16, "bold"), text_color="#fbe122")
    bias_lbl.grid(row=2, column=0, pady=10, sticky="w")

    bias_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=120, placeholder_text="0 es Random", placeholder_text_color= "#8d918e")
    bias_entry.grid(row=2, column=1, pady=10, sticky="w", padx=2)

    # Alpha
    alpha_txt = "Alpha (α):"
    alpha_lbl = ctk.CTkLabel(master=train_frame, text=alpha_txt, font=("Arial", 16, "bold"), text_color="#fbe122")
    alpha_lbl.grid(row=2, column=2, pady=10, sticky="w")

    alpha_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="0.005", placeholder_text_color= "#8d918e")
    alpha_entry.grid(row=2, column=3, pady=10, sticky="w", padx=2)

    # Betha
    betha_txt = "Betha (β):"
    betha_lbl = ctk.CTkLabel(master=train_frame, text=betha_txt, font=("Arial", 16, "bold"), text_color="#fbe122")
    betha_lbl.grid(row=2, column=4, pady=10, sticky="w")

    betha_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="0.3", placeholder_text_color= "#8d918e")
    betha_entry.grid(row=2, column=5, pady=10, sticky="w", padx=2)

    # Precision
    precision_txt = "Precision (ρ):"
    precision_lbl = ctk.CTkLabel(master=train_frame, text=precision_txt, font=("Arial", 16, "bold"), text_color="#fbe122")
    precision_lbl.grid(row=2, column=6, pady=10, sticky="w")

    precision_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="0.0001", placeholder_text_color= "#8d918e")
    precision_entry.grid(row=2, column=7, pady=10, sticky="w", padx=2)

    # Combobox for the function to use in H layer
    function_txt = "Función de Activación Capa Oculta (H):"
    function_lbl = ctk.CTkLabel(master=train_frame, text=function_txt, font=("Arial", 16, "bold"), text_color="#fbe122")
    function_lbl.grid(row=3, column=0, pady=10, sticky="w", columnspan=3)

    layer_h_options = ["Tangente Hiperbólica", "Sigmoidal", "Softplus"]
    layer_h_combobox = ctk.CTkComboBox(master=train_frame, values=layer_h_options, font=("Arial", 16), width=200, border_width=0)
    layer_h_combobox.grid(row=3, column=3, pady=10, sticky="w", columnspan=3)

    # Combobox for the function to use in O layer
    function_txt2 = "Función de Activación Capa Salida (O):"
    function_lbl2 = ctk.CTkLabel(master=train_frame, text=function_txt2, font=("Arial", 16, "bold"), text_color="#fbe122")
    function_lbl2.grid(row=4, column=0, pady=10, sticky="w", columnspan=3)

    layer_o_options = ["Tangente Hiperbólica", "Sigmoidal", "Softplus", "ReLU", "Leaky ReLU", "Lineal"]
    layer_o_combobox = ctk.CTkComboBox(master=train_frame, values=layer_o_options, font=("Arial", 16), width=200, border_width=0)
    layer_o_combobox.grid(row=4, column=3, pady=10, sticky="w", columnspan=3)
    # endregion

    # Buttons for dowlnoad the template and load the data
    # Download template button
    download_btn = ctk.CTkButton(master=train_frame, text="Descargar Plantilla", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=lambda: download_json(f"base_{num_excercise}_template",directory=f"Data/case_{num_excercise}"))
    download_btn.grid(row=5, column=0, pady=10, sticky="n", columnspan=2)

    # Load data button
    load_data_btn = ctk.CTkButton(master=train_frame, text="Cargar Datos", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    load_data_btn.grid(row=5, column=2, pady=10, sticky="n", columnspan=2)

    # Status label
    status_lbl = ctk.CTkLabel(master=train_frame, text="No Cargado", font=("Arial", 16, "bold"), text_color="red")
    status_lbl.grid(row=5, column=4, pady=10, sticky="n", columnspan=2)

    last_training.configure(command=lambda: chargue_last_training_data(train_frame, status_lbl, bias_entry, alpha_entry, betha_entry, precision_entry, layer_h_combobox, layer_o_combobox, num_excercise, momentum, hidden_layer_entry, max_epochs_entry))
    load_data_btn.configure(command=lambda: chargue_data_training(train_frame, status_lbl))

    # Status of the training start
    status_lbl2 = ctk.CTkLabel(master=train_frame, text="", font=("Arial", 16, "bold"), text_color="red")
    status_lbl2.grid(row=8, column=2, pady=10, sticky="n", columnspan=2)

    # Train button
    train_btn = ctk.CTkButton(master=train_frame, text="Iniciar Entrenamiento", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    train_btn.grid(row=8, column=0, pady=10, sticky="n", columnspan=2)
    train_btn.configure(command= lambda: start_training(status_lbl, status_lbl2,bias_entry, alpha_entry, betha_entry, precision_entry, layer_h_combobox, layer_o_combobox, num_excercise, momentum, hidden_layer_entry, max_epochs_entry))

    # Stop training button
    stop_btn = ctk.CTkButton(master=train_frame, text="Detener Entrenamiento", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    stop_btn.grid(row=8, column=4, pady=10, sticky="n", columnspan=2)
    stop_btn.configure(command= lambda: set_stop_training(True))

    # Download weights button
    download_weights_btn = ctk.CTkButton(master=train_frame, text="Descargar Pesos", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", state="disabled")
    download_weights_btn.grid(row=10, column=0, pady=10, sticky="n", columnspan=2)
    download_weights_btn.configure(command= lambda: download_weights(num_excercise))
    
    # Results buton -> Command = section => Frame with weights json info and graphs
    results_btn = ctk.CTkButton(master=train_frame, text="Resultados", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", state="disabled")
    results_btn.grid(row=10, column=2, pady=10, sticky="n", columnspan=2)
    results_btn.configure(command= lambda: results_frame_creation(train_frame))
    
    # Downoload training data button
    download_training_data_btn = ctk.CTkButton(master=train_frame, text="Descargar Datos de Entrenamiento", fg_color="#fbe122", width=240, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", state="disabled")
    download_training_data_btn.grid(row=10, column=4, pady=10, sticky="n", columnspan=4)
    download_training_data_btn.configure(command= lambda: download_training_data(num_excercise))

def changue_status_training(status_label, text, color, download_weights_btn, download_training_data_btn, results_btn):
    status_label.configure(text=text, text_color=color)
    
    if text == "Entrenamiento finalizado":
        download_weights_btn.configure(state="normal")
        download_training_data_btn.configure(state="normal")
        results_btn.configure(state="normal")
    else:
        download_weights_btn.configure(state="disabled")
        download_training_data_btn.configure(state="disabled")
        results_btn.configure(state="disabled")

def chargue_last_training_data(master, status, bias_entry, alpha_entry, betha_entry, precision_entry, layer_h_combobox, layer_o_combobox, excercise_number, momentum, hidden_layer_entry, max_epochs_entry):
    """
    Function to chargue the last training data in the entries and comboboxes

    Parameters:
    bias_entry: Entry for the bias
    alpha_entry: Entry for the alpha
    betha_entry: Entry for the betha
    precision_entry: Entry for the precision
    layer_h_combobox: Combobox for the function in the hidden layer
    layer_o_combobox: Combobox for the function in the output layer
    excercise_number: Number of the excercise
    momentum: Checkbox for the momentum (Selected = True, Deselected = False)
    hidden_layer_entry: Entry for the quantity of neurons in the hidden layer
    max_epochs_entry: Entry for the max epochs
    """
    global data_train_json

    # Get the last training data from the json file
    last_training_data = load_json(f"base_{excercise_number}_training", directory=f"Data/case_{excercise_number}")

    # Set the values in the entries
    bias_entry.delete(0, END)
    bias_entry.insert(0, str(last_training_data["initial_bias"]))

    alpha_entry.delete(0, END)
    alpha_entry.insert(0, str(last_training_data["alpha"]))

    betha_entry.delete(0, END)
    betha_entry.insert(0, str(last_training_data["b"]))

    precision_entry.delete(0, END)
    precision_entry.insert(0, str(last_training_data["precision"]))

    hidden_layer_entry.delete(0, END)
    hidden_layer_entry.insert(0, str(last_training_data["qty_neurons"]))

    max_epochs_entry.delete(0, END)
    max_epochs_entry.insert(0, str(last_training_data["max_epochs"]))

    # Set the values in the comboboxes
    option_h = last_training_data["function_h"]
    if option_h == "tanh":
        layer_h_combobox.set("Tangente Hiperbólica")
    elif option_h == "sigmoid":
        layer_h_combobox.set("Sigmoidal")
    elif option_h == "Softplus":
        layer_h_combobox.set("Softplus")
    
    option_o = last_training_data["function_o"]
    if option_o == "tanh":
        layer_o_combobox.set("Tangente Hiperbólica")
    elif option_o == "sigmoid":
        layer_o_combobox.set("Sigmoidal")
    else:
        layer_o_combobox.set(option_o)
    
    # Set the value of the momentum
    momentum.select() if last_training_data["momentum"] else momentum.deselect()

    data_train_json = load_json(f"base_{excercise_number}_template", directory=f"Data/case_{excercise_number}")
    data_frame = create_data_frame(master, data_train_json)

    status.configure(text="Cargado", text_color="green")

def chargue_data_training(master, status):
    global data_train_json
    # load json data
    data_train_json, filename = load_json()
 
    # frame to show info of the data
    data_frame = create_data_frame(master, data_train_json)

    status.configure(text="Cargado", text_color="green")

def create_data_frame(master, data_json, row=7, is_training=True):
    global  epoch_label, total_error_label, labels_error
    epoch_label, total_error_label, labels_error = [], [], []
    data_frame = ctk.CTkScrollableFrame(master=master, corner_radius=8, fg_color="#ffffff", height=400)
    data_frame.grid(row=row, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    data_frame = grid_setup(data_frame)

    # Title of the data
    title_txt = "Datos de Entrenamiento"
    title = ctk.CTkLabel(master=data_frame, text=title_txt, font=("Arial", 16, "bold"), text_color="#11371a", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=5, sticky="new", columnspan=12)

    # Inputs
    inputs_txt = "Entradas: "
    inputs_lbl = ctk.CTkLabel(master=data_frame, text=inputs_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    inputs_lbl.grid(row=1, column=0, pady=5, sticky="new", columnspan=6)

    inputs = data_json["inputs"]
    inputs_txt2 = ""
    for i in range(len(inputs)):
        inputs_txt2 += f"Patrón {i+1}: {inputs[i]}\n"
    inputs_lbl2 = ctk.CTkLabel(master=data_frame, text=inputs_txt2, font=("Consolas", 16), text_color= "black" , justify="center", wraplength=400)
    inputs_lbl2.grid(row=2, column=0, pady=5, sticky="new", columnspan=6)

    # Outputs
    outputs_txt = "Salidas: "
    outputs_lbl = ctk.CTkLabel(master=data_frame, text=outputs_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    outputs_lbl.grid(row=1, column=6, pady=5, sticky="new", columnspan=6)

    outputs = data_json["outputs"]
    outputs_txt2 = ""
    for i in range(len(outputs)):
        outputs_txt2 += f"Patrón {i+1}: {outputs[i]}\n"
    outputs_lbl2 = ctk.CTkLabel(master=data_frame, text=outputs_txt2, font=("Consolas", 16), text_color= "black" , justify="center", wraplength=400)
    outputs_lbl2.grid(row=2, column=6, pady=5, sticky="new", columnspan=6)

    if is_training:
        # Create the training data process frame
        training_process_frame, epoch_label, total_error_label, labels_error = create_training_process_frame(master, len(inputs))

    return data_frame

def start_training(status, status2, bias_entry, alpha_entry, betha_entry, precision_entry, layer_h_combobox, layer_o_combobox, excercise_number, momentum, hidden_layer_entry, max_epochs_entry):
    global data_train_json, labels_error, epoch_label, total_error_label, download_weights_btn, download_training_data_btn, results_btn, num_excercise
    
    num_excercise = excercise_number
    
    set_stop_training(False)

    # region Values validation
    if data_train_json is None:
        status.configure(text="Error: No se han cargado los datos", text_color="red")
        return
    
    if bias_entry.get() == "":
        status.configure(text="Error: No se ha ingresado el Bias", text_color="red")
        return
    
    bias = float(bias_entry.get())
    if bias < -1 or bias > 1:
        status.configure(text="Error: El Bias debe ser mayor o igual entre -1 y 1 para mejor entrenamiento", text_color="red")
        return
    
    if alpha_entry.get() == "":
        status.configure(text="Error: No se ha ingresado el Alpha", text_color="red")
        return
    
    alpha = float(alpha_entry.get())
    if alpha > 1 or alpha <= 0:
        status.configure(text="Error: El Alpha debe ser mayor a 0 y menor a 1 Para evitar divergencia", text_color="red")
        return
    
    if betha_entry.get() == "":
        status.configure(text="Error: No se ha ingresado el Betha", text_color="red")
        return

    betha = float(betha_entry.get())
    if betha < 0.1 or betha > 0.9:
        status.configure(text="Error: El Betha debe ser mayor o igual a 0", text_color="red")
        return
    
    if precision_entry.get() == "":
        status.configure(text="Error: No se ha ingresado la Precisión", text_color="red")
        return
    
    precision = float(precision_entry.get())
    if precision <= 0:
        status.configure(text="Error: La Precisión debe ser mayor a 0", text_color="red")
        return
    
    if hidden_layer_entry.get() == "":
        status.configure(text="Error: No se ha ingresado la cantidad de neuronas en la capa oculta", text_color="red")
        return
    
    qty_neurons = int(hidden_layer_entry.get())
    if qty_neurons < 1:
        status.configure(text="Error: La cantidad de neuronas en la capa oculta debe ser mayor a 0", text_color="red")
        return
    
    if max_epochs_entry.get() == "":
        status.configure(text="Error: No se ha ingresado el número máximo de épocas", text_color="red")
        return
    
    max_epochs = int(max_epochs_entry.get())
    if max_epochs < 100:
        status.configure(text="Error: El número máximo de épocas debe ser mayor a 100", text_color="red")
        return
    
    function_h = layer_h_combobox.get()
    function_o = layer_o_combobox.get()

    momentum_value = momentum.get()
    m = True if momentum_value == 1 else False
    # endregion

    # Start the training process
    inputs = data_train_json["inputs"]
    outputs = data_train_json["outputs"]
    train_data = {
        "inputs": inputs,
        "outputs": outputs,
        "alpha": alpha,
        "betha": betha,
        "bias": bias,
        "precision": precision,
        "function_h_name": function_h,
        "function_o_name": function_o,
        "momentum": m,
        "qty_neurons": qty_neurons,
        "max_epochs": max_epochs,
    }

    print("Train Data")
    print(json.dumps(train_data, indent=2))

    # Start the training process
    training_thread = threading.Thread(target=backpropagation_training, args=(train_data,epoch_label, total_error_label, labels_error, status2, download_weights_btn, download_training_data_btn, results_btn))
    training_thread.start()

    changue_status_training(status2, "Entrenando...", "orange", download_weights_btn, download_training_data_btn, results_btn)

def create_training_process_frame(master, num_patters):
    global labels_error, epoch_label, total_error_label

    training_frame = ctk.CTkFrame(master=master, corner_radius=8, fg_color="#ffffff")
    training_frame.grid(row=9, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    training_frame = grid_setup(training_frame)

    # Title of the training process
    title_txt = "Proceso de Entrenamiento"
    title = ctk.CTkLabel(master=training_frame, text=title_txt, font=("Arial", 16, "bold"), text_color="#11371a", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=5, sticky="new", columnspan=12)

    # Label for the epoch
    epoch_txt = "Época: "
    epoch_lbl = ctk.CTkLabel(master=training_frame, text=epoch_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    epoch_lbl.grid(row=1, column=0, pady=5, sticky="new", columnspan=2)

    epoch_label = ctk.CTkLabel(master=training_frame, text="0", font=("Arial", 16), text_color="#11371a", justify="center", anchor="center")
    epoch_label.grid(row=1, column=2, pady=5, sticky="new", columnspan=2)

    # Label for the total error 
    total_error_txt = "Error Total: "
    total_error_lbl = ctk.CTkLabel(master=training_frame, text=total_error_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    total_error_lbl.grid(row=1, column=4, pady=5, sticky="new", columnspan=2)

    text_total_error = "∞"
    total_error_label = ctk.CTkLabel(master=training_frame, text=text_total_error, font=("Arial", 16), text_color="#11371a", justify="center", anchor="center")
    total_error_label.grid(row=1, column=6, pady=5, sticky="new", columnspan=2)

    # for to put the labels of the error, in the row there are four => Patterns number |E|: Error
    # if the error is lower than the precision, the text color will be green, else red
    row = 2
    col = 0
    for i in range(num_patters):
        pattern_txt = f"Patrón {i+1} |E|: "
        pattern_lbl = ctk.CTkLabel(master=training_frame, text=pattern_txt, font=("Consolas", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
        pattern_lbl.grid(row=row, column=col, pady=5, sticky="new")

        col += 1
        # put the error of the pattern with 10 decimals
        text_error = "∞"
        pattern_lbl2 = ctk.CTkLabel(master=training_frame, text=text_error, font=("Consolas", 16), text_color="red", justify="center", anchor="center")
        pattern_lbl2.grid(row=row, column=col, pady=5, sticky="new", columnspan=2)

        labels_error.append(pattern_lbl2)

        col += 2

        if (i+1) % 3 == 0:
            row += 1
            col = 0

    return training_frame, epoch_label, total_error_label, labels_error

def update_training_process(epoch, errores_patrones, total_error, precision, epoch_label, total_error_label, labels_error):

    epoch_label.configure(text=str(epoch))

    total_error_label.configure(text=f"{total_error:.10f}")

    for i in range(len(errores_patrones)):
        text_color = "green" if errores_patrones[i] <= precision else "red"
        labels_error[i].configure(text=f"{errores_patrones[i]:.10f}", text_color=text_color)

def download_weights(num_excercise):
    weights_json =  get_weights_json()
    download_json(filename= f"Resultados_ejercicio_{num_excercise}", data= weights_json)

def download_training_data(num_excercise):
    data_train_json = get_graph_json()
    download_json(filename= f"Entrenamiento_ejercicio_{num_excercise}", data= data_train_json)

def results_frame_creation(master, is_train = True, weights_json = None, test_data = None):
    global data_train_json, num_excercises
    inputs, outputs = None, None
    
    print(f"Excersice results frame creation {num_excercise}")
    
    weights_json =  get_weights_json() if weights_json is None else weights_json
    results_frame = ctk.CTkFrame(master=master, corner_radius=8, fg_color="#ffffff")
    results_frame.grid(row=11, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    results_frame = grid_setup(results_frame)

    row = 0
    if is_train:
        graph_json = get_graph_json()
        row = add_weights_info(results_frame, weights_json, row, title_txt="Resultados del Entrenamiento")
    
        # Horizontal separator
        ctk.CTkFrame(master=results_frame, corner_radius=0, fg_color="#11371a", height=2).grid(row=row, column=0, pady=5, sticky="nsew", columnspan=12)
        row += 1

        row = add_graph_info(results_frame, graph_json, row)

        # Horizontal separator
        ctk.CTkFrame(master=results_frame, corner_radius=0, fg_color="#11371a", height=2).grid(row=row, column=0, pady=5, sticky="nsew", columnspan=12)
        row += 1

    if is_train:
        inputs = data_train_json["inputs"]
        outputs = data_train_json["outputs"]
    else:
        inputs = test_data["inputs"]
        outputs = test_data["outputs"]

    # Test Results Section
    test_data = {
        "inputs": inputs,
        "outputs": outputs,
        "weights_h": weights_json["weights_h"],
        "weights_o": weights_json["weights_o"],
        "bias_h": weights_json["bias_h"],
        "bias_o": weights_json["bias_o"],
        "qty_neurons": weights_json["qty_neurons"],
        "function_h_name": weights_json["function_h_name"],
        "function_o_name": weights_json["function_o_name"]
    }

    add_results_info(results_frame, test_data, row, num_excercise)

def add_weights_info(frame, weights_json, row, title_txt):
    # Title of the results
    title = ctk.CTkLabel(master=frame, text=title_txt, font=("Arial", 16, "bold"), text_color="#11371a", anchor="center", justify="center")
    title.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1
    # Weights H layer label
    weights_h_txt = "Pesos de la Capa Oculta (H):"
    weights_h_lbl = ctk.CTkLabel(master=frame, text=weights_h_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    weights_h_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1
    # For to put the weights of the H layer, Each label will have the weights of a neuron
    weights_h = weights_json["weights_h"]
    col = 0
    colspan = 4
    for j in range(len(weights_h)):
        
        weights_txt = f"J {j+1}:\n "
        for i in range(len(weights_h[j])):
            weights_txt += f"W[{j+1}][{i+1}]{weights_h[j][i]:.10f}\n"
        
        if (len(weights_h) == 1) or (j == len(weights_h) - 1 and len(weights_h) % 3 == 1):
            colspan = 12    
    
        weights_lbl = ctk.CTkLabel(master=frame, text=weights_txt, font=("Consolas", 16), text_color="#11371a", justify="center", wraplength=840)
        weights_lbl.grid(row=row, column=col, pady=5, sticky="new", columnspan=colspan)
        if (j+1) % 3 == 0:
            col = 0
            row += 1
        else:
            col += 4
    row += 1
    # Bias H layer label
    bias_h_txt = "Bias de la Capa Oculta (H):"
    bias_h_lbl = ctk.CTkLabel(master=frame, text=bias_h_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    bias_h_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    # For to put the bias of the H layer, Each label will have the bias of a neuron
    bias_h = weights_json["bias_h"]
    for i in range(len(bias_h)):
        bias_txt = f"J {i+1}: {bias_h[i]:.10f}"
        bias_lbl = ctk.CTkLabel(master=frame, text=bias_txt, font=("Consolas", 16), text_color="#11371a", justify="center", wraplength=840)
        bias_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
        row += 1
    
    # Horizontal separator
    ctk.CTkFrame(master=frame, corner_radius=0, fg_color="#11371a", height=2).grid(row=row, column=0, pady=5, sticky="nsew", columnspan=12)
    row += 1

    # Weights O layer label
    weights_o_txt = "Pesos de la Capa de Salida (O):"
    weights_o_lbl = ctk.CTkLabel(master=frame, text=weights_o_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    weights_o_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    # For to put the weights of the O layer, Each label will have the weights of a neuron
    weights_o = weights_json["weights_o"]
    col = 0
    colspan = 4
    for j in range(len(weights_o)):
        
        weights_txt = f"K {j+1}:\n"
        for i in range(len(weights_o[j])):
            weights_txt += f"W[{j+1}][{i+1}]{weights_o[j][i]:.10f}\n"
            
        if (len(weights_o) == 1) or (j == len(weights_o) - 1 and len(weights_o) % 3 == 1):
            colspan = 12  
            
        weights_lbl = ctk.CTkLabel(master=frame, text=weights_txt, font=("Consolas", 16), text_color="#11371a", justify="center", wraplength=840)
        weights_lbl.grid(row=row, column=col, pady=5, sticky="new", columnspan=colspan)
        if (j+1) % 3 == 0:
            col = 0
            row += 1
        else:
            col += 4
    row += 1
    # Bias O layer label
    bias_o_txt = "Bias de la Capa de Salida (O):"
    bias_o_lbl = ctk.CTkLabel(master=frame, text=bias_o_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    bias_o_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    # For to put the bias of the O layer, Each label will have the bias of a neuron
    bias_o = weights_json["bias_o"]
    for i in range(len(bias_o)):
        bias_txt = f"K {i+1}: {bias_o[i]:.10f}"
        bias_lbl = ctk.CTkLabel(master=frame, text=bias_txt, font=("Consolas", 16), text_color="#11371a", justify="center", wraplength=840)
        bias_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
        row += 1
    
    return row

def add_graph_info(frame, graph_json, row):
     # Labels for max_epochs, cuantity neurons, function h name, function o name,
    # betha if momentum is selected like "Momentum: Si con Betha = {value}" otherwise "Momentum: No"
    
    max_epochs_txt = f"Número Máximo de Épocas: {graph_json['max_epochs']}"
    max_epochs_lbl = ctk.CTkLabel(master=frame, text=max_epochs_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    max_epochs_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    qty_neurons_txt = f"Cantidad de Neuronas en Capa Oculta (H): {graph_json['qty_neurons']}"
    qty_neurons_lbl = ctk.CTkLabel(master=frame, text=qty_neurons_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    qty_neurons_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    function_h_txt = f"Función de Activación Capa Oculta (H): {graph_json['function_h']}"
    function_h_lbl = ctk.CTkLabel(master=frame, text=function_h_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    function_h_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    function_o_txt = f"Función de Activación Capa de Salida (O): {graph_json['function_o']}"
    function_o_lbl = ctk.CTkLabel(master=frame, text=function_o_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    function_o_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    momentum_txt = f"Momentum: {'Si' if graph_json['momentum'] else 'No'}"
    if graph_json['momentum']:
        momentum_txt += f" con Betha = {graph_json['b']}"
    
    momentum_lbl = ctk.CTkLabel(master=frame, text=momentum_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    momentum_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    # Date of the training
    date_txt = f"Fecha del Entrenamiento: {graph_json['training_date']}"
    date_lbl = ctk.CTkLabel(master=frame, text=date_txt, font=("Arial", 16, "bold"), text_color="#11371a", justify="center", anchor="center")
    date_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    # Graphs
    # Error graph
    error_graph = graph_json["totals_errors"]
    figure_errors = plot_error_total_by_epoch(error_graph, last_epoch=graph_json["epochs"])
    canvas_errors = FigureCanvasTkAgg(figure_errors, master=frame)
    canvas_errors.draw()
    canvas_errors.get_tk_widget().grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    # Arquitecture graph
    arquitecture_graph = graph_json["arquitecture"]
    labels = max(arquitecture_graph) < 5
    figure_arquitecture = plot_neural_network_with_labels(arquitecture_graph, labels)
    canvas_arquitecture = FigureCanvasTkAgg(figure_arquitecture, master=frame)
    canvas_arquitecture.draw()
    canvas_arquitecture.get_tk_widget().grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    return row

def add_results_info(frame, test_data, row, num_excercise=0):

    results, errors = test_neural_network(test_data)

    print(f"Excersice number in add_results_info {num_excercise}")

    # Title of the test results
    title_test_txt = "Resultados de la Prueba"
    title_test = ctk.CTkLabel(master=frame, text=title_test_txt, font=("Arial", 16, "bold"), text_color="#11371a", anchor="center", justify="center")
    title_test.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    # For each input the label will show # patter, input, expected output, output, error (values separated by \n)
    for i in range(len(results)):

        # Horizontal separator
        ctk.CTkFrame(master=frame, corner_radius=0, fg_color="#11371a", height=2).grid(row=row, column=0, pady=5, sticky="nsew", columnspan=12)
        row += 1

        test_txt = f"Patrón {i+1}:\n"
        test_lbl = ctk.CTkLabel(master=frame, text=test_txt, font=("Consolas", 16), text_color="#11371a", justify="center", wraplength=840)
        test_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
        row += 1

        input = test_data["inputs"][i]
        output = test_data["outputs"][i]
        error = errors[i]
        output_char = []
        result_char = []
        # Case 4 the output needs to be convert with char function to show the correct value
        if num_excercise == 4:
            for j in range(len(output)):
                output_char.append(chr(round(output[j])))
                result_char.append(chr(round(results[i][j])))
        
        y_obtained = ""
        for j in range(len(results[i])):
            y_obtained += f"{results[i][j]:.10f}\n"

        if num_excercise != 4:
            test_txt2 = f"Entradas:\n {input}\nSalidas Esperadas:\n {output}\nSalidas Obtenidas:\n {y_obtained}\nError:\n {error}"
        else:
            test_txt2 = f"Entradas:\n {input}\nSalidas Esperadas:\n {output} = {output_char}\nSalidas Obtenidas:\n {y_obtained} = {result_char} \nError:\n {error}"
        
        test_lbl2 = ctk.CTkLabel(master=frame, text=test_txt2, font=("Consolas", 16), text_color="#11371a", justify="center", wraplength=840)
        test_lbl2.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
        row += 1

    return row

def test_frame_creation(master, number_excercise=0):
    global data_test_json, weights_test_json, num_excercise

    num_excercise = number_excercise

    test_frame = ctk.CTkScrollableFrame(master=master, corner_radius=0, fg_color="#11371A",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    test_frame.grid(row=0, column=2, sticky="nsew", columnspan=12, rowspan=12)
    test_frame = grid_setup(test_frame)

    # Title of the test
    title_txt = "Sección de Prueba"
    title = ctk.CTkLabel(master=test_frame, text=title_txt, font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)

    # checkbox for data test defect
    defect_test = ctk.CTkCheckBox(master=test_frame, text="Usar Datos de Prueba por Defecto", font=("Arial", 16, "bold"), text_color="#fbe122")
    defect_test.grid(row=1, column=0, pady=5, padx=15 , sticky="new", columnspan=12)

    # Load data button
    load_data_btn = ctk.CTkButton(master=test_frame, text="Cargar Entradas y Salidas", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    load_data_btn.grid(row=2, column=0, pady=10, sticky="n", columnspan=2)

    # Status label
    status_lbl = ctk.CTkLabel(master=test_frame, text="No Cargado", font=("Arial", 16, "bold"), text_color="red")
    status_lbl.grid(row=2, column=2, pady=10, sticky="n", columnspan=3)

    # Load data button
    load_weights_btn = ctk.CTkButton(master=test_frame, text="Cargar Pesos", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    load_weights_btn.grid(row=2, column=6, pady=10, sticky="n", columnspan=2)

    # Status label
    status_lbl2 = ctk.CTkLabel(master=test_frame, text="No Cargado", font=("Arial", 16, "bold"), text_color="red")
    status_lbl2.grid(row=2, column=8, pady=10, sticky="n", columnspan=3)

    # Test button
    test_btn = ctk.CTkButton(master=test_frame, text="Iniciar Prueba", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", state="disabled")
    test_btn.grid(row=5, column=0, pady=10, sticky="n", columnspan=2)


    defect_test.configure(command= lambda: chargue_default_data_test(test_frame, num_excercise, status_lbl, status_lbl2, test_btn))
    load_data_btn.configure(command= lambda: chargue_data_test(test_frame, status_lbl, status_lbl2, test_btn))
    load_weights_btn.configure(command= lambda: chargue_weights_test(test_frame, status_lbl, status_lbl2, test_btn))
    test_btn.configure(command= lambda: results_frame_creation(test_frame, is_train=False, weights_json=weights_test_json, test_data=data_test_json))    

def chargue_default_data_test(frame, num_excercise, status, status2, test_btn):
    global data_test_json, weights_test_json

    # load json data
    data_test_json = load_json(filename=f"base_{num_excercise}_template", directory=f"Data/case_{num_excercise}")
 
    # frame to show info of the data
    data_frame = create_data_frame(frame, data_test_json, row=3, is_training = False)

    if data_frame is None:
        status.configure(text="Error al cargar los datos", text_color="red")
        return
    
    status.configure(text="Datos Cargados", text_color="green")

    weights_test_json = load_json(filename=f"base_{num_excercise}_weights", directory=f"Data/case_{num_excercise}")

    # frame to show info of the weights
    weights_frame = create_weights_frame(frame, weights_test_json, row=4)

    if weights_frame is None:
        status2.configure(text="Error al cargar los pesos", text_color="red")
        return

    status2.configure(text="Pesos Cargados", text_color="green")

    test_btn.configure(state="normal")

def chargue_data_test(master, status, status2, test_btn):
    global data_test_json
    # load json data
    data_test_json, filename = load_json()
 
    # frame to show info of the data
    data_frame = create_data_frame(master, data_test_json, row=3, is_training = False)

    status.configure(text="Datos Cargados", text_color="green")

    if data_frame is None:
        status.configure(text="Error al cargar los datos", text_color="red")
        return
    
    if status2.cget("text") == "Pesos Cargados":
        test_btn.configure(state="normal")

def chargue_weights_test(master, status, status2, test_btn):
    global weights_test_json
    # load json data
    weights_test_json, filename = load_json()
 
    # frame to show info of the data
    data_frame = create_weights_frame(master, weights_test_json)
    status2.configure(text="Pesos Cargados", text_color="green")

    if data_frame is None:
        status2.configure(text="Error al cargar los pesos", text_color="red")
        return

    if status.cget("text") == "Datos Cargados":
        test_btn.configure(state="normal")

def create_weights_frame(frame, data_weights_json, row=4):
    weights_frame = ctk.CTkFrame(master=frame, corner_radius=8, fg_color="#ffffff")
    weights_frame.grid(row=row, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    weights_frame = grid_setup(weights_frame)

    # Title of the weights
    title_txt = "Pesos y Bias"
    row = add_weights_info(weights_frame, data_weights_json, 0, title_txt)

    return weights_frame

if __name__ == "__main__":
    GUI_creation()