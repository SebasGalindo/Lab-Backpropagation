# Description: This file contains the GUI creation and the functions to create the different frames of the application.
# Authors: John Sebastián Galindo Hernández, Miguel Ángel Moreno Beltrán

# region Import External libraries
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import END
import tkinter as tk
from PIL import Image
import json
import threading
import concurrent.futures
# endregion

# region Import Internal libraries
from utils import get_resource_path, open_link, download_json, load_json, save_json
from backpropagation import backpropagation_training, test_neural_network, set_stop_training, get_graph_json, get_weights_json
from graphics import plot_neural_network_with_labels, plot_error_total_by_epoch, plot_histograms
from image_treatment import load_images, get_kernel, apply_kernel, download_images, resize_image, plain_image, normalize_image_vector, get_histogram_data
# endregion

# region Variables
ctk.set_appearance_mode("dark")
# Global variables
main_window, content_frame, train_frame = None, None, None
download_weights_btn, download_training_data_btn, results_btn = None, None, None
data_inputs_text, errors_text, weight_text, results_text = None, None, None, None

# Data variables
data_train_json, data_test_json, weights_test_json = None, None, None
num_excercise = 0

# Image variables
is_folder_img = False
images, filtered_images = None, None
actual_first_image = None
images_labels, filtered_images_labels = None, None
txt_initial_train_images, txt_treated_train_images, txt_plained_train_images = None, None, None
images_train_status_lbl = None
next_images_btn, before_images_btn,apply_kernel_btn, remove_kernel_btn, image_treatment_btn_2 = None, None, None, None, None
download_images_btn, download_kernel_btn, resize_images_btn, plained_images_btn, finish_data_btn = None, None, None, None, None
pre_treated_check = None
kernel_names = [    
    "3x3_Identity",
    "3x3_Gaussian Blur",
    "5x5_Gaussian Blur",
    "7x7_gaussian_blur",
    "3x3_Sobel Vertical",
    "5x5_Sobel Vertical",
    "3x3_Sobel Horizontal",
    "5x5_Sobel Horizontal",
    "3x3_Laplacian",
    "5x5_Laplacian",
    "3x3_Prewitt",
    "5x5_Prewitt",
    "3x3_Sharpen",
    "5x5_Sharpen",
    "3x3_Emboss",
    "5x5_Emboss",
    "3x3_Box Blur",
    "5x5_Box Blur",
    "3x3_High-Pass Filter",
    "5x5_High-Pass Filter",
    "3x3_Motion Blur",
    "5x5_Motion Blur",
    "3x3_Edge Detection (Roberts Cross)",
    "5x5_Edge Detection (Roberts Cross)",
    "3x3_Diagonal Edge Detection",
    "5x5_Diagonal Edge Detection",
    "3x3_Laplacian Sharpen",
    "5x5_Laplacian Sharpen",
    "3x3_Gabor Filter",
    "5x5_Gabor Filter",
    "7x7_Gabor Filter",
    "3x3_Edge Detection (Laplacian of Gaussian)",
    "3x3_Edge Detection (Laplacian of Gaussian) plus",
    "3x3_Edge Detection (Laplacian of Gaussian) plus2",
    "5x5_Edge Detection (Laplacian of Gaussian)",
    "3x3_Average Blur",
    "5x5_Average Blur",
    "3x3_Edge Enhancement",
    "3x3_Edge Enhancement less", 
    "5x5_Edge Enhancement",
    "3x3_Outline Filter",
    "5x5_Outline Filter",
    "3x3_Ridge Detection",
    "5x5_Ridge Detection",
    "3x3_Gaussian Sharpen",
    "5x5_Gaussian Sharpen",
    "3x3_High Boost Filter",
    "5x5_High Boost Filter",
    "3x3_Edge Detection (Kirsch)",
    "5x5_Edge Detection (Kirsch)",
    "3x3_Edge Detection (Robinson)",
    "5x5_Edge Detection (Robinson)", 
    "rgb_to_bgr", 
    "bgr_to_rgb", 
    "gray_to_rgb",
    "gray_to_bgr", 
    "rgb_to_gray", 
    "bgr_to_gray", 
    "less_red", 
    "less_green", 
    "less_blue", 
    "more_red",
    "more_green", 
    "more_blue",
    ]
kernel_history, kernel_data = [], {}
kernel_series = ["Pez Cirujano","Trucha Arcoíris"]
selected_kernel_series = None
actual_tags = []
images_history = []
train_images_data, label_categories = [], None
chargued_tags_frame = None
info_window = None
image_training_frame = None

# Test images variables
image_status_lbl, txt_categories = None, None
image_frame, treated_image_frame, categories_frame = None, None, None
test_image, test_filtered_image = None, None
test_categories, default_test_categories = [], ["Pez Cirujano","Trucha Arcoíris" ]
weights_status_lbl, load_weights_btn, result_lbl = None, None, None
width_entry, height_entry, interpolation_combobox = None, None, None

# GUI variables
GUI_ELEMENTS = {
    "colors": {
        "background": "#003a22",
        "foreground": "#E8B817",
        "hover_button": "#BA9312",
        "info_button": "#0364b8",
        "info_button_hover": "#033663",
    },
    "fonts": { # Fonts are initialized in GUI_creation function
        "title_font": None,
        "subtitle_font": None,
        "text_font": None,
        "button_font": None,
    }
}
# endregion

# region main layout functions
def main_windowSetup():
    """
    Setup the main window of the application.
    """
    
    main_window = ctk.CTk()
    main_window.title("Inteligencia Artificial - Backpropagation")
    main_window.geometry("1200x700")
    main_window.after(0, lambda: main_window.focus_force())
    main_window.after(0, lambda: main_window.state("zoomed"))
    
    icon_path = get_resource_path("Resources/brand_logo.ico")
    main_window.iconbitmap(icon_path)
    main_window.lift()
    main_window.grid_columnconfigure(1, weight=0)
    main_window.grid_columnconfigure(2, weight=1) 
    for i in range(12):
        main_window.grid_rowconfigure(i, weight=1)
    return main_window

def sidebar_creation(master_window):
    """
    Create the sidebar of the application.
    
    Parameters:
    master_window (ctk.CTK): Main window of the application.
    
    Returns:
    ctk.CTKFrame: Sidebar of the application.
    """
    global GUI_ELEMENTS
    sidebar = ctk.CTkFrame(master=master_window, width=200, fg_color=GUI_ELEMENTS["colors"]["background"], corner_radius=0)
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
    excersice_one_btn = ctk.CTkButton(master=sidebar, text="Ejercicio Uno", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    excersice_one_btn.grid(row=4, column=0, pady=10, sticky="n")
    excersice_one_btn.configure(command= lambda: excersice_frame_creation(master_window, 1))

    excersice_two_btn = ctk.CTkButton(master=sidebar, text="Ejercicio Dos", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    excersice_two_btn.grid(row=5, column=0, pady=10, sticky="n")
    excersice_two_btn.configure(command= lambda: excersice_frame_creation(master_window, 2))

    excersice_three_btn = ctk.CTkButton(master=sidebar, text="Ejercicio Tres", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    excersice_three_btn.grid(row=6, column=0, pady=10, sticky="n")
    excersice_three_btn.configure(command= lambda: excersice_frame_creation(master_window, 3))

    excersice_four_btn = ctk.CTkButton(master=sidebar, text="Ejercicio Cuatro", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    excersice_four_btn.grid(row=7, column=0, pady=10, sticky="n")
    excersice_four_btn.configure(command= lambda: excersice_frame_creation(master_window, 4))

    train_btn = ctk.CTkButton(master=sidebar, text="Nuevo Entrenamiento (Matrices)", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    train_btn.grid(row=8, column=0, pady=10, sticky="n")
    train_btn.configure(command= lambda: excersice_frame_creation(master=master_window))

    test_solution_btn = ctk.CTkButton(master=sidebar, text="Probar Soluciones (Matrices)", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    test_solution_btn.grid(row=9, column=0, pady=10, sticky="n")
    test_solution_btn.configure(command = lambda: test_frame_creation(master= master_window, number_excercise = 0))

    img_treatment_btn = ctk.CTkButton(master=sidebar, text="Tratamiento de Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    img_treatment_btn.grid(row=10, column=0, pady=10, sticky="n")
    img_treatment_btn.configure(command = lambda: image_treatment_frame_creation(master_window))
    
    img_training_btn = ctk.CTkButton(master=sidebar, text="Entrenamiento con Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    img_training_btn.grid(row=11, column=0, pady=10, sticky="n")
    img_training_btn.configure(command = lambda: image_training_frame_creation(master_window))
    
    test_images_btn = ctk.CTkButton(master=sidebar, text="Probar Soluciones (Imágenes)", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    test_images_btn.grid(row=12, column=0, pady=10, sticky="n")
    test_images_btn.configure(command = lambda: test_image_frame_creation(master= master_window))
    
    return sidebar

def grid_setup(frame):
    """
    Setup the grid configuration of a frame.
    
    Parameters:
    frame (ctk.CTkFrame): Frame to setup the grid.
    
    Returns:
    ctk.CTkFrame: Frame with the grid configuration.
    """
    for i in range(12): 
        frame.grid_rowconfigure(i, weight=1)
        frame.grid_columnconfigure(i, weight=1)
    return frame

def GUI_creation():
    """
    Create the GUI of the application. 
    This function creates the main window, the sidebar and the initial frame and runs the main window.
    """

    global main_window, content_frame, GUI_ELEMENTS

    # Build the main window
    main_window = main_windowSetup()

    # Sidebar creation
    sidebar = sidebar_creation(main_window)

    # Vertical separator
    ctk.CTkFrame(master=main_window, width=2, fg_color="#0B2310").grid(row=0, column=1, sticky="ns")

    content_frame = initial_frame(main_window)

    GUI_ELEMENTS["fonts"]["title_font"] = ctk.CTkFont("Arial", 24, "bold")
    GUI_ELEMENTS["fonts"]["subtitle_font"] = ctk.CTkFont("Arial", 20, "bold")
    GUI_ELEMENTS["fonts"]["text_font"] = ctk.CTkFont("Consolas", 18)
    GUI_ELEMENTS["fonts"]["button_font"] = ctk.CTkFont("comfortaa", 14)

    # Run the main window
    main_window.mainloop()

def principal_frame_creation(master_window):
    """
    Create the principal frame of the application.
    
    Parameters:
    master_window (ctk.CTK): Main window of the application.
    
    Returns:
    ctk.CTkFrame: Principal frame of the application.
    """

    principal_frame = ctk.CTkScrollableFrame(master=master_window, corner_radius=0, fg_color=GUI_ELEMENTS["colors"]["background"],scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    principal_frame.grid(row=0, column=2, sticky="nsew", rowspan=12)
    principal_frame = grid_setup(principal_frame)

    return principal_frame

def initial_frame(event=None, master=None):
    """
    Create the initial frame of the application.
    This frame is created when the title is clicked and when the application is started.
    This frame contains the explanation of the application.
    
    Parameters:
    event (ctk.Event): Event that triggered the function.
    master (ctk.CTK): Main window of the application.
    """
    principal_frame = principal_frame_creation(master)

    # Create the section title
    title = ctk.CTkLabel(master=principal_frame, text="Sección de Explicación", font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title.grid(row=0, column=0, pady=20, sticky="nsew", columnspan=2, padx=15)

    # App Explanation section
    explanation_txt = ("Esta aplicación permite realizar entrenamiento con una red neuronal artifical con el modelo Backpropagation, "
                        "Existen 3 botones para ver los resutados de los ejercicios propuestos en clase."
                        "Por otro lado se cuenta con la funcionalidad de realizar un nuevo entrenamiento y probar soluciones con el modelo entrenado.")
    explanation_lbl = ctk.CTkLabel(master=principal_frame, text=explanation_txt, font=("Arial", 16), justify="left", wraplength=1200, text_color="#ffffff")
    explanation_lbl.grid(row=1, column=0, pady=5, sticky="nsew", columnspan=2, padx=15)

    # Training backpropagation explanation section
    # Create the section title
    title2 = ctk.CTkLabel(master=principal_frame, text="Sección de entrenamiento", font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
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
    explanation2_lbl = ctk.CTkLabel(master=principal_frame, text=explanation2_txt, font=("Arial", 16), justify="left", wraplength=1200)
    explanation2_lbl.grid(row=3, column=0, pady=5, sticky="nsew", columnspan=2, padx=15)

    # Test explanation section
    # Create the section title
    title4 = ctk.CTkLabel(master=principal_frame, text="Sección de Pruebas", font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title4.grid(row=4, column=0, pady=20, sticky="nsew", columnspan=2, padx=15)
    # explanation text
    explanation4_txt = ("La sección de Pruebas permite realizar pruebas con el modelo entrenado, "
                        "para ello, se deberá cargar dos archivos en formato JSON, "
                        "uno con las entradas y otro con los datos usados en el entrenamiento."
                        "La opcion por defecto es usar el boton que permite cargar el último entrenamiento realizado y que está alojaddo en la app."
                        "Una vez cargados los datos, se deberá presionar el botón 'Probar Soluciones' para comenzar el proceso."
                        "Finalmente, se mostrara una la lista de entradas, salidas deseadas, salidas obtenidas y error obtenido."
                        )
    explanation4_lbl = ctk.CTkLabel(master=principal_frame, text=explanation4_txt, font=("Arial", 16), justify="left", wraplength=1200)
    explanation4_lbl.grid(row=5, column=0, pady=5, sticky="nsew", columnspan=2, padx=15)
    
    # Github logo 
    github_path = get_resource_path("Resources/github_PNG.png")
    logo_github_img = Image.open(github_path)
    logo_github = ctk.CTkImage(dark_image=logo_github_img, size=(216, 80))

    # link to the Github Project
    github_link = ctk.CTkLabel(master=principal_frame, text="Codigo del proyecto", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], cursor="hand2", image=logo_github, compound="right")
    github_url = "https://github.com/SebasGalindo/Lab-Backpropagation"
    github_link.bind("<Button-1>", lambda e: open_link(link = github_url))
    github_link.grid(row=6, column=0, pady=20, sticky="nsew", padx=15)

    # Documentation logo
    doc_path = get_resource_path("Resources/doc_logo.png")
    logo_doc_img = Image.open(doc_path)
    logo_doc = ctk.CTkImage(dark_image=logo_doc_img, size=(80, 80))

    # link to the Documentation
    doc_link = ctk.CTkLabel(master=principal_frame, text="IEEE del proyecto", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], cursor="hand2", image=logo_doc, compound="right")
    doc_url = "https://mailunicundiedu-my.sharepoint.com/:f:/g/personal/mamorenobeltran_ucundinamarca_edu_co/ErQl-i2wSjdCrChwQEnBHwYBVXGC-gZ2jGlar40-mhWDoA?e=UM0jt7"
    doc_link.bind("<Button-1>", lambda e: open_link(link = doc_url))
    doc_link.grid(row=6, column=1, pady=20, sticky="nsew", padx=15)

    # Explanation of the new image treatment section
    # Create the section title
    title3 = ctk.CTkLabel(master=principal_frame, text="Secciónes de Imágenes", font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title3.grid(row=7, column=0, pady=20, sticky="nsew", columnspan=2, padx=15)
    # explanation text
    explanation3_txt = ("La sección de tratamiento de imágenes permite realizar el tratamiento de imágenes para su posterior uso en el entrenamiento de una red neuronal."
                        "Para ello, se deberá cargar un conjunto de imágenes en formato PNG o JPG, "
                        "una vez cargadas las imágenes, se deberá realizar el proceso como es explicado en la sección."
                        "Por otro lado, se tiene una sección de entrenamiento y pruebas con imágenes que también permite estan explicadas en la sección con el boton de ayuda.")
    explanation3_lbl = ctk.CTkLabel(master=principal_frame, text=explanation3_txt, font=("Arial", 16), justify="left", wraplength=1200)
    explanation3_lbl.grid(row=8, column=0, pady=5, sticky="nsew", columnspan=2, padx=15)
# endregion

# region Matrix Excercise functions
def explanation_frame_creation(master_window, num_excercise):
    """
    Create the explanation frame of the application.
    
    Parameters:
    master_window (ctk.CTK): Main window of the application.
    num_excercise (int): Number of the excercise.
    
    Returns:
    ctk.CTkFrame: Explanation frame of the application.
    """
    
    explanation_frame = ctk.CTkScrollableFrame(master=master_window, corner_radius=8, fg_color="#004127", scrollbar_button_color="#004127", scrollbar_button_hover_color="#004127", height=400)
    explanation_frame.grid(row=0, column=0, sticky="nsew", columnspan=12, rowspan=6, padx=10, pady=5)
    explanation_frame = grid_setup(explanation_frame)

    # Get the explanation.json file
    explanation_json = load_json("explanations")
    explanation_json = explanation_json[f"case_{num_excercise}"]

    # Put the title, description, type excersice, inputs text and outputs text, number of patterns and the requiremets
    # Title
    title_txt = "Ejercicio Número: " + str(num_excercise)
    title = ctk.CTkLabel(master=explanation_frame, text=title_txt, font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title.grid(row=0, column=0, pady=5, sticky="nsew", columnspan=12)

    # Description
    description_txt = explanation_json["description"]
    description = ctk.CTkLabel(master=explanation_frame, text=description_txt, font=("Arial", 16), justify="left", wraplength=860)
    description.grid(row=1, column=0, pady=5, sticky="w", columnspan=12)

    # Type of excersice
    type_txt = "Tipo de Ejercicio:"
    type_lbl = ctk.CTkLabel(master=explanation_frame, text=type_txt, font=("Arial", 16, "bold"), justify="left", wraplength=900, text_color=GUI_ELEMENTS["colors"]["foreground"])
    type_lbl.grid(row=2, column=0, pady=5, sticky="w", columnspan=2)

    type_txt2 = explanation_json["type"]
    type_lbl2 = ctk.CTkLabel(master=explanation_frame, text=type_txt2, font=("Arial", 16), justify="left", wraplength=900)
    type_lbl2.grid(row=2, column=2, pady=5, sticky="w", columnspan=10)

    # Inputs text
    inputs_txt = "Entradas: "
    inputs_lbl = ctk.CTkLabel(master=explanation_frame, text=inputs_txt, font=("Arial", 16, "bold"), justify="left", wraplength=800, text_color=GUI_ELEMENTS["colors"]["foreground"])
    inputs_lbl.grid(row=3, column=0, pady=5, sticky="w", columnspan=12)

    inputs_txt2 = explanation_json["inputs"]
    inputs_lbl2 = ctk.CTkLabel(master=explanation_frame, text=inputs_txt2, font=("Arial", 16), justify="left", wraplength=860)
    inputs_lbl2.grid(row=4, column=0, pady=5, sticky="w", columnspan=12)

    # Outputs text
    outputs_txt = "Salidas: "
    outputs_lbl = ctk.CTkLabel(master=explanation_frame, text=outputs_txt, font=("Arial", 16, "bold"), justify="left", wraplength=900, text_color=GUI_ELEMENTS["colors"]["foreground"])
    outputs_lbl.grid(row=5, column=0, pady=5, sticky="w", columnspan=12)

    outputs_txt2 = explanation_json["outputs"]
    outputs_lbl2 = ctk.CTkLabel(master=explanation_frame, text=outputs_txt2, font=("Arial", 16), justify="left", wraplength=860)
    outputs_lbl2.grid(row=6, column=0, pady=5, sticky="w", columnspan=12)

    # Number of patterns
    patterns_txt = "Número de Patrones: "
    patterns_lbl = ctk.CTkLabel(master=explanation_frame, text=patterns_txt, font=("Arial", 16, "bold"), justify="left", wraplength=900, text_color=GUI_ELEMENTS["colors"]["foreground"])
    patterns_lbl.grid(row=7, column=0, pady=5, sticky="w", columnspan=2)

    patterns_txt2 = explanation_json["number_of_patterns"]
    patterns_lbl2 = ctk.CTkLabel(master=explanation_frame, text=patterns_txt2, font=("Arial", 16), justify="left", wraplength=900)
    patterns_lbl2.grid(row=7, column=2, pady=5, sticky="w", columnspan=10)

    # Requirements
    requirements_txt = "Requerimientos: "
    requirements_lbl = ctk.CTkLabel(master=explanation_frame, text=requirements_txt, font=("Arial", 16, "bold"), justify="left", wraplength=900, text_color=GUI_ELEMENTS["colors"]["foreground"])
    requirements_lbl.grid(row=8, column=0, pady=5, sticky="w", columnspan=12)

    requirements_txt2 = explanation_json["requirements"]
    requirements_lbl2 = ctk.CTkLabel(master=explanation_frame, text=requirements_txt2, font=("Arial", 16), justify="left", wraplength=860)
    requirements_lbl2.grid(row=9, column=0, pady=5, sticky="w", columnspan=12)

    # Image of the example
    example_path = get_resource_path(explanation_json["example"])
    example_img = Image.open(example_path)
    width = example_img.size[0]
    height = example_img.size[1]
    width = int(width/1.5)
    height = int(height/1.5)
    example_img = ctk.CTkImage(dark_image=example_img, size=(width, height))

    example_lbl = ctk.CTkLabel(master=explanation_frame, image=example_img, compound="center", text="")
    example_lbl.grid(row=10, column=0, pady=5, sticky="nsew", columnspan=12)

    return explanation_frame

def excersice_frame_creation(master=None, num_excercise=0):
    """
    Create the excersice frame of the application.
    This frame is created when the user clicks on the excersice button.
    This frame contains the explanation of the excersice and the buttons to train and test the model.

    Parameters:
    master (ctk.CTK): Main window of the application.
    num_excercise (int): Number of the excercise.
    """
    global content_frame

    print(f"Excersice en excersice frame creation {num_excercise}")

    content_frame = ctk.CTkScrollableFrame(master=master, corner_radius=0, fg_color=GUI_ELEMENTS["colors"]["background"], scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    content_frame.grid(row=0, column=2, sticky="nsew", columnspan=12, rowspan=12)
    content_frame = grid_setup(content_frame)
    # Add the Explication Section
    row = 0
    if num_excercise != 0:
        explanation_frame = explanation_frame_creation(content_frame, num_excercise)
        row = 6
        # Buttons for train and test the model
        # Train button

        train_btn = ctk.CTkButton(master=content_frame, text="Entrenamiento", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
        train_btn.grid(row=row, column=0, pady=10, sticky="n")
        train_btn.configure(command= lambda: train_frame_creation(master=content_frame, num_excercise=num_excercise))

        test_btn = ctk.CTkButton(master=content_frame, text="Pruebas", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
        test_btn.grid(row=row, column=1, pady=10, sticky="n")
        test_btn.configure(command= lambda: test_frame_creation(master=master, number_excercise=num_excercise))

    else:
        train_frame_creation(master=content_frame, num_excercise=num_excercise, row=row)

def train_frame_creation(master = None, num_excercise = 0, row = 8, is_for_image = False):
    """
    Create the train frame of the application.
    This frame is created when the user clicks on the train button.
    This frame contains the inputs, comboboxes and checkboxes to train the model.
    
    Parameters:
    master (ctk.Frame): Frame where the train frame will be created.
    num_excercise (int): Number of the excercise.
    row (int): Row where the train frame will be created.
    is_for_image (bool): If the training is for an image or not
    """

    global data_train_json, download_weights_btn, download_training_data_btn, results_btn, train_frame, train_images_data,\
        label_categories

    print(f"Excersice number in train frame creation {num_excercise}")

    train_frame = None
    train_frame = ctk.CTkFrame(master=master, corner_radius=0, fg_color=GUI_ELEMENTS["colors"]["background"])
    train_frame.grid(row=row, column=0, sticky="nsew", columnspan=12, rowspan=4, padx=10, pady=5)
    train_frame = grid_setup(train_frame)

    # region Inputs, Comboboxes and Checkboxes for data training
    # Checkbox for use the last training data
    if not is_for_image:
        last_training = ctk.CTkCheckBox(master=train_frame, text="Usar datos del entrenamiento por defecto", font=("Arial", 16), text_color=GUI_ELEMENTS["colors"]["foreground"])
        last_training.grid(row=0, column=0, pady=10, sticky="w", columnspan=4)

    # Checkbox for momentun
    momentum = ctk.CTkCheckBox(master=train_frame, text="Momentum", font=("Arial", 16), text_color=GUI_ELEMENTS["colors"]["foreground"])
    momentum.grid(row=0, column=4, pady=10, sticky="w", columnspan=2)

    # Inputs for quantity of neurons in the hidden layer
    # Hidden Layer
    hidden_layer_txt = "Cantidad de Neuronas en Capa Oculta (H):"
    hidden_layer_lbl = ctk.CTkLabel(master=train_frame, text=hidden_layer_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    hidden_layer_lbl.grid(row=1, column=0, pady=10, sticky="w", columnspan=3)

    hidden_layer_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="3", placeholder_text_color= "#8d918e")
    hidden_layer_entry.grid(row=1, column=3, pady=10, sticky="w", columnspan=2)

    # Input for max epochs
    max_epochs_txt = "Número Máximo de Épocas:"
    max_epochs_lbl = ctk.CTkLabel(master=train_frame, text=max_epochs_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    max_epochs_lbl.grid(row=1, column=5, pady=10, sticky="w", columnspan=2)

    max_epochs_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="1000", placeholder_text_color= "#8d918e")
    max_epochs_entry.grid(row=1, column=7, pady=10, sticky="w", columnspan=2)

    # Add Bias, Alpha, Betha and Precision inputs
    # Bias
    bias_txt = "Bias (θ):"
    bias_lbl = ctk.CTkLabel(master=train_frame, text=bias_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    bias_lbl.grid(row=2, column=0, pady=10, sticky="w")

    bias_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=120, placeholder_text="0 es Random", placeholder_text_color= "#8d918e")
    bias_entry.grid(row=2, column=1, pady=10, sticky="w", padx=2)

    # Alpha
    alpha_txt = "Alpha (α):"
    alpha_lbl = ctk.CTkLabel(master=train_frame, text=alpha_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    alpha_lbl.grid(row=2, column=2, pady=10, sticky="w")

    alpha_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="0.005", placeholder_text_color= "#8d918e")
    alpha_entry.grid(row=2, column=3, pady=10, sticky="w", padx=2)

    # Betha
    betha_txt = "Betha (β):"
    betha_lbl = ctk.CTkLabel(master=train_frame, text=betha_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    betha_lbl.grid(row=2, column=4, pady=10, sticky="w")

    betha_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="0.3", placeholder_text_color= "#8d918e")
    betha_entry.grid(row=2, column=5, pady=10, sticky="w", padx=2)

    # Precision
    precision_txt = "Precision (ρ):"
    precision_lbl = ctk.CTkLabel(master=train_frame, text=precision_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    precision_lbl.grid(row=2, column=6, pady=10, sticky="w")

    precision_entry = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=100, placeholder_text="0.0001", placeholder_text_color= "#8d918e")
    precision_entry.grid(row=2, column=7, pady=10, sticky="w", padx=2)

    # Combobox for the function to use in H layer
    function_txt = "Función de Activación Capa Oculta (H):"
    function_lbl = ctk.CTkLabel(master=train_frame, text=function_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    function_lbl.grid(row=3, column=0, pady=10, sticky="w", columnspan=3)

    layer_h_options = ["Tangente Hiperbólica", "Sigmoidal", "Softplus","ReLU", "Leaky ReLU", "Lineal", "ELU", "Swish"]
    layer_h_combobox = ctk.CTkComboBox(master=train_frame, values=layer_h_options, font=("Arial", 16), width=200, border_width=0)
    layer_h_combobox.grid(row=3, column=3, pady=10, sticky="w", columnspan=3)

    # Combobox for the function to use in O layer
    function_txt2 = "Función de Activación Capa Salida (O):"
    function_lbl2 = ctk.CTkLabel(master=train_frame, text=function_txt2, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    function_lbl2.grid(row=4, column=0, pady=10, sticky="w", columnspan=3)

    layer_o_options = ["Tangente Hiperbólica", "Sigmoidal", "Softplus", "ReLU", "Leaky ReLU", "Lineal", "ELU", "Swish"]
    layer_o_combobox = ctk.CTkComboBox(master=train_frame, values=layer_o_options, font=("Arial", 16), width=200, border_width=0)
    layer_o_combobox.grid(row=4, column=3, pady=10, sticky="w", columnspan=3)
    # endregion

    # Buttons for dowlnoad the template and load the data
    # Download template button
    if not is_for_image:
        download_btn = ctk.CTkButton(master=train_frame, text="Descargar Plantilla", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010", command=lambda: download_json(f"base_{num_excercise}_template",directory=f"Data/case_{num_excercise}"))
        download_btn.grid(row=5, column=0, pady=10, sticky="n", columnspan=2)

    # Load data button
    if not is_for_image:
        load_data_btn = ctk.CTkButton(master=train_frame, text="Cargar Datos", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
        load_data_btn.grid(row=5, column=2, pady=10, sticky="n", columnspan=2)

    # Status label
    status_lbl = ctk.CTkLabel(master=train_frame, text="No Cargado", font=("Arial", 16, "bold"), text_color="red")
    status_lbl.grid(row=5, column=4, pady=10, sticky="n", columnspan=2)

    if is_for_image:
        status_lbl.configure(text="Imagenes Cargadas", text_color="green")

    if not is_for_image:
        last_training.configure(command=lambda: chargue_last_training_data(train_frame, status_lbl, bias_entry, alpha_entry, betha_entry, precision_entry, layer_h_combobox, layer_o_combobox, num_excercise, momentum, hidden_layer_entry, max_epochs_entry))
        load_data_btn.configure(command=lambda: chargue_data_training(train_frame, status_lbl))

    # Status of the training start
    status_lbl2 = ctk.CTkLabel(master=train_frame, text="", font=("Arial", 16, "bold"), text_color="red")
    status_lbl2.grid(row=8, column=2, pady=10, sticky="n", columnspan=2)

    # Train button
    train_btn = ctk.CTkButton(master=train_frame, text="Iniciar Entrenamiento", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    train_btn.grid(row=8, column=0, pady=10, sticky="n", columnspan=2)
    train_btn.configure(command= lambda: start_training(status_lbl, status_lbl2,bias_entry, alpha_entry, betha_entry, precision_entry, layer_h_combobox, layer_o_combobox, num_excercise, momentum, hidden_layer_entry, max_epochs_entry, is_for_image = is_for_image))

    # Stop training button
    stop_btn = ctk.CTkButton(master=train_frame, text="Detener Entrenamiento", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    stop_btn.grid(row=8, column=4, pady=10, sticky="n", columnspan=2)
    stop_btn.configure(command= lambda: set_stop_training(True))

    # Download weights button
    download_weights_btn = ctk.CTkButton(master=train_frame, text="Descargar Pesos", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010", state="disabled")
    download_weights_btn.grid(row=10, column=0, pady=10, sticky="n", columnspan=2)
    download_weights_btn.configure(command= lambda: download_weights(num_excercise))
    
    # Results buton -> Command = section => Frame with weights json info and graphs
    results_btn = ctk.CTkButton(master=train_frame, text="Resultados", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010", state="disabled")
    results_btn.grid(row=10, column=2, pady=10, sticky="n", columnspan=2)
    if is_for_image:
        results_btn.configure(command= lambda: results_frame_creation(train_frame, is_for_image=True))
    else:
        results_btn.configure(command= lambda: results_frame_creation(train_frame))
            
    # Downoload training data button
    download_training_data_btn = ctk.CTkButton(master=train_frame, text="Descargar Datos de Entrenamiento", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=240, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010", state="disabled")
    download_training_data_btn.grid(row=10, column=4, pady=10, sticky="n", columnspan=4)
    download_training_data_btn.configure(command= lambda: download_training_data(num_excercise))
    
    inputs, outputs = [], []
    
    if is_for_image: 
        label_categories = set()
        
        for category in train_images_data:
            label_categories.add(category["label"])
        
        for category in train_images_data:
            status_c = category["status"]
            images = category["images"]
            label = category["label"]
            
            if status_c == "flattened":
                n_images = [normalize_image_vector(image) for image in images]
                category["images"] = n_images
                
                for i in range(len(n_images)):
                    inputs.append(n_images[i])
                    # append the index of the set category labels to the outputs
                    outputs.append([list(label_categories).index(label)])

        data_train_json = {
            "inputs": inputs,
            "outputs": outputs
        }
        
        errors_text = create_training_process_frame(train_frame, len(inputs))

def changue_status_training(status_label, text, color, download_weights_btn, download_training_data_btn, results_btn):
    """
    Function to changue the status of the training in the GUI

    Parameters:
    status_label: Label to show the status
    text: Text to show in the label
    color: Color of the text
    download_weights_btn: Button to download the weights
    download_training_data_btn: Button to download the training data
    results_btn: Button to show the
    """
    
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
    """
    Function to chargue the data in the data frame
    This function is called when the user clicks on the load data button and creates the data frame with the data loaded
    
    Parameters:
    master: Frame where the data frame will be created
    status: Label to show the status of the data
    """
    global data_train_json
    # load json data
    data_train_json, filename = load_json()
 
    # frame to show info of the data
    data_frame = create_data_frame(master, data_train_json)

    status.configure(text="Cargado", text_color="green")

def create_data_frame(master, data_json, row=7, is_training=True):
    """
    Create the data frame with the data loaded in the application
    This frame contains the data loaded in the application and shows the inputs and outputs of the data
    
    Parameters:
    master: Frame where the data frame will be created
    data_json: JSON with the data loaded
    row: Row where the data frame will be created
    is_training: Boolean to know if the data is for training
    """
    
    global   data_inputs_text, errors_text, GUI_ELEMENTS
    
    data_inputs_text = ctk.CTkTextbox(master=master, corner_radius=8 , font=("Consolas", 16), fg_color="#ffffff", wrap="word", width=860, height=300)
    data_inputs_text.grid(row=row, column=0, sticky="new", columnspan=12, pady=10, padx=10)
    
    data_inputs_text.configure(state="normal") 
    data_inputs_text.delete(1.0, "end")
    
    data_inputs_text.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    data_inputs_text.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    data_inputs_text.tag_config("subtitle_2", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    data_inputs_text.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="black")
    
    title_txt = "Datos de Entrenamiento"
    data_inputs_text.insert("end", f"{title_txt}\n\n", "title")

    inputs_txt = "Entradas: "
    outputs_txt = "Salidas: "
    inputs = data_json["inputs"]
    outputs = data_json["outputs"]
    for i in range(len(inputs)):
        data_inputs_text.insert("end", f"Patrón {i+1}: \n", 'subtitle')
        data_inputs_text.insert("end", f"{inputs_txt}", 'subtitle_2')
        data_inputs_text.insert("end", f"{inputs[i]}\n", 'pattern')
        data_inputs_text.insert("end", f"{outputs_txt}", 'subtitle_2')
        data_inputs_text.insert("end", f"{outputs[i]}\n\n", 'pattern')

    data_inputs_text.configure(state="disabled")
    if is_training:
        # Create the training data process textbox
        errors_text = create_training_process_frame(master, len(inputs))

    return data_inputs_text

def start_training(status, status2, bias_entry, alpha_entry, betha_entry, precision_entry, layer_h_combobox, layer_o_combobox, excercise_number, momentum, hidden_layer_entry, max_epochs_entry, is_for_image=False):
    """
    Function to start the training process of the model
    This function is called when the user clicks on the train button
    This function validates the inputs and starts the training process and creates a thread to train the model

    Parameters:
    status: Label to show the status of the training
    status2: Label to show the status of the training process
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
    is_for_image: Boolean to know if the training is for an image
    
    """
    
    global data_train_json, errors_text ,download_weights_btn, download_training_data_btn, results_btn, num_excercise,\
        main_window
    
    num_excercise = excercise_number
    
    set_stop_training(False)

    # region Values validation
    if data_train_json is None and not is_for_image:
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

    if function_h == "Tangente Hiperbólica":
        function_h = "tanh"
    elif function_h == "Sigmoidal":
        function_h = "sigmoid"

    if function_o == "Tangente Hiperbólica":
        function_o = "tanh"
    elif function_o == "Sigmoidal":
        function_o = "sigmoid"
        

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
    if not is_for_image:
        print(json.dumps(train_data, indent=2))
    else:
        print("Data for images")
        temp_train_data = train_data.copy()
        temp_train_data["inputs"] = "Images"
        print(json.dumps(temp_train_data, indent=2))
    
    # Start the training process
    normalize = False if is_for_image else True
    training_thread = threading.Thread(target=backpropagation_training, args=(train_data, errors_text, status2, download_weights_btn, download_training_data_btn, results_btn, main_window, normalize))
    training_thread.start()

    changue_status_training(status2, "Entrenando...", "orange", download_weights_btn, download_training_data_btn, results_btn)

def create_training_process_frame(master, num_patterns, row=9):
    """
    Create the training process frame in the application
    This frame contains the process of the training with the actual number ephoc, the errors of the patterns and the total error
    
    Parameters:
    master: Frame where the training process frame will be created
    num_patterns: Number of patterns in the data
    row: Row where the training process frame will be created
    """
    
    global errors_text, GUI_ELEMENTS
    errors_text = ctk.CTkTextbox(master=master, corner_radius=8, font=("Arial", 16), fg_color="#ffffff", wrap="word", width=860, height= 660, activate_scrollbars=True)
    errors_text.grid(row=row, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)

    errors_text.configure(state="normal")
    errors_text.delete(1.0, "end")

    errors_text.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    errors_text.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    errors_text.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="#70600f")
    errors_text.tag_config("error_high", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="red")
    errors_text.tag_config("error_low", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="green")

    title_txt = "Proceso de Entrenamiento"
    errors_text.insert("end", f"{title_txt}\n", "title")

    epoch_txt = "Época: 0"
    total_error_txt = "Error Total:"
    
    errors_text.insert("end", f"{epoch_txt}\t\t\t", "subtitle")
    errors_text.insert("end", f"{total_error_txt} ", "subtitle")
    errors_text.insert("end", "∞\n\n", "error_high")

    j = 1
    for i in range(num_patterns):
        pattern_txt = f"Patrón {i+1} |E|:"
        error_txt = "∞"
        if j % 4 != 0:
            errors_text.insert("end", f"{pattern_txt} ", "subtitle")
            errors_text.insert("end", f"{error_txt}\t\t", "error_high")
            j += 1
        else:
            errors_text.insert("end", f"{pattern_txt} ", "subtitle")
            errors_text.insert("end", f"{error_txt}\n", "error_high")
            j = 1

    errors_text.configure(state="disabled")
    return errors_text

def update_errors_ui(epoch, errores_patrones, total_error, precision, errors_text, main_window, final = False):
    """
    Update the errors in the training process frame in the application
    This function is called when the training process is running and updates the errors in the frame

    Parameters:
    epoch: Actual epoch in the training process
    errores_patrones: List with the errors of the patterns
    total_error: Total error in the training process
    precision: Precision of the training to know which color to use in the errors
    errors_text: Textbox with the errors
    main_window: Main window of the application
    final: Boolean to know if the training is finished
    """
    errors_text.configure(state="normal")
    errors_text.delete(1.0, "end")

    errors_text.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    errors_text.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    errors_text.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="#70600f")
    errors_text.tag_config("error_high", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="red")
    errors_text.tag_config("error_low", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="green")

    title_txt = "Proceso de Entrenamiento"
    errors_text.insert("end", f"{title_txt}\n\n", "title")

    epoch_txt = "Época: " + str(epoch)
    total_error_txt = "Error Total: "
    total_error_txt2 = f"{total_error:.10f}\n"
    
    errors_text.insert("end", f"{epoch_txt}\t\t\t\t\t", "subtitle")
    errors_text.insert("end", f"{total_error_txt} ", "subtitle")
    if total_error <= precision:
        errors_text.insert("end", f"{total_error_txt2}\n", "error_low")
    else:
        errors_text.insert("end", f"{total_error_txt2}\n", "error_high")

    j = 1
    for i in range(len(errores_patrones)):
        color_tag = "error_low" if errores_patrones[i] <= precision else "error_high"
        pattern_txt = f"Patrón {i+1} |E|:"
        error_txt = f"{errores_patrones[i]:.10f}"
        if color_tag == "error_low" and len(errores_patrones) > 200 and not final:
            continue
        if errores_patrones[i] <= (precision/2) and not final and len(errores_patrones) > 50:
            continue
        if j % 4 != 0:
            errors_text.insert("end", f"{pattern_txt} ", "subtitle")
            errors_text.insert("end", f"{error_txt}\t\t\t\t", color_tag)
            j += 1
        else:
            errors_text.insert("end", f"{pattern_txt} ", "subtitle")
            errors_text.insert("end", f"{error_txt}\n", color_tag)
            j = 1

    main_window.update_idletasks() 
    
def download_weights(num_excercise):
    """
    Function to download the weights in a json file
    This function is called when the user clicks on the download weights button after the training process is finished
    """
    weights_json =  get_weights_json()
    download_json(filename= f"Resultados_ejercicio_{num_excercise}", data= weights_json)

def download_training_data(num_excercise):
    """
    Function to download the training data in a json file
    This function is called when the user clicks on the download training data button after the training process is finished
    """
    data_train_json = get_graph_json()
    download_json(filename= f"Entrenamiento_ejercicio_{num_excercise}", data= data_train_json)

def results_frame_creation(master, is_train = True, weights_json = None, test_data = None, is_for_image = False):
    """
    Function to create the results frame in the application
    This frame contains the results of the training process with the weights, the graphs and the test results
    
    Parameters:
    master: Frame where the results frame will be created
    is_train: Boolean to know if the results are for the training process
    weights_json: JSON with the weights of the model
    test_data: JSON with the test data
    is_for_image: Boolean to know if the training is for an image in order to don't show weights info (because is too large)   
    """
    global data_train_json, num_excercises
    inputs, outputs = None, None
    
    print("in results frame creation is for image", is_for_image)
    
    weights_json =  get_weights_json() if weights_json is None else weights_json
    results_frame = ctk.CTkFrame(master=master, corner_radius=8, fg_color="#ffffff")
    results_frame.grid(row=11, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    results_frame = grid_setup(results_frame)

    row = 0
    if is_train:
        graph_json = get_graph_json()
        if not is_for_image:
            
            print("Weights Info")
            
            row = add_weights_info(results_frame, weights_json, row, title_txt="Resultados del Entrenamiento")
    
            # Horizontal separator
            ctk.CTkFrame(master=results_frame, corner_radius=0, fg_color=GUI_ELEMENTS["colors"]["background"], height=2).grid(row=row, column=0, pady=5, sticky="nsew", columnspan=12)
            row += 1

        row = add_graph_info(results_frame, graph_json, row)

        # Horizontal separator
        ctk.CTkFrame(master=results_frame, corner_radius=0, fg_color=GUI_ELEMENTS["colors"]["background"], height=2).grid(row=row, column=0, pady=5, sticky="nsew", columnspan=12)
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

    print("Results Info")

    add_results_info(results_frame, test_data, row, num_excercise, is_for_image)

def add_weights_info(frame, weights_json, row, title_txt):
    """
    Add the weights info in the results frame
    This function is called when the user clicks on the results button after the training process is finished
    
    Parameters:
    frame: Frame where the weights info will be created
    weights_json: JSON with the weights of the model
    row: Row where the weights info will be created
    title_txt: Title of the weights info    
    """
    global weight_text, GUI_ELEMENTS
    
    weight_text = ctk.CTkTextbox(master=frame, corner_radius=8, font=("Arial", 16), fg_color="#ffffff", wrap="word", width=860, height=400)
    weight_text.grid(row=row, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    row += 1
    
    weight_text.configure(state="normal")
    weight_text.delete(1.0, "end")

    weight_text.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    weight_text.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    weight_text.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="#70600f")
    weight_text.tag_config("separator", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground=GUI_ELEMENTS["colors"]["background"], justify="center")

    weight_text.insert("end", f"{title_txt}\n\n", "title")
    weights_h_txt = "Pesos de la Capa Oculta (H):"
    weight_text.insert("end", f"{weights_h_txt}\n\n", "subtitle")

    weights_h = weights_json["weights_h"]
    for j in range(len(weights_h)):
        weights_txt = f"J {j+1}:\n"
        for i in range(len(weights_h[j])):
            if (i + 1) % 4 != 0:
                weights_txt += f" W[{j+1}][{i+1}] {weights_h[j][i]:.10f}"
            else:
                weights_txt += f" W[{j+1}][{i+1}] {weights_h[j][i]:.10f}\n"
        weight_text.insert("end", f"{weights_txt}\n", "pattern")

    separator_txt = "-" * 86
    weight_text.insert("end", f"{separator_txt}\n", "separator")

    bias_h_txt = "Bias de la Capa Oculta (H):"
    weight_text.insert("end", f"\n{bias_h_txt}\n\n", "subtitle")
    
    bias_h = weights_json["bias_h"]
    for i in range(len(bias_h)):
        bias_txt = f"J {i+1}: {bias_h[i]:.10f}\n"
        weight_text.insert("end", f"{bias_txt}\n", "pattern")

    weight_text.insert("end", f"{separator_txt}\n", "separator")

    weights_o_txt = "Pesos de la Capa de Salida (O):"
    weight_text.insert("end", f"\n{weights_o_txt}\n\n", "subtitle")

    weights_o = weights_json["weights_o"]
    for j in range(len(weights_o)):
        weights_txt = f"K {j+1}:\n"
        for i in range(len(weights_o[j])):
            if (i + 1) % 4 != 0:
                weights_txt += f" W[{j+1}][{i+1}] {weights_o[j][i]:.10f}"
            else:
                weights_txt += f" W[{j+1}][{i+1}] {weights_o[j][i]:.10f}\n"
        weight_text.insert("end", f"{weights_txt}\n", "pattern")

    weight_text.insert("end", f"{separator_txt}\n", "separator")

    bias_o_txt = "Bias de la Capa de Salida (O):"
    weight_text.insert("end", f"\n{bias_o_txt}\n\n", "subtitle")

    bias_o = weights_json["bias_o"]
    for i in range(len(bias_o)):
        bias_txt = f"K {i+1}: {bias_o[i]:.10f}\n"
        weight_text.insert("end", f"{bias_txt}\n", "pattern")

    weight_text.configure(state="disabled")

    return row

def add_graph_info(frame, graph_json, row):
    """
    Add the graphs info in the results frame
    This function is called when the user clicks on the results button after the training process is finished
    
    Parameters:
    frame: Frame where the graphs info will be created
    graph_json: JSON with the graphs info
    row: Row where the graphs info will be created
    """
    max_epochs_txt = f"Número Máximo de Épocas: {graph_json['max_epochs']}"
    max_epochs_lbl = ctk.CTkLabel(master=frame, text=max_epochs_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["background"], justify="center", anchor="center")
    max_epochs_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    qty_neurons_txt = f"Cantidad de Neuronas en Capa Oculta (H): {graph_json['qty_neurons']}"
    qty_neurons_lbl = ctk.CTkLabel(master=frame, text=qty_neurons_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["background"], justify="center", anchor="center")
    qty_neurons_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    function_h_txt = f"Función de Activación Capa Oculta (H): {graph_json['function_h']}"
    function_h_lbl = ctk.CTkLabel(master=frame, text=function_h_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["background"], justify="center", anchor="center")
    function_h_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    function_o_txt = f"Función de Activación Capa de Salida (O): {graph_json['function_o']}"
    function_o_lbl = ctk.CTkLabel(master=frame, text=function_o_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["background"], justify="center", anchor="center")
    function_o_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    momentum_txt = f"Momentum: {'Si' if graph_json['momentum'] else 'No'}"
    if graph_json['momentum']:
        momentum_txt += f" con Betha = {graph_json['b']}"
    
    momentum_lbl = ctk.CTkLabel(master=frame, text=momentum_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["background"], justify="center", anchor="center")
    momentum_lbl.grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
    row += 1

    # Date of the training
    date_txt = f"Fecha del Entrenamiento: {graph_json['training_date']}"
    date_lbl = ctk.CTkLabel(master=frame, text=date_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["background"], justify="center", anchor="center")
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
    if max(graph_json["arquitecture"]) <= 35:
        arquitecture_graph = graph_json["arquitecture"]
        labels = max(arquitecture_graph) < 5
        figure_arquitecture = plot_neural_network_with_labels(arquitecture_graph, labels)
        canvas_arquitecture = FigureCanvasTkAgg(figure_arquitecture, master=frame)
        canvas_arquitecture.draw()
        canvas_arquitecture.get_tk_widget().grid(row=row, column=0, pady=5, sticky="new", columnspan=12)
        row += 1

    return row

def add_results_info(frame, test_data, row, num_excercise=0, is_for_image = False):
    """
    Add the results info in the results frame
    This function is called when the user clicks on the results button after the training process is finished
    
    Parameters:
    frame: Frame where the results info will be created
    test_data: JSON with the test data
    row: Row where the results info will be created
    num_excercise: Number of the excercise
    is_for_image: Boolean to know if the training is for an image in order to show the label of the results
    """
    global results_text, GUI_ELEMENTS
    print("in add results info is for image", is_for_image)

    results_text = ctk.CTkTextbox(master=frame, corner_radius=8, font=("Arial", 16), fg_color="#ffffff", wrap="word", width=860, height=400)
    results_text.grid(row=row, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)

    results_text.configure(state="normal")
    results_text.delete(1.0, "end")

    results_text.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    results_text.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    results_text.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    results_text.tag_config("separator", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground=GUI_ELEMENTS["colors"]["background"], justify="center")

    title_test_txt = "Resultados de la Prueba"
    results_text.insert("end", f"{title_test_txt}\n\n", "title")

    results, errors = test_neural_network(test_data)
    label_list = list(label_categories) if is_for_image else None

    for i in range(len(results)):
        print("Writing Results ... ", results[i])
        separator_txt = "-" * 86
        results_text.insert("end", f"{separator_txt}\n", "separator")

        test_txt = f"Patrón {i+1}:\n"
        results_text.insert("end", f"{test_txt}", "subtitle")

        input = test_data["inputs"][i]
        output = test_data["outputs"][i]
        error = errors[i]
        output_char = []
        result_char = []
        output_label = ""
        result_label = ""

        if num_excercise == 4:
            for j in range(len(output)):
                output_char.append(chr(round(output[j])))
                result_char.append(chr(round(results[i][j])))

        if is_for_image:
            try:
                output_round = round(output[0])
                output_label = label_list[output_round]
            except ValueError:
                output_label = "Desconocido"
                
            try:
                result_round = round(results[i][0])
                result_label = label_list[result_round]
            except ValueError:
                result_label = "Desconocido"

        y_obtained = ""
        for j in range(len(results[i])):
            y_obtained += f"{results[i][j]:.10f}"
            if not is_for_image:
                y_obtained += '\n'

        if num_excercise != 4 and not is_for_image:
            test_txt2 = f"Entradas:\n {input}\nSalidas Esperadas:\n {output}\nSalidas Obtenidas:\n {y_obtained}\nError:\n {error}"
        elif is_for_image:
            print("is for image")
            test_txt2 = f"Entradas:\n image \nSalidas Esperadas:\n {output} = {output_label}\nSalidas Obtenidas:\n {y_obtained} = {result_label} \nError:\n {error}"
        else:
            test_txt2 = f"Entradas:\n {input}\nSalidas Esperadas:\n {output} = {output_char}\nSalidas Obtenidas:\n {y_obtained} = {result_char}\nError:\n {error}"
        
        results_text.insert("end", f"{test_txt2}\n\n", "pattern")

    results_text.configure(state="disabled")
    print("Results Info Done")
    return row + 1

def test_frame_creation(master, number_excercise=0):
    """
    Create the test frame in the application
    This frame contains the test section with the data and the weights to test the model
    
    Parameters:
    master: Frame where the test frame will be created
    number_excercise: Number of the excercise
    
    """
    global data_test_json, weights_test_json, num_excercise

    num_excercise = number_excercise

    test_frame = ctk.CTkScrollableFrame(master=master, corner_radius=0, fg_color=GUI_ELEMENTS["colors"]["background"],scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    test_frame.grid(row=0, column=2, sticky="nsew", columnspan=12, rowspan=12)
    test_frame = grid_setup(test_frame)

    # Title of the test
    title_txt = "Sección de Prueba"
    title = ctk.CTkLabel(master=test_frame, text=title_txt, font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)

    # checkbox for data test defect
    defect_test = ctk.CTkCheckBox(master=test_frame, text="Usar Datos de Prueba por Defecto", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    defect_test.grid(row=1, column=0, pady=5, padx=15 , sticky="new", columnspan=12)

    # Load data button
    load_data_btn = ctk.CTkButton(master=test_frame, text="Cargar Entradas y Salidas", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    load_data_btn.grid(row=2, column=0, pady=10, sticky="n", columnspan=2)

    # Status label
    status_lbl = ctk.CTkLabel(master=test_frame, text="No Cargado", font=("Arial", 16, "bold"), text_color="red")
    status_lbl.grid(row=2, column=2, pady=10, sticky="n", columnspan=3)

    # Load data button
    load_weights_btn = ctk.CTkButton(master=test_frame, text="Cargar Pesos", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    load_weights_btn.grid(row=2, column=6, pady=10, sticky="n", columnspan=2)

    # Status label
    status_lbl2 = ctk.CTkLabel(master=test_frame, text="No Cargado", font=("Arial", 16, "bold"), text_color="red")
    status_lbl2.grid(row=2, column=8, pady=10, sticky="n", columnspan=3)

    # Test button
    test_btn = ctk.CTkButton(master=test_frame, text="Iniciar Prueba", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010", state="disabled")
    test_btn.grid(row=5, column=0, pady=10, sticky="n", columnspan=2)


    defect_test.configure(command= lambda: chargue_default_data_test(test_frame, num_excercise, status_lbl, status_lbl2, test_btn))
    load_data_btn.configure(command= lambda: chargue_data_test(test_frame, status_lbl, status_lbl2, test_btn))
    load_weights_btn.configure(command= lambda: chargue_weights_test(test_frame, status_lbl, status_lbl2, test_btn))
    test_btn.configure(command= lambda: results_frame_creation(test_frame, is_train=False, weights_json=weights_test_json, test_data=data_test_json))    

def chargue_default_data_test(frame, num_excercise, status, status2, test_btn):
    """
    Function to load the default test data in the application
    This function is called when the user clicks on the checkbox to use the default test data
    The data is loaded from the json files in the Data folder and the frames are created with the data and the weights
    
    Parameters:
    frame: Frame where the test data will be loaded
    num_excercise: Number of the excercise
    status: Label to show the status of the data
    status2: Label to show the status of the weights
    test_btn: Button to start the test
    
    """
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
    """
    Function to load the test data in the application
    This function is called when the user clicks on the load data button
    The data is loaded from a json file and the frame is created with the data to show it
    
    Parameters:
    master: Frame where the test data will be loaded
    status: Label to show the status of the data
    status2: Label to show the status of the weights
    test_btn: Button to start the test
    """
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
    """
    Function to load the test weights in the application
    This function is called when the user clicks on the load weights button
    The weights are loaded from a json file and the frame is created with the weights to show it
    
    Parameters:
    master: Frame where the test weights will be loaded
    status: Label to show the status of the data
    status2: Label to show the status of the weights
    test_btn: Button to start the test
    """
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
    """
    Create the weights frame in the application
    This frame contains the weights and bias of the model
    
    Parameters:
    frame: Frame where the weights frame will be created
    data_weights_json: JSON with the weights of the model
    row: Row where the weights frame will be created
    """
    
    weights_frame = ctk.CTkFrame(master=frame, corner_radius=8, fg_color="#ffffff")
    weights_frame.grid(row=row, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    weights_frame = grid_setup(weights_frame)

    # Title of the weights
    title_txt = "Pesos y Bias"
    row = add_weights_info(weights_frame, data_weights_json, 0, title_txt)

    return weights_frame
# endregion

# region Image Treatment
def image_treatment_frame_creation(frame):
    """
    Create the image treatment frame in the application
    This frame contains the image treatment section with the filters and the kernel selection
    
    Parameters:
    frame: Frame where the image treatment frame will be created
    """
    global is_folder_img, kernel_names, apply_kernel_btn, remove_kernel_btn,image_treatment_btn_2, filtered_images,download_images_btn,\
        download_kernel_btn, resize_images_btn

    image_treatment_frame = ctk.CTkScrollableFrame(master=frame, corner_radius=0, fg_color=GUI_ELEMENTS["colors"]["background"],scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    image_treatment_frame.grid(row=0, column=2, sticky="nsew", columnspan=12, rowspan=12)
    image_treatment_frame = grid_setup(image_treatment_frame)

    # Title of the image treatment
    title_txt = "Tratamiento de Imágenes"
    title = ctk.CTkLabel(master=image_treatment_frame, text=title_txt, font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)

    # frame for the first functions
    first_row_frame = ctk.CTkFrame(master=image_treatment_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    first_row_frame.grid(row=1, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    first_row_frame = grid_setup(first_row_frame)

    # Check button for folder with images
    folder_check = ctk.CTkCheckBox(master=first_row_frame, text="Usar Carpeta con Imágenes", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    folder_check.grid(row=1, column=0, pady=20, padx=15 , sticky="new", columnspan=4)
    folder_check.configure(command= lambda: check_folder(folder_check))

    # Load image button
    image_treatment_btn = ctk.CTkButton(master=first_row_frame, text="Cargar Datos", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    image_treatment_btn.grid(row=1, column=4, pady=10, sticky="wn", columnspan=2, padx=10)
    image_treatment_btn.configure(command= lambda: chargue_images(image_treatment_frame, is_folder_img))
    
    # button for load images with the kernel applied
    image_treatment_btn_2 = ctk.CTkButton(master=first_row_frame, text="Cargar Imágenes Filtradas", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    image_treatment_btn_2.grid(row=1, column=6, pady=10, sticky="wn", columnspan=2)
    image_treatment_btn_2.configure(state = "disabled")
    image_treatment_btn_2.configure(command= lambda: chargue_filtered_images(image_treatment_frame))
    
    # Help button
    help_btn = ctk.CTkButton(master=first_row_frame, text="Ayuda", fg_color=GUI_ELEMENTS["colors"]["info_button"], width=100, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["info_button_hover"], text_color=GUI_ELEMENTS["colors"]["foreground"])
    help_btn.grid(row=1, column=8, pady=10, sticky="en", columnspan=4, padx=10)
    help_btn.configure(command= lambda: show_info_treatment())
    
    first_functions_frame = ctk.CTkFrame(master=image_treatment_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    first_functions_frame.grid(row=2, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    first_functions_frame = grid_setup(first_functions_frame)
    
    # CTkLabel for kernel selection
    kernel_lbl = ctk.CTkLabel(master=first_functions_frame, text="Kernel:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    kernel_lbl.grid(row=2, column=0, pady=10, padx=15, sticky="wsn", columnspan=1)
    
    # CTkComboBox for kernel selection
    kernel_combobox = ctk.CTkComboBox(master=first_functions_frame,values=kernel_names  , width=200, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], fg_color=GUI_ELEMENTS["colors"]["foreground"], text_color="#0F1010", state="readonly")
    kernel_combobox.grid(row=2, column=1, pady=10, padx = 10, sticky="wn", columnspan=2)
    
    # CTkLabel for stride value
    stride_lbl = ctk.CTkLabel(master=first_functions_frame, text="Stride:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    stride_lbl.grid(row=2, column=3, pady=10, sticky="en", columnspan=1)

    # CTkEntry for stride value
    stride_entry = ctk.CTkEntry(master=first_functions_frame, width=60, font=GUI_ELEMENTS["fonts"]["button_font"])
    stride_entry.grid(row=2, column=4, pady=10, padx = 10, sticky="wn", columnspan=1)
    stride_entry.insert(0, "1")
    
    # CTkLabel for padding value
    padding_lbl = ctk.CTkLabel(master=first_functions_frame, text="Padding:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    padding_lbl.grid(row=2, column=5, pady=10, sticky="en", columnspan=1)
    
    # CTkEntry for padding value
    padding_entry = ctk.CTkEntry(master=first_functions_frame, width=60, font=GUI_ELEMENTS["fonts"]["button_font"])
    padding_entry.grid(row=2, column=6, pady=10, padx = 10, sticky="wn", columnspan=1)
    padding_entry.insert(0, "0")
    
    # CTkLabel for percentaje value
    percentaje_lbl = ctk.CTkLabel(master=first_functions_frame, text="Porcentaje para color:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    percentaje_lbl.grid(row=2, column=7, pady=10, sticky="en", columnspan=1)

    # CTkEntry for percentaje value
    percentaje_entry = ctk.CTkEntry(master=first_functions_frame, width=60, font=GUI_ELEMENTS["fonts"]["button_font"])
    percentaje_entry.grid(row=2, column=8, pady=10, padx = 10, sticky="wn", columnspan=1)
    percentaje_entry.insert(0, "0.5")
    
    # Button for apply kernel
    apply_kernel_btn = ctk.CTkButton(master=first_functions_frame ,  text="Aplicar Kernel", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    apply_kernel_btn.grid(row=2, column=9, pady=10, sticky="wn", columnspan=2)
    apply_kernel_btn.configure(command= lambda: apply_sel_kernel(kernel_combobox.get(), int(stride_entry.get()), int(padding_entry.get()), float(percentaje_entry.get()) , image_treatment_frame))
    apply_kernel_btn.configure(state = "disabled")  
    
    # Button for remove kernel
    remove_kernel_btn = ctk.CTkButton(master=first_functions_frame, text="Deshacer Tratamiento", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    remove_kernel_btn.grid(row=2, column=11, pady=10, sticky="wn", columnspan=2)
    remove_kernel_btn.configure(command= lambda: remove_kernel())
    remove_kernel_btn.configure(state = "disabled")  
    
    # Add the resize section to the frame   
    resize_frame = ctk.CTkFrame(master=image_treatment_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    resize_frame.grid(row=3, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    resize_frame = resize_frame_creation(resize_frame, for_training=False, for_treatment=True)
    
    # Frame for additional functions of the images
    additional_functions_frame = ctk.CTkFrame(master=image_treatment_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    additional_functions_frame.grid(row=4, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    additional_functions_frame = grid_setup(additional_functions_frame)
    
    # label for the angle of rotation of the images
    angle_lbl = ctk.CTkLabel(master=additional_functions_frame, text="Ángulo:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    angle_lbl.grid(row=0, column=0, pady=10, padx=15, sticky="swen", columnspan=1)
    
    # entry for the angle of rotation of the images
    angle_entry = ctk.CTkEntry(master=additional_functions_frame, width=60, font=GUI_ELEMENTS["fonts"]["button_font"])
    angle_entry.grid(row=0, column=1, pady=10, padx = 10, sticky="swen", columnspan=1)
    
    # Button for rotate images
    rotate_images_btn = ctk.CTkButton(master=additional_functions_frame, text="Rotar Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=120, height=30, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    rotate_images_btn.grid(row=0, column=2, pady=10, sticky="wn", columnspan=2)
    rotate_images_btn.configure(command= lambda: rotate_images(angle_entry.get()))
    
    # ComboBox for the type of colormap in cv2 options
    colormaps = ["AUTUMN", "BONE", "JET", "WINTER", 
                 "RAINBOW", "OCEAN", "SUMMER", "SPRING", 
                 "COOL", "HSV", "PINK", "HOT", "PARULA",
                 "MAGMA", "INFERNO", "PLASMA", "VIRIDIS", 
                 "CIVIDIS","TWILIGHT", "TWILIGHT_SHIFTED", "TURBO",
                 "DEEPGREEN"]
    
    combobox_colormap = ctk.CTkComboBox(master=additional_functions_frame, values=colormaps, width=200, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], fg_color=GUI_ELEMENTS["colors"]["foreground"], text_color="#0F1010", state="readonly")
    combobox_colormap.grid(row=0, column=4, pady=10, padx=10, sticky="swen", columnspan=2)
    
    # Button for apply colormap
    apply_colormap_btn = ctk.CTkButton(master=additional_functions_frame, text="Aplicar Mapa de Color", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    apply_colormap_btn.grid(row=0, column=6, pady=10, sticky="wn", columnspan=2)
    apply_colormap_btn.configure(command= lambda: apply_colormap(combobox_colormap.get()))
    
    # Button to show the histogram of the images
    histogram_btn = ctk.CTkButton(master=additional_functions_frame, text="Mostrar Histograma", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    histogram_btn.grid(row=0, column=8, pady=10, sticky="wn", columnspan=2)
    histogram_btn.configure(command= lambda: show_histogram())
    
    # Label for the compression factor of the images
    compression_lbl = ctk.CTkLabel(master=additional_functions_frame, text="Factor de Compresión:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    compression_lbl.grid(row=1, column=0, pady=10, padx=15, sticky="swen", columnspan=1)
    
    # Entry for the compression factor of the images
    compression_entry = ctk.CTkEntry(master=additional_functions_frame, width=60, font=GUI_ELEMENTS["fonts"]["button_font"])
    compression_entry.grid(row=1, column=1, pady=10, padx = 10, sticky="swen", columnspan=1)
    
    # Button to compress the images with the DTC algorithm
    compress_btn = ctk.CTkButton(master=additional_functions_frame, text="Comprimir Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    compress_btn.grid(row=1, column=2, pady=10, sticky="wn", columnspan=2)
    compress_btn.configure(command= lambda: compress_images(compression_entry.get()))

    # label for the umbral value for the binarization of the images
    umbral_lbl = ctk.CTkLabel(master=additional_functions_frame, text="Umbral:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    umbral_lbl.grid(row=1, column=6, pady=10, padx=15, sticky="swen", columnspan=1)
    
    # entry for the umbral value for the binarization of the images
    umbral_entry = ctk.CTkEntry(master=additional_functions_frame, width=60, font=GUI_ELEMENTS["fonts"]["button_font"])
    umbral_entry.grid(row=1, column=7, pady=10, padx = 10, sticky="swen", columnspan=1)
    
    # Button to binarize the images
    binarize_btn = ctk.CTkButton(master=additional_functions_frame, text="Binarizar Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    binarize_btn.grid(row=1, column=8, pady=10, sticky="wn", columnspan=2)
    binarize_btn.configure(command= lambda: binarize_images(umbral_entry.get()))
    
    # Ctk label for the tag of the images
    tag_lbl = ctk.CTkLabel(master=image_treatment_frame, text="Etiqueta:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    tag_lbl.grid(row=6, column=0, pady=10, padx=15, sticky="swen", columnspan=1)
    
    # Ctk entry for the tag of the images
    tag_entry = ctk.CTkEntry(master=image_treatment_frame, width=160, font=GUI_ELEMENTS["fonts"]["button_font"])
    tag_entry.grid(row=6, column=1, pady=10, padx = 10, sticky="swen", columnspan=3)
    
    # Button for download images
    download_images_btn = ctk.CTkButton(master=image_treatment_frame, text="Descargar Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    download_images_btn.grid(row=6, column=4, pady=10, sticky="wn", columnspan=2)
    download_images_btn.configure(command= lambda: download_images(tag_entry.get(), filtered_images))
    download_images_btn.configure(state = "disabled")
    
    # Button for download kernel used
    download_kernel_btn = ctk.CTkButton(master=image_treatment_frame, text="Descargar Registro de Kernels", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    download_kernel_btn.grid(row=6, column=6, pady=10, sticky="wn", columnspan=3)
    download_kernel_btn.configure(command= lambda: download_kernel_data())
    download_kernel_btn.configure(state = "disabled")

def check_folder(folder_check):
    """
    Check if the user wants to load images from a folder
    
    Parameters:
    folder_check: Check button to load images from a folder
    """
    global is_folder_img
    if folder_check.get() == 1:
        is_folder_img = True
    else: 
        is_folder_img = False
            
def chargue_images(frame, is_folder):
    """
    Load the images in the application
    This function is called when the user clicks on the load images button
    The images are loaded from a folder or from a file and the frames are created with the images to show them
    
    Parameters:
    frame: Frame where the images will be loaded
    is_folder: Boolean to know if the images are loaded from a folder or from a file
    """
    global images, images_history, apply_kernel_btn, images_labels, filtered_images_labels, kernel_history, filtered_images

    images_history = []

    images_temp = load_images(is_folder)

    print("images temp ",images_temp)

    if (images_temp is None or len(images_temp) == 0) and images is None:
        return
    
    if len(images_temp) == 0 and images is None:
        return
    
    if len(images_temp) == 0 and images is not None:
        print("Images already loaded")
        return
    
    if images_temp is not None:
        images = images_temp
        images_labels = None
        filtered_images_labels = None
        kernel_history = []
        
    num_images_resized = ""    
    for i in range(len(images)):
        if images[i].shape[1] > 500:
            aspect_ratio = 500 / float(images[i].shape[1])
            ne_height = int(images[i].shape[0] * aspect_ratio)
            images[i] = resize_image(images[i], 500, ne_height, "Bicubic")
            num_images_resized += f"{i+1}, "

    if num_images_resized != "":
        num_images_resized = num_images_resized[:-2]
        show_default_error(f"Las imágenes {num_images_resized} han sido redimensionada a un maximum de 500 pixeles de ancho")

    # Show the images in a new frame
    filtered_images = images.copy()
    images_txt = create_images_frame(frame, images)
    create_images_frame(frame, filtered_images, col=6, is_filtered=True)
    apply_kernel_btn.configure(state = "normal")
          
def chargue_filtered_images(frame):
    """
    Load the images with the kernel applied in the application
    This function is called when the user clicks on the load filtered images button
    The images are filtered with the kernel selected and the frames are created with the images to show them
    
    Parameters:
    frame: Frame where the images will be loaded
    """
    
    global filtered_images, images, images_labels, filtered_images_labels, images_history, remove_kernel_btn, kernel_history, download_kernel_btn \
    , kernel_data, download_kernel_btn, next_images_btn, before_images_btn

    if filtered_images is None:
        return
    images = filtered_images.copy()
    
    images_history.append(images)
    
    kernel_history.append(kernel_data)
    print("Kernel added: ", kernel_data)
    
    if len(kernel_history) >= 1:
        download_kernel_btn.configure(state = "normal")
    
    if len(images_history) > 1:
        remove_kernel_btn.configure(state = "normal")
    
    update_show_images(next_images_btn, before_images_btn, 0)
          
def create_images_frame(frame, images, col=0, is_filtered=False):
    """
    Create the images frame in the application
    This frame contains the images to show them
    
    Parameters:
    frame: Frame where the images will be loaded
    images: Images to show in the frame
    col: Column where the images will be loaded
    is_filtered: Boolean to know if the images are filtered or not
    """   
    # Load the images, each image need to be in a label
    global images_labels, actual_first_image, filtered_images_labels, next_images_btn, before_images_btn, filtered_images

    main_frame = ctk.CTkScrollableFrame(master=frame, corner_radius=8, fg_color="#ffffff", width=600 ,height=600, orientation="vertical")
    main_frame.grid(row=5, column=col, sticky="nsw", columnspan=6, pady=10, padx=10)
    main_frame = grid_setup(main_frame)

    image_frame = ctk.CTkScrollableFrame(master=main_frame, corner_radius=8, fg_color="#ffffff", orientation="horizontal")
    image_frame.grid(row=0, column=0, sticky="nsew", columnspan=12, pady=10, padx=10, rowspan=12)
    image_frame = grid_setup(image_frame)
    
    # create a label for each image in the first 10 images, if there are more than 10 images create a button to show next images
    images_labels = [] if images_labels is None else images_labels
    filtered_images_labels = [] if filtered_images_labels is None else filtered_images_labels
    
    last_image = 10 if len(images) > 10 else len(images)
    actual_first_image = 0
    j= 1
    num_images_resized = ""
    height_total = 0
    for i in range(actual_first_image, last_image):
        image = Image.fromarray(images[i])
        photo = ctk.CTkImage(dark_image=image, light_image=image,size=(images[i].shape[1], images[i].shape[0]))
        height_total += images[i].shape[0]
        label = ctk.CTkLabel(master=image_frame, image=photo, text="", compound="center")
        label.grid(row=j, column=0, sticky="nsew", pady=5, columnspan=12)
        if is_filtered:
            filtered_images_labels.append(label)
        else:
            images_labels.append(label)
        j += 1
    
    next_images_btn = ctk.CTkButton(master=image_frame, text=">", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=80, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    next_images_btn.grid(row=0, column=1, sticky="nsew", padx=2)
    next_images_btn.configure(command= lambda: update_show_images(next_images_btn, before_images_btn, actual_first_image+10))
    
    before_images_btn = ctk.CTkButton(master=image_frame, text="<", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=80, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    before_images_btn.grid(row=0, column=0, sticky="nsew", padx=0)
    before_images_btn.configure(command= lambda: update_show_images(next_images_btn, before_images_btn, actual_first_image-10))

    height_total += 100
    image_frame.configure(height=height_total)
    
    if last_image == len(images) and last_image == len(filtered_images):
        next_images_btn.configure(state="disabled")
    else:
        next_images_btn.configure(state="normal")
        
    if actual_first_image == 0:
        before_images_btn.configure(state="disabled")
    else:
        before_images_btn.configure(state="normal")     
    
def update_show_images(next_images_btn, before_images_btn, actual_first_img):
    """
    Update the images to show in the frame
    This function is called when the user clicks on the next or before images button
    
    Parameters:
    next_images_btn: Button to show the next images
    before_images_btn: Button to show the before images
    actual_first_img: Index of the first image to show
    """
    global images_labels, images, actual_first_image, filtered_images_labels, filtered_images

    actual_first_image = actual_first_img
    last_image = (actual_first_img+10) if len(images) > actual_first_img + 10 else len(images)
    j = 0
    for i in range(actual_first_img,last_image):
        width = images[i].shape[1]
        height = images[i].shape[0]
        image_temp = Image.fromarray(images[i])
        photo = ctk.CTkImage(dark_image=image_temp, light_image=image_temp, size=(width, height))
        images_labels[j].configure(image=photo)
        
        width = filtered_images[i].shape[1]
        height = filtered_images[i].shape[0]
        image_f = Image.fromarray(filtered_images[i])
        photo_f = ctk.CTkImage(dark_image=image_f, light_image=image_f, size=(width, height))
        filtered_images_labels[j].configure(image=photo_f)      
        
        j += 1

    # complete the for when is less than 10 images to delete the rest of the images in the labels
    for i in range(j, len(images_labels)):
        images_labels[i].configure(image=None)
        filtered_images_labels[i].configure(image=None)

    if last_image == len(images):
        next_images_btn.configure(state="disabled")
    else:
        next_images_btn.configure(state="normal")
        
    if actual_first_image == 0:
        before_images_btn.configure(state="disabled")
    else:
        before_images_btn.configure(state="normal")

def clear_frame(frame):
    """
    Clear the frame
    """
    for widget in frame.winfo_children():
        widget.destroy()

def apply_sel_kernel(kernel_name, stride=1, padding=0, percentaje = 0.5 , sub_info = None):
    """
    Apply the kernel selected to the images
    This function is called when the user clicks on the apply kernel button

    Parameters:
    kernel_name: Name of the kernel to apply
    stride: Stride value to apply the kernel
    padding: Padding value to apply the kernel
    percentaje: Percentaje value to apply the kernel
    sub_info: Additional information to apply the kernel
    
    """
    global images, images_history, filtered_images, next_images_btn, before_images_btn, actual_first_image, image_treatment_btn_2\
        , download_images_btn, kernel_data

    mod_kernels = ["rgb_to_bgr", "bgr_to_rgb", "gray_to_rgb", "gray_to_bgr", "rgb_to_gray", "bgr_to_gray"]
    percentaje_kernels = ["less_red", "less_green", "less_blue", "more_red", "more_green", "more_blue"] 
    special_functions = ["rotate_image", "apply_colormap", "compress_image", "binarize_image"]
    type_data = ""
    kernel = None
    
    print("Applying kernel: ", kernel_name)
    
    if images is None:
        show_default_error("No hay imágenes cargadas para aplicar el proceso")
    
    if stride < 1:
        show_default_error("El stride debe ser mayor o igual a 1")
        return
    
    if kernel_name in mod_kernels:
        type_data = "color modification"
        kernel = get_kernel(kernel_name)
        kernel_data = {
            "name": kernel_name,
            "percentaje": percentaje,
            "type": type_data
        }
    elif kernel_name in percentaje_kernels:
        type_data = "color percentaje"
        kernel = get_kernel(kernel_name, percentaje)
        kernel_data = {
            "name": kernel_name,
            "percentaje": percentaje,
            "type": type_data
        }
    elif kernel_name in special_functions:
        type_data = "special function"
        kernel = get_kernel(kernel_name)
        kernel_data = {
            "name": kernel_name,
            "sub_info": sub_info,
            "type": type_data
        }
        
    elif kernel_name == "resize_image":
        type_data = "resize"
        kernel = get_kernel(kernel_name)
        kernel_data = {
            "name": kernel_name,
            "sub_info": sub_info,
            "type": type_data,
        } 
    else:
        type_data = "kernel"
        kernel = get_kernel(kernel_name)
        kernel_data = {
            "name": kernel_name,
            "stride": stride,
            "padding": padding,
            "type": type_data
        }
        
    if images_history is None:
        images_history = [] 

    # add the images to the history
    if len(images_history) == 0:
        images_history.append(images)

    # initialize the filtered images with arrays
    filtered_images = [[] for i in range(len(images))]

    # Crear un pool de procesos
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Enviar las tareas al pool de procesos, junto con el índice para asegurar el orden
        futures = None
        if type_data == "kernel":
            futures = {executor.submit(apply_kernel, images[i], kernel, padding, stride): i for i in range(len(images))}
        elif type_data == "color percentaje":
            futures = {executor.submit(kernel, images[i], percentaje): i for i in range(len(images))}
        elif type_data == "color modification":
            futures = {executor.submit(kernel, images[i]): i for i in range(len(images))}
        elif type_data == "resize":
            futures = {executor.submit(kernel, images[i], sub_info["width"], sub_info["height"], sub_info["interpolation"]): i for i in range(len(images))}
        elif type_data == "special function":
            data = 0
            data = sub_info["colormap"] if kernel_name == "apply_colormap" else data
            data = sub_info["compression"] if kernel_name == "compress_image" else data
            data = sub_info["umbral"] if kernel_name == "binarize_image" else data
            data = sub_info["angle"] if kernel_name == "rotate_image" else data
            futures = {executor.submit(kernel, images[i], data): i for i in range(len(images))}
        # Recuperar los resultados en el orden original
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]  # Recuperamos el índice de la imagen
            filtered_images[index] = future.result()  # Asignamos el resultado en el índice correcto
    
        update_show_images(next_images_btn, before_images_btn, actual_first_image)
    
    image_treatment_btn_2.configure(state = "normal")
    download_images_btn.configure(state = "normal")

def remove_kernel():
    """
    Remove the kernel applied to the images
    This function is called when the user clicks on the remove kernel button
    The images are restored to the last state before the kernel was applied
    """
    global images_history, images, filtered_images, remove_kernel_btn, download_images_btn, kernel_history, download_kernel_btn
    
    if len(images_history) == 1:
        remove_kernel_btn.configure(state="disabled")
    
    if len(images_history) >= 1:
        images_history.pop()
        images = images_history[-1]
        filtered_images = images
        update_show_images(next_images_btn, before_images_btn, actual_first_image)
        
    if filtered_images is None or len(filtered_images) == 0:
        remove_kernel_btn.configure(state="disabled")
        download_images_btn.configure(state="disabled")
        
    if len(kernel_history) >= 1:
        kernel_history.pop()
    
    if len(kernel_history) < 1:
        download_kernel_btn.configure(state="disabled")
    
def download_kernel_data():
    """
    Download the kernel data applied to the images
    This function is called when the user clicks on the download kernel button
    """
    global kernel_history, kernel_data
    
    if len(kernel_history) == 0:
        print("No kernels to download")
        return
    
    kernel_history.append(kernel_data)
    
    kernels_json = {
        "kernels": kernel_history
    }
    
    filename = "kernel_data"
    save_json(kernels_json, filename)
    
    print(f"Kernel data saved in {filename}.json")

def resize_images(width, height, interpolation):
    """
    Resize the images in the application to train them
    This function is called when the user clicks on the resize images button

    Parameters:
    width: Width of the images
    height: Height of the images
    interpolation: Interpolation method to resize the images
    """
    
    global train_images_data, treated_images_frame, txt_treated_train_images, txt_plained_train_images
    
    print(f"Resizing images to width: {width}, height: {height}, interpolation: {interpolation}")
    
    if width == "" or height == "" or interpolation == "":
        show_info_resize_images("Debe ingresar el ancho, alto y la interpolación")
        return
    
    if not width.isdigit() or not height.isdigit():
        show_info_resize_images("El ancho y el alto deben ser números enteros")
        return
    
    if txt_treated_train_images is None:
        txt_treated_train_images = ctk.CTkTextbox(master=plained_images_frame, width=600, fg_color="#ffffff")
        txt_treated_train_images.grid(row=1, column=0, pady=5, padx=2, sticky="nsew")
                
    txt_treated_train_images.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    txt_treated_train_images.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    txt_treated_train_images.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="#70600f")
    
    txt_treated_train_images.delete("1.0", "end")
    
    for category in train_images_data:
        images = category["images"]
        label = category["label"]
        status = category["status"]
        
        if status == "treated":
            images = [resize_image(image, int(width), int(height), interpolation) for image in images] 
            category["images"] = images
            txt_treated_train_images.insert("end", f"{label}\n", "title")
            txt_treated_train_images.insert("end", f"Número de imágenes: {len(images)}\n", "subtitle")
            txt_treated_train_images.insert("end", f"Forma de las imágenes: {images[0].shape}\n", "subtitle")
            # insert separator line
            sp_line = "-"*25
            txt_treated_train_images.insert("end", f"{sp_line}\n", "pattern")
     
def resize_treatment_images(width, height, interpolation): 
    """
    Resize the images in the application to treat them
    This function is called when the user clicks on the resize images button
    """
    print(f"Resizing images to width: {width}, height: {height}, interpolation: {interpolation}")
    
    if width == "" or height == "" or interpolation == "":
        show_info_resize_images("Debe ingresar el ancho, alto y la interpolación")
        return
    
    if not width.isdigit() or not height.isdigit():
        show_info_resize_images("El ancho y el alto deben ser números enteros")
        return
    
    apply_sel_kernel(kernel_name="resize_image", sub_info={"width": int(width), "height": int(height), "interpolation": interpolation})
     
def apply_colormap(colormap):
    """
    Apply the colormap to the images in the application to treat them
    This function is called when the user clicks on the apply colormap button
    
    Parameters:
    colormap: Colormap name to apply to the images
    """
    if colormap == "":
        show_default_error("Debe seleccionar un mapa de colores")
        return
    
    apply_sel_kernel(kernel_name="apply_colormap", sub_info={"colormap": colormap})
   
def compress_images(compression):
    """
    Compress the images in the application to treat them
    This function is called when the user clicks on the compress images button
    
    Parameters:
    compression: Level of compression to apply to the images
    """
    if compression == "":
        show_default_error("Debe seleccionar un nivel de compresión")
        return
    
    try:
        compression = float(compression)
    except:
        show_default_error("El nivel de compresión debe ser un número decimal")
        return
    
    apply_sel_kernel(kernel_name="compress_image", sub_info={"compression": compression})
    
def binarize_images(umbral):
    """
    Binarize the images in the application to treat them
    This function is called when the user clicks on the binarize images button
    
    Parameters:
    umbral: Umbral to binarize the images
    """
    if umbral == "":
        show_default_error("Debe ingresar un umbral")
        return
    
    if not umbral.isdigit():
        show_default_error("El umbral debe ser un número entero")
        return
    
    if not int(umbral) in range(0, 255):
        show_default_error("El umbral debe estar en el rango de 0 a 255")
        return
    
    apply_sel_kernel(kernel_name="binarize_image", sub_info={"umbral": int(umbral)})
   
def rotate_images(angle):
    """
    Rotate the images in the application to treat them
    This function is called when the user clicks on the rotate images button
    
    Parameters:
    angle: Angle to rotate the images in degrees as an integer and in the range of -360 to 360
    """
    if angle == "":
        show_default_error("Debe ingresar un ángulo de rotación")
        return
    
    if not angle.isdigit():
        show_default_error("El ángulo de rotación debe ser un número entero")
        return
    
    if int(angle) not in range(-360, 360):
        show_default_error("El ángulo de rotación debe estar en el rango de 0 a 360")
        return
    
    apply_sel_kernel(kernel_name="rotate_image", sub_info={"angle": int(angle)})   

def show_histogram():
    """
    Show the histogram of the images in the application
    This function is called when the user clicks on the show histogram button
    """
    global filtered_images, info_window
    
    histogram_images_data = []
    
    for image in filtered_images:
        histogram_images_data.append(get_histogram_data(image))
    
    figure = plot_histograms(histogram_images_data)
    
    # Top level window for the histogram
    info_window = ctk.CTkToplevel(master=main_window, fg_color=GUI_ELEMENTS["colors"]["background"])
    info_window.title("Histograma de las Imágenes")
    info_window.geometry("800x600")
    info_window.resizable(False, False)
    info_window = grid_setup(info_window)
    icon_path = get_resource_path("Resources/brand_logo.ico")
    info_window.iconbitmap(icon_path)
    info_window.attributes("-topmost", True)
    info_window.bind("esc", lambda e: info_window.destroy())
    
    # Scrollable frame to put the figure canvas
    frame = ctk.CTkScrollableFrame(master=info_window, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    frame.grid(row=0, column=0, sticky="nsew", columnspan=12, rowspan=12)
    frame = grid_setup(frame)
    
    # Only one figure, create one canvas
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", columnspan=12, rowspan=12)
    
def resize_frame_creation(frame, for_training = False, for_treatment = False):
    """
    Create the frame for the resize images in the application
    This frame contains the inputs for the width and height of the images and the interpolation method
    
    Parameters:
    frame: Frame where the resize frame will be loaded
    for_training: Boolean to know if the frame is for training the images
    for_treatment: Boolean to know if the frame is for treating the images
    """
    
    global resize_images_btn, width_entry, height_entry, interpolation_combobox
    # Input for the width of the images
    width_lbl = ctk.CTkLabel(master=frame, text="Ancho:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    width_lbl.grid(row=0, column=0, pady=10, padx=15, sticky="sen")
    
    width_entry = ctk.CTkEntry(master=frame, width=120, font=("Arial", 14, "bold"))
    width_entry.grid(row=0, column=1, pady=10, padx = 10, sticky="swn")
    
    # Input for the height of the images
    height_lbl = ctk.CTkLabel(master=frame, text="Alto:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    height_lbl.grid(row=0, column=2, pady=10, padx=15, sticky="sen")
    
    height_entry = ctk.CTkEntry(master=frame, width=120, font=("Arial", 14, "bold"))
    height_entry.grid(row=0, column=3, pady=10, padx = 10, sticky="swn")    
    
    # Interpolation options
    interpolations = ["Nearest", "Bilinear", "Bicubic", "Area-based", "Lanczos", "Spline"]
    
    interpolation_label = ctk.CTkLabel(master=frame, text="Interpolación:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    interpolation_label.grid(row=0, column=4, pady=10, padx=15, sticky="sen")
    
    interpolation_combobox = ctk.CTkComboBox(master=frame, values=interpolations, width=200, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], fg_color=GUI_ELEMENTS["colors"]["foreground"], text_color="#0F1010", state="readonly")
    interpolation_combobox.grid(row=0, column=5, pady=10, padx = 10, sticky="wn", columnspan=2)
    
    # Button for resize the images
    if not for_training:
        resize_images_btn = ctk.CTkButton(master=frame, text="Redimensionar Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
        resize_images_btn.grid(row=0, column=7, pady=10, padx = 10, sticky="wn", columnspan=2)
        resize_images_btn.configure(command= lambda: resize_images(width_entry.get(), height_entry.get(), interpolation_combobox.get()))
        resize_images_btn.configure(state="disabled") 
    
    if for_treatment:
        resize_images_btn.configure(command= lambda: resize_treatment_images(width_entry.get(), height_entry.get(), interpolation_combobox.get()))
        resize_images_btn.configure(state="normal")
    
    return frame
# endregion

# region Image Training Frame  
def image_training_frame_creation(master_window):
    """
    Create the image training frame in the application
    This frame contains the options to train the images
    The user can load the images, apply a kernel series and train the images
    
    Parameters:
    master_window: Frame where the image training frame will be loaded
    """
    global chargued_tags_frame, treated_images_frame, plained_images_frame, kernel_series, images_train_status_lbl,\
        kernel_combobox, load_kernel_btn, pre_treated_check, status_kernel_lbl, image_treatment_btn, info_btn, load_images_btn,\
        actual_tags, plained_images_btn, finish_data_btn, image_training_frame, train_images_data, label_categories,\
        txt_initial_train_images, txt_treated_train_images, txt_plained_train_images
     
    txt_initial_train_images, txt_treated_train_images, txt_plained_train_images = None, None, None
    train_images_data, label_categories = [], None
    actual_tags = []
    
    image_training_frame = ctk.CTkScrollableFrame(master=master_window, corner_radius=0, fg_color=GUI_ELEMENTS["colors"]["background"],scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    image_training_frame.grid(row=0, column=2, sticky="nsew", columnspan=12, rowspan=12)
    image_training_frame = grid_setup(image_training_frame)
    
    # Title of the image training
    title_txt = "Entrenamiento de Imágenes"
    title = ctk.CTkLabel(master=image_training_frame, text=title_txt, font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title.grid(row=0, column=0, pady=10, sticky="new", columnspan=11)
    
    # Button for information about the process
    info_btn = ctk.CTkButton(master=image_training_frame, text="Ayuda", fg_color=GUI_ELEMENTS["colors"]["info_button"], width=100, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["info_button_hover"], text_color=GUI_ELEMENTS["colors"]["foreground"])
    info_btn.grid(row=0, column=11, pady=10, sticky="en")
    info_btn.configure(command= lambda: show_info_image_training())
    
    # Label for the tag of the images
    tag_lbl = ctk.CTkLabel(master=image_training_frame, text="Etiqueta:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    tag_lbl.grid(row=1, column=0, pady=10, padx=15, sticky="sen")
    
    # Entry for the tag of the images
    tag_entry = ctk.CTkEntry(master=image_training_frame, width=200, font=("Arial", 14, "bold"))
    tag_entry.grid(row=1, column=1, pady=10, padx = 10, sticky="swen", columnspan=2)
    
    # Button for load a category of images
    load_images_btn = ctk.CTkButton(master=image_training_frame, text="Cargar Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    load_images_btn.grid(row=1, column=3, pady=10, sticky="wn", columnspan=2)
    load_images_btn.configure(command= lambda: load_images_category(tag_entry.get()))
    
    # State label for the load images process
    images_train_status_lbl = ctk.CTkLabel(master=image_training_frame, text="", font=("Arial", 18, "bold"), text_color="red")
    images_train_status_lbl.grid(row=1, column=5, pady=10, sticky="n", columnspan=3)
    
    # Image Treatment subtitle
    image_treatment_subtitle = ctk.CTkLabel(master=image_training_frame, text="Tratamiento de Imágenes", font=("Arial", 18, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], justify="left")
    image_treatment_subtitle.grid(row=2, column=0, pady=20, padx=10 ,sticky="nw", columnspan=12)
    
    # Combobox for the selection of the kernel series
    kernel_combobox = ctk.CTkComboBox(master=image_training_frame, values=kernel_series, width=200, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], fg_color=GUI_ELEMENTS["colors"]["foreground"], text_color="#0F1010")
    kernel_combobox.grid(row=3, column=0, pady=10, padx = 10, sticky="wn", columnspan=2)
    kernel_combobox.configure(state="disabled", command = select_kernel_series)
    
    # Button for load a kernel series
    load_kernel_btn = ctk.CTkButton(master=image_training_frame, text="Cargar Serie de Kernels", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    load_kernel_btn.grid(row=3, column=2, pady=10, sticky="wn", columnspan=3)
    load_kernel_btn.configure(command= lambda: load_kernel_series())
    load_kernel_btn.configure(state="disabled")
    
    # Checkbox for the selection of the pre-treated images
    pre_treated_check = ctk.CTkCheckBox(master=image_training_frame, text="Imágenes Pretratadas", font=("Arial", 14, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    pre_treated_check.grid(row=3, column=5, pady=10, padx=15, sticky="wsn", columnspan=1)
    pre_treated_check.configure(command= lambda: check_pre_treated(pre_treated_check))
    pre_treated_check.configure(state="disabled")
    
    # Status label for the series of kernels
    status_kernel_lbl = ctk.CTkLabel(master=image_training_frame, text="", font=("Arial", 14, "bold"), text_color="red", wraplength=200)
    status_kernel_lbl.grid(row=3, column=6, pady=10, sticky="wn", columnspan=3)
    
    # Button for start the image treatment process
    image_treatment_btn = ctk.CTkButton(master=image_training_frame, text="Empezar Tratamiento de Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    image_treatment_btn.grid(row=3, column=10, padx = 5 ,pady=10, sticky="en", columnspan=2)
    image_treatment_btn.configure(command= lambda: start_treatment_images())
    image_treatment_btn.configure(state="disabled")

    # Subtitle for flatten process
    flatten_subtitle = ctk.CTkLabel(master=image_training_frame, text="Aplanamiento de Imágenes", font=("Arial", 18, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], justify="left")
    flatten_subtitle.grid(row=4, column=0, pady=20, padx=10 ,sticky="nw", columnspan=12)
    
    # create the frame for the resize images
    resize_images_frame = ctk.CTkFrame(master=image_training_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    resize_images_frame = resize_frame_creation(resize_images_frame)
    resize_images_frame.grid(row=5, column=0, sticky="nsew", columnspan=10, pady=10, padx=10)

    # Button for flatten the images
    plained_images_btn = ctk.CTkButton(master=image_training_frame, text="Aplanar Imágenes", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    plained_images_btn.grid(row=5, column=10, padx =15 ,pady=10, sticky="e", columnspan=2)
    plained_images_btn.configure(command= lambda: plained_images())
    plained_images_btn.configure(state="disabled")

    # Subtitle for data information
    data_info_subtitle = ctk.CTkLabel(master=image_training_frame, text="Información de los Datos", font=("Arial", 18, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], justify="left")
    data_info_subtitle.grid(row=7, column=0, pady=20, padx=10 ,sticky="nw", columnspan=12)

    # frame for the information of the images
    data_info_frame = ctk.CTkFrame(master=image_training_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    data_info_frame = grid_setup(data_info_frame)
    data_info_frame.grid(row=8, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    
    # Frame for the tags of the images
    chargued_tags_frame = default_frame_img_training(data_info_frame, "Categorías Cargadas Sin Procesar", 0, 0)   
    
    # Frame for the images already treated
    treated_images_frame = default_frame_img_training(data_info_frame, "Categorías Pretratadas", 0, 4)
    
    # Frame for the images already plained
    plained_images_frame = default_frame_img_training(data_info_frame, "Categorías Planas", 0, 8)
    
    #Button for finish the data input process
    finish_data_btn = ctk.CTkButton(master=image_training_frame, text="Finalizar Ingreso de Datos", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    finish_data_btn.grid(row=9, column=10, padx =15 ,pady=10, sticky="e", columnspan=3)
    finish_data_btn.configure(command= lambda: finish_data_input())
    finish_data_btn.configure(state="disabled")

def load_images_category(tag_name):
    """
    Load the images of a category in the application to treat them
    This function is called when the user clicks on the load images button
    
    Parameters:
    tag_name: Name of the category of the images
    """
    global images, images_train_status_lbl, chargued_tags_frame, train_images_data, txt_initial_train_images,\
        kernel_combobox, load_kernel_btn, pre_treated_check, image_treatment_btn, load_images_btn, actual_tags,\
        status_kernel_lbl

    if tag_name == "" or tag_name is None:
        images_train_status_lbl.configure(text="Debe ingresar una etiqueta", text_color="red")
        return

    if tag_name in actual_tags:
        images_train_status_lbl.configure(text="La etiqueta ya fue ingresada", text_color="red")
        return

    # Load the images
    images = load_images(is_folder=True)    

    if images is None or len(images) == 0:
        images_train_status_lbl.configure(text="Error al cargar las imágenes", text_color="red")
        return
    
    actual_tags.append(tag_name)
    
    category_data = {
        "label": tag_name,
        "images": images,
        "status": "loaded"
    }
    
    print("Category data: ", category_data)
    
    train_images_data.append(category_data)
    
    if txt_initial_train_images is None:
        txt_initial_train_images = ctk.CTkTextbox(master=chargued_tags_frame, width=600, fg_color="#ffffff")
        txt_initial_train_images.grid(row=1, column=0, pady=5, padx=2, sticky="nsew")
                
    txt_initial_train_images.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    txt_initial_train_images.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    txt_initial_train_images.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="#70600f")
    
    txt_initial_train_images.delete("1.0", "end")

    txt_initial_train_images.insert("end", f"{tag_name}\n", "title")
    txt_initial_train_images.insert("end", f"Número de imágenes: {len(images)}\n", "subtitle")
    txt_initial_train_images.insert("end", f"Forma de las imágenes: {images[0].shape}\n", "subtitle")
    # insert separator line
    sp_line = "-"*25
    txt_initial_train_images.insert("end", f"{sp_line}\n", "pattern")
    
    images_train_status_lbl.configure(text=f"Imágenes {tag_name} Cargadas", text_color="green")
    
    kernel_combobox.configure(state="readonly")
    load_kernel_btn.configure(state="normal")
    pre_treated_check.configure(state="normal")    
    image_treatment_btn.configure(state="normal")
    load_images_btn.configure(state="disabled")
    status_kernel_lbl.configure(text="", text_color="red")

def select_kernel_series(kernel_series):
    """
    Select the kernel series to apply to the images
    This function is called when the user selects a kernel series in the combobox
    kerlen_series is a json file with the kernel series to apply
    
    Parameters:
    kernel_series: Name of the kernel series to apply
    """
    global selected_kernel_series, kernel_combobox, load_kernel_btn, status_kernel_lbl
    
    
    if kernel_series is None or kernel_series == "None" or kernel_series == "":
        status_kernel_lbl.configure(text="Debe seleccionar una serie de kernels", text_color="red")
        return
    
    filename = "cirujano_kernels" if kernel_series == "Pez Cirujano" else None
    filename = "trucha_kernels" if kernel_series == "Trucha Arcoíris" else filename 
    selected_kernel_series = load_json(filename=f"kernels_series/{filename}")
    
    try:
        if selected_kernel_series["kernels"] is None or len(selected_kernel_series["kernels"]) == 0:    
            status_kernel_lbl.configure(text="No hay kernels en la serie", text_color="red")
            return 
    except Exception as e:
        print(e)
        status_kernel_lbl.configure(text="Error al cargar la serie de kernels", text_color="red")
        return
    
    status_kernel_lbl.configure(text=f"{kernel_series} Cargado", text_color="green")
    
def load_kernel_series():
    """
    Load the kernel series in the application to apply them to the images
    This function is called when the user clicks on the load kernel series button
    """
    global selected_kernel_series, status_kernel_lbl
    
    # load the kernel series from the json file
    selected_kernel_series, filename = load_json() 

    if select_kernel_series is None:
        status_kernel_lbl.configure(text="Error al cargar la serie de kernels", text_color="red")
        return

    try:
        if "kernels" in selected_kernel_series is None or len("kernels") in selected_kernel_series == 0:    
            status_kernel_lbl.configure(text="No hay kernels en la serie", text_color="red")
            return 
    except Exception as e:
        print(e)
        status_kernel_lbl.configure(text="Error al cargar la serie de kernels", text_color="red")
        return
    
    status_kernel_lbl.configure(text=f"{filename} Cargada", text_color="green")

def check_pre_treated(pre_treated_check):
    """
    Check if the images are pre-treated or not in the application to apply the kernel series
    This function is called when the user clicks on the pre-treated images checkbox    
    
    Parameters:
    pre_treated_check: Checkbox to know if the images are pre-treated or not                                                               
    """
    global selected_kernel_series, kernel_combobox, load_kernel_btn, status_kernel_lbl
    
    if pre_treated_check.get() == 1:
        kernel_combobox.configure(state="disabled")
        load_kernel_btn.configure(state="disabled")
        selected_kernel_series = "None"
        status_kernel_lbl.configure(text="", text_color="green")  
    else:
        kernel_combobox.configure(state="readonly")
        load_kernel_btn.configure(state="normal")
        selected_kernel_series = None

    status_kernel_lbl.configure(text="", text_color="red")

def start_treatment_images():
    """
    Start the treatment of the images in the application
    This function is called when the user clicks on the start treatment images button
    The images are treated with the selected kernel series
    """
    
    global selected_kernel_series, train_images_data, treated_images_frame, txt_treated_train_images, txt_initial_train_images,\
        status_kernel_lbl, load_images_btn, image_treatment_btn, pre_treated_check, kernel_combobox, plained_images_btn, resize_images_btn

    if txt_treated_train_images is None:
        print("Creating treated images frame")
        txt_treated_train_images = ctk.CTkTextbox(master=treated_images_frame, width=600, fg_color="#ffffff")
        txt_treated_train_images.grid(row=1, column=0, pady=5, padx=2, sticky="nsew")
                
    txt_treated_train_images.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    txt_treated_train_images.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    txt_treated_train_images.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="#70600f")

    txt_treated_train_images.delete("1.0", "end")
    
    if selected_kernel_series is None:
        status_kernel_lbl.configure(text="serie de kernels no seleccionada", text_color="red")
        return
    
    print(f"Selected kernel series: {selected_kernel_series}")
    
    for category in train_images_data:
        images = category["images"]
        label = category["label"]
        status = category["status"]

        print(f"Processing category: {label} with status: {status}")

        if status == "loaded" and selected_kernel_series == "None":
            category["status"] = status = "treated"
        
        if status == "loaded" and selected_kernel_series != "None":
            # apply the selected kernel series to the images
            category["images"] = apply_kernel_series(images, selected_kernel_series)
            category["status"] = status = "treated"
            images = category["images"]
        
        if status == "treated":
            txt_treated_train_images.insert("end", f"{label}\n", "title")
            txt_treated_train_images.insert("end", f"Número de imágenes: {len(images)}\n", "subtitle")
            txt_treated_train_images.insert("end", f"Forma de las imágenes: {images[0].shape}\n", "subtitle")
            # insert separator line
            sp_line = "-"*25
            txt_treated_train_images.insert("end", f"{sp_line}\n", "pattern")
            
    txt_initial_train_images.delete("1.0", "end")
    load_images_btn.configure(state="normal")
    image_treatment_btn.configure(state="disabled")
    status_kernel_lbl.configure(text="Imágenes Tratadas", text_color="green")
    pre_treated_check.deselect()
    pre_treated_check.configure(state="disabled")
    kernel_combobox.configure(state="disabled")
    plained_images_btn.configure(state="normal")
    resize_images_btn.configure(state="normal")
    load_kernel_btn.configure(state="disabled")
    
def apply_kernel_series(images, selected_kernel_series):
    """
    Apply the kernel series to the images
    This function is called when the user clicks on the start treatment images button
    The images are treated with the selected kernel series

    Parameters:
    images: Images to treat
    selected_kernel_series: Kernel series to apply
    """
    
    global main_window
    mod_kernels = ["rgb_to_bgr", "bgr_to_rgb", "gray_to_rgb", "gray_to_bgr", "rgb_to_gray", "bgr_to_gray"]
    percentaje_kernels = ["less_red", "less_green", "less_blue", "more_red", "more_green", "more_blue"]
    special_functions = ["rotate_image", "apply_colormap", "compress_image", "binarize_image"]
            
    filtered_images = images.copy()
    kernel = None
    for kernel_json in selected_kernel_series["kernels"]:
        
        kernel_name = kernel_json["name"]
        type_data = kernel_json["type"]
        sub_info = kernel_json["sub_info"] if "sub_info" in kernel_json else None
        percentaje = kernel_json["percentaje"] if "percentaje" in kernel_json else 0
        padding = kernel_json["padding"] if "padding" in kernel_json else 0
        stride = kernel_json["stride"] if "stride" in kernel_json else 1


        if kernel_name in percentaje_kernels:
            kernel = get_kernel(kernel_name, percentaje)
        else:
            kernel = get_kernel(kernel_name)
        
        print(str("-" * 50))
        print(f"Applying kernel: \n {json.dumps(kernel_json, indent=4)}")
        show_treatment_info(kernel_json, len(images))
        
        main_window.update()
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Enviar las tareas al pool de procesos, junto con el índice para asegurar el orden
            futures = None
            if type_data == "kernel":
                futures = {executor.submit(apply_kernel, filtered_images[i], kernel, padding, stride): i for i in range(len(filtered_images))}
            elif type_data == "color percentaje":
                futures = {executor.submit(kernel, filtered_images[i], percentaje): i for i in range(len(filtered_images))}
            elif type_data == "color modification":
                futures = {executor.submit(kernel, filtered_images[i]): i for i in range(len(filtered_images))}
            elif type_data == "resize":
                futures = {executor.submit(kernel, filtered_images[i], sub_info["width"], sub_info["height"], sub_info["interpolation"]): i for i in range(len(filtered_images))}
            elif type_data == "special function":
                data = 0
                data = sub_info["colormap"] if kernel_name == "apply_colormap" else data
                data = sub_info["compression"] if kernel_name == "compress_image" else data
                data = sub_info["umbral"] if kernel_name == "binarize_image" else data
                data = sub_info["angle"] if kernel_name == "rotate_image" else data
                futures = {executor.submit(kernel, filtered_images[i], data): i for i in range(len(filtered_images))}
            # Recuperar los resultados en el orden original
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]  # Recuperamos el índice de la imagen
                filtered_images[index] = future.result()  # Asignamos el resultado en el índice correcto
                
    return filtered_images

def plained_images():
    """
    Flatten the images in the application to train them
    This function is called when the user clicks on the plained images button
    """
    
    global train_images_data, plained_images_frame, txt_plained_train_images, txt_treated_train_images, txt_initial_train_images,\
    finish_data_btn, plained_images_btn
    
    common_resolution = None
    
    qty_treated = len([category for category in train_images_data if category["status"] == "treated"])
    if qty_treated == 0:
        show_info_resize_images("No hay categorías tratadas", main_info=False)
        plained_images_btn.configure(state="disabled")
        return
    
    
    for category in train_images_data:
        images = category["images"]
        status = category["status"]
        if status == "treated":
            common_resolution = images[0].shape
            break
        
    # Verify if the images have the same resolution
    for category in train_images_data:
        if category["status"] != "treated":
            continue
        
        for image in category["images"]:
            if image.shape != common_resolution:
                show_info_resize_images()
                return

    if txt_plained_train_images is None:
        txt_plained_train_images = ctk.CTkTextbox(master=plained_images_frame, width=600, fg_color="#ffffff")
        txt_plained_train_images.grid(row=1, column=0, pady=5, padx=2, sticky="nsew")
        
    txt_plained_train_images.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    txt_plained_train_images.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    txt_plained_train_images.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="#70600f")
    
    txt_plained_train_images.delete("1.0", "end")   
    
    for category in train_images_data:
        images = category["images"]
        label = category["label"]
        status = category["status"]
        
        if status == "treated":
            print(f"Flattening images for category: {label}")
            category["images"] = images = [plain_image(image) for image in images]
            category["status"] = status = "flattened"
        
        if status == "flattened":
            txt_plained_train_images.insert("end", f"{label}\n", "title")
            txt_plained_train_images.insert("end", f"Número de imágenes: {len(images)}\n", "subtitle")
            txt_plained_train_images.insert("end", f"Forma de las imágenes: ({len(images[0])})\n", "subtitle")
            # insert separator line
            sp_line = "-"*25
            txt_plained_train_images.insert("end", f"{sp_line}\n", "pattern")
            
    txt_treated_train_images.delete("1.0", "end") 
    txt_initial_train_images.delete("1.0", "end")
    
    if len(train_images_data) >= 2:
        finish_data_btn.configure(state="normal")  
            
def default_frame_img_training(master,title,row,col):
    """
    Create a default frame for the image training process in the application
    This frame contains the information of the images loaded, treated and plained

    Parameters:
    master: Frame where the default frame will be loaded
    title: Title of the default frame
    row: Row where the default frame will be loaded
    col: Column where the default frame will be loaded
    """
    
    default_frame = ctk.CTkScrollableFrame(master=master, corner_radius=4, fg_color="#ffffff", height=200)
    default_frame.grid(row=row, column=col, sticky="nsew", columnspan=4, padx=20)
    default_frame = grid_setup(default_frame)
    
    # Title of the default frame
    title_label = ctk.CTkLabel(master=default_frame, text=title, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["background"], anchor="center", justify="center")
    title_label.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)
    
    return default_frame

def finish_data_input():
    """
    Finish the data input process in the application to train the images
    This function is called when the user clicks on the finish data input button
    This function disables the buttons for the data input process
    """
    global load_images_btn, image_treatment_btn, pre_treated_check, kernel_combobox, load_kernel_btn, plained_images_btn,\
        resize_images_btn, finish_data_btn, image_training_frame, train_images_data, txt_initial_train_images, txt_treated_train_images\

    if len(train_images_data) < 2:
        print("Debe cargar al menos dos categorías de imágenes")
        show_info_resize_images("Debe cargar al menos dos categorías de imágenes", False)
        return

    load_images_btn.configure(state="disabled")
    image_treatment_btn.configure(state="disabled")
    pre_treated_check.configure(state="disabled")
    kernel_combobox.configure(state="disabled")
    load_kernel_btn.configure(state="disabled")
    plained_images_btn.configure(state="disabled")
    resize_images_btn.configure(state="disabled")
    finish_data_btn.configure(state="disabled")
        
    train_frame_creation(image_training_frame, num_excercise=0, row=10, is_for_image=True)
# endregion

# region Image Test Frame
def test_image_frame_creation(master):
    """
    Create the frame for the test images in the application
    This frame contains the buttons to load the image, add and delete categories, and the checkbox for the default categories
    also contains the scrollable frame for the categories of the images, and options for the resize of the images
    This frame shows the information of the process of the test images and the image processed and the result of the test
    
    Parameters:
    master: Frame where the test image frame will be loaded
    """
    global data_test_json, weights_test_json, num_excercise, data_train_json, image_frame, treated_image_frame, image_status_lbl, \
        categories_frame, kernel_combobox, load_kernel_btn, pre_treated_check, status_kernel_lbl, weights_status_lbl, load_weights_btn, \
        result_lbl
        
    test_frame = ctk.CTkScrollableFrame(master=master, corner_radius=0, fg_color=GUI_ELEMENTS["colors"]["background"],scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    test_frame.grid(row=0, column=2, sticky="nsew", columnspan=12, rowspan=12)
    test_frame = grid_setup(test_frame)

    # Title of the test
    title_txt = "Sección de Pruebas para Imagenes"
    title = ctk.CTkLabel(master=test_frame, text=title_txt, font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title.grid(row=0, column=0, pady=10, sticky="new", columnspan=11)

    # Button for show information about the process
    info_btn = ctk.CTkButton(master=test_frame, text="Ayuda", fg_color=GUI_ELEMENTS["colors"]["info_button"], width=100, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["info_button_hover"], text_color=GUI_ELEMENTS["colors"]["foreground"])
    info_btn.grid(row=0, column=11, pady=10, sticky="en")
    info_btn.configure(command= lambda: show_info_image_test())

    # Button for load the image
    load_folder_btn = ctk.CTkButton(master=test_frame, text="Cargar Imágen", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    load_folder_btn.grid(row=1, column=0, pady=10, padx=20, sticky="wn", columnspan=2)
    load_folder_btn.configure(command= lambda: load_test_image())
    
    # Label for the status of the image charge
    image_status_lbl = ctk.CTkLabel(master=test_frame, text="Imagen No Cargada", font=("Arial", 18, "bold"), text_color="red")
    image_status_lbl.grid(row=1, column=2, pady=10, sticky="n", columnspan=3)
    
    # chargue the resize section
    resize_frame = ctk.CTkFrame(master=test_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    resize_frame = resize_frame_creation(resize_frame, for_training = True)
    resize_frame.grid(row=2, column=0, sticky="nsew", columnspan=12, pady=10, padx=10)
    
    # label for entry a category response option
    response_lbl = ctk.CTkLabel(master=test_frame, text="Ingresar posible Categoría:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    response_lbl.grid(row=3, column=0, pady=10, padx=15, sticky="sen", columnspan=3)
    
    # Entry for the response of the category
    response_entry = ctk.CTkEntry(master=test_frame, width=200, font=("Arial", 14, "bold"))
    response_entry.grid(row=3, column=3, pady=10, padx = 10, sticky="swen", columnspan=2)
    
    # Button for add the category
    add_category_btn = ctk.CTkButton(master=test_frame, text="Agregar Categoría", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    add_category_btn.grid(row=3, column=5, pady=10, padx=15, sticky="wn", columnspan=2)
    add_category_btn.configure(command= lambda: add_test_category(response_entry.get()))
    
    # Button for delete a category
    delete_category_btn = ctk.CTkButton(master=test_frame, text="Eliminar Categoría", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    delete_category_btn.grid(row=3, column=7, pady=10, padx=15, sticky="wn", columnspan=2)
    delete_category_btn.configure(command= lambda: delete_test_category(response_entry.get()))
    
    # Checkbox for the selection of default categories
    default_categories_check = ctk.CTkCheckBox(master=test_frame, text="Categorías por defecto", font=("Arial", 14, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    default_categories_check.grid(row=3, column=9, pady=10, padx=15, sticky="sen", columnspan=2)
    default_categories_check.configure(command= lambda: check_default_categories(default_categories_check))
    
    # Scrollable frame for the categories
    categories_frame = ctk.CTkScrollableFrame(master=test_frame, corner_radius=8, fg_color="#ffffff", height=300)
    categories_frame.grid(row=4, column=0, sticky="nsew", columnspan=4, pady=10, padx=10)
    categories_frame = grid_setup(categories_frame)
    
    # Add the label title for the categories
    title_categories = ctk.CTkLabel(master=categories_frame, text="Categorías", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["background"], anchor="center", justify="center")
    title_categories.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)
    
    kernels_frame = ctk.CTkFrame(master=test_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    kernels_frame = grid_setup(kernels_frame)
    kernels_frame.grid(row=4, column=4, sticky="nsew", columnspan=8, pady=10, padx=10)
    
    # Label for the kernel series section
    kernel_series_lbl = ctk.CTkLabel(master=kernels_frame, text="Carga de Kernels", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    kernel_series_lbl.grid(row=0, column=0, pady=10, padx=15, sticky="swn", columnspan=12)
    
    # Combobox for the selection of the kernel seriesw
    kernel_combobox = ctk.CTkComboBox(master=kernels_frame, values=kernel_series, width=200, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], fg_color=GUI_ELEMENTS["colors"]["foreground"], text_color="#0F1010")
    kernel_combobox.grid(row=1, column=0, pady=10, padx = 10, sticky="sewn", columnspan=3)
    kernel_combobox.configure(command = select_kernel_series)
    
    # Button for load a kernel series
    load_kernel_btn = ctk.CTkButton(master=kernels_frame, text="Cargar Serie de Kernels", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    load_kernel_btn.grid(row=2, column=0, padx = 15 , pady=10, sticky="swn", columnspan=2)
    load_kernel_btn.configure(command= lambda: load_kernel_series())
    
    # Checkbox for the selection of the pre-treated images
    pre_treated_check = ctk.CTkCheckBox(master=kernels_frame, text="Imágenes Pretratadas", font=("Arial", 14, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    pre_treated_check.grid(row=3, column=0, pady=10, padx=15, sticky="sewn", columnspan=2)
    pre_treated_check.configure(command= lambda: check_pre_treated(pre_treated_check))
    
    # Label for the status of the kernel series
    status_kernel_lbl = ctk.CTkLabel(master=kernels_frame, text="Kernel No Cargado", font=("Arial", 14, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    status_kernel_lbl.grid(row=4, column=0, pady=10, padx=15, sticky="swn", columnspan=2)
    
    # Button for load the weights of the model
    load_weights_btn = ctk.CTkButton(master=test_frame, text="Cargar Pesos", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    load_weights_btn.grid(row=5, column=0, pady=10, padx=15, sticky="wn", columnspan=2)
    load_weights_btn.configure(command= lambda: charge_test_images_weights())
    
    # Checkbox for the selection of the default weights
    default_weights_check = ctk.CTkCheckBox(master=test_frame, text="Pesos por defecto", font=("Arial", 14, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    default_weights_check.grid(row=5, column=2, pady=10, padx=15, sticky="sen", columnspan=2)
    default_weights_check.configure(command= lambda: charge_default_weights(default_weights_check))
    
    # Label for the status of the weights
    weights_status_lbl = ctk.CTkLabel(master=test_frame, text="Pesos No Cargados", font=("Arial", 14, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    weights_status_lbl.grid(row=5, column=4, pady=10, padx=15, sticky="sen", columnspan=2)
    
    
    images_frame = ctk.CTkFrame(master=test_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"])
    images_frame = grid_setup(images_frame)
    images_frame.grid(row=6, column=0, sticky="nw", columnspan=12, pady=10, padx=10)
    
    # frame with a border radius (is for the chargued image) height 500px
    image_frame = ctk.CTkFrame(master=images_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"], border_width=2, border_color=GUI_ELEMENTS["colors"]["foreground"])
    image_frame = grid_setup(image_frame)
    image_frame.grid(row=6, column=0, sticky="ne", columnspan=6, pady=10, padx=10)    

    # Label for the chargued image
    image_lbl = ctk.CTkLabel(master=image_frame, text="Imagen Cargada", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"])
    image_lbl.grid(row=0, column=0, pady=10, padx=10, sticky="new", columnspan=12)

    # frame with a border radius (is for the treated image) height 500px 
    treated_image_frame = ctk.CTkFrame(master=images_frame, corner_radius=8, fg_color=GUI_ELEMENTS["colors"]["background"], border_width=2, border_color=GUI_ELEMENTS["colors"]["foreground"])
    treated_image_frame = grid_setup(treated_image_frame)
    treated_image_frame.grid(row=6, column=6, sticky="nw", columnspan=6, pady=10, padx=10)
    
    # Label for the treated image
    treated_image_lbl = ctk.CTkLabel(master=treated_image_frame, text="Imagen Tratada", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    treated_image_lbl.grid(row=0, column=0, pady=10, padx=10, sticky="new", columnspan=12)
    
    # Button for start the test process
    start_test_btn = ctk.CTkButton(master=test_frame, text="Empezar Prueba", fg_color=GUI_ELEMENTS["colors"]["foreground"], width=180, height=40, font=GUI_ELEMENTS["fonts"]["button_font"], hover_color=GUI_ELEMENTS["colors"]["hover_button"], text_color="#0F1010")
    start_test_btn.grid(row=7, column=0, pady=10, padx=15, sticky="wn", columnspan=2)
    start_test_btn.configure(command= lambda: start_image_test())
    
    # Label for the result
    result_lbl = ctk.CTkLabel(master=test_frame, text="Resultado:", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    result_lbl.grid(row=7, column=2, pady=10, padx=15, sticky="sen", columnspan=4)

def load_test_image():
    """
    Load the test image in the application
    This function is called when the user clicks on the load image button
    """
    global image_frame, image_status_lbl, test_image, test_filtered_image
    
    image = load_images(False)
    
    if image is None or len(image) == 0:
        image_status_lbl.configure(text="Error al cargar la imagen", text_color="red")
        return
    
    if len(image) > 1:
        image_status_lbl.configure(text="Debe cargar solo una imagen", text_color="red")
        return
    
    # Create the test image copy without any modification for presentation
    test_image = image[0].copy()
    image = image[0]
    
    width = image.shape[1]
    height = image.shape[0]
    
    image_info = f"Resolución:\n Alto: {height} - Ancho: {width} - Canales de Color: {image.shape[2]}"
      
    if image.shape[1] > 500:
        aspect_ratio = 500 / float(image.shape[1])
        ne_height = int(image.shape[0] * aspect_ratio)
        image = resize_image(image, 500, ne_height, "Bicubic")
        width = image.shape[1]
        height = image.shape[0]
        image_info += f"\n\nResolución ajustada (Solo para visualización):\n Alto: {height} - Ancho: {width} - Canales de Color: {image.shape[2]}\n"
    
    if test_image.shape[0] > 1500 or test_image.shape[1] > 1500:
        test_image = image.copy()
    
    print("width: ", width, "height: ", height)
    image = Image.fromarray(image)
    image = ctk.CTkImage(dark_image=image, light_image=image, size=(width, height))
    
    image_lbl = ctk.CTkLabel(master=image_frame, image=image , text=image_info, compound="bottom", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center")
    image_lbl.grid(row=1, column=0, pady=10, padx=10, sticky="nsew")
    
    image_status_lbl.configure(text="Imagen Cargada", text_color="green")
    image_frame.configure(width = 500)
    
    test_filtered_image = None
    
def add_test_category(category):
    """
    Add a category to the test images in the application
    This function is called when the user clicks on the add category button
    
    Parameters:
    category: Category to add to the test images
    """
    global categories_frame, test_categories, txt_categories

    if category == "":
        show_default_error("Debe ingresar una categoría")    
        return
    
    if category in test_categories:
        show_default_error("La categoría ya existe")
        return
    
    test_categories.append(category)
    
    # Create a textbox with the categories
    txt_categories = ctk.CTkTextbox(master=categories_frame, width=600, fg_color="#ffffff")
    txt_categories.grid(row=1, column=0, pady=5, padx=2, sticky="nsew")
    
    txt_categories.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    txt_categories.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
    txt_categories.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="#70600f")
    
    txt_categories.delete("1.0", "end")
    
    for category in test_categories:
        index = test_categories.index(category)
        txt_categories.insert("end", f"{index} - {category}\n", "title")  

def delete_test_category(category):
    """
    Delete a category from the test images in the application
    This function is called when the user clicks on the delete category button
    
    Parameters:
    category: Category to delete from the test images
    """
    
    global categories_frame, test_categories, txt_categories

    if category == "":
        show_default_error("Debe ingresar una categoría")
        return
    
    if category not in test_categories:
        show_default_error("La categoría no existe")
        return
        
    test_categories.remove(category)
    
    # Create a textbox with the categories
    txt_categories = ctk.CTkTextbox(master=categories_frame, width=600, fg_color="#ffffff")
    txt_categories.grid(row=1, column=0, pady=5, padx=2, sticky="nsew")

    txt_categories.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
    
    txt_categories.delete("1.0", "end")
    
    for category in test_categories:
        index = test_categories.index(category)
        txt_categories.insert("end", f"{index} - {category}\n", "title")  

def check_default_categories(check):
    """
    Check the default categories for the test images in the application
    This function is called when the user clicks on the default categories checkbox
    
    Parameters:
    check: Checkbutton to know if the default categories are selected
    """
    
    global categories_frame, test_categories, default_test_categories, txt_categories

    checked = True if check.get() == 1 else False

    if not checked:
        test_categories = []
        txt_categories.delete("1.0", "end")
        
    if checked:
        test_categories = default_test_categories
        
        # Create a textbox with the categories
        txt_categories = ctk.CTkTextbox(master=categories_frame, width=600, fg_color="#ffffff")
        txt_categories.grid(row=1, column=0, pady=5, padx=2, sticky="nsew")
        
        txt_categories.tag_config("title", cnf = {"font": GUI_ELEMENTS["fonts"]["title_font"]}, foreground=GUI_ELEMENTS["colors"]["background"])
        txt_categories.tag_config("subtitle", cnf = {"font": GUI_ELEMENTS["fonts"]["subtitle_font"]}, foreground="#70600f")
        txt_categories.tag_config("pattern", cnf = {"font": GUI_ELEMENTS["fonts"]["text_font"]}, foreground="#70600f")
        
        txt_categories.delete("1.0", "end")
        
        for category in test_categories:
            index = test_categories.index(category)
            txt_categories.insert("end", f"{index} - {category}\n", "title")  

def charge_default_weights(check):
    """
    Check the default weights for the test images in the application
    This function is called when the user clicks on the default weights checkbox
    
    Parameters:
    check: Checkbutton to know if the default weights are selected
    """
    global weights_status_lbl, weights_test_json
    
    if check.get() == 1:
        weights_test_json = load_json("case_5/base_5_weights")
        weights_status_lbl.configure(text="Pesos Cargados", text_color="green")
        return
    
    if check.get() == 0:
        weights_test_json = None
        weights_status_lbl.configure(text="Pesos No Cargados", text_color="red")
        return 

def charge_test_images_weights():
    """
    Load the weights for the test images in the application
    This function is called when the user clicks on the load weights button or the default weights checkbox
    """
    global weights_status_lbl, weights_test_json
    
    weights, filename = load_json()

    if weights is None:
        weights_status_lbl.configure(text="Error al cargar los pesos", text_color="red")
        return
    
    weights_test_json = weights
    
    weights_status_lbl.configure(text=f"Pesos {filename} Cargados", text_color="green")

def start_image_test():
    """
    Start the test process for the images in the application
    This function is called when the user clicks on the start test button
    """
    global test_image, treated_image_frame, test_categories, weights_test_json, result_lbl, selected_kernel_series,\
        width_entry, height_entry, interpolation_combobox, test_filtered_image

    result_lbl.configure(text=f"Resultado: ")

    if test_image is None:
        show_default_error("Debe cargar una imagen")
        return
    
    if len(test_categories) < 1:
        show_default_error("Debe ingresar al menos dos categorías")
        return
    
    if weights_test_json is None:
        show_default_error("Debe cargar los pesos")
        return
    
    if selected_kernel_series is None:
        show_default_error("Debe cargar o seleccionar una serie de kernels")
        return

    image = None
    if test_filtered_image is None:
        print("kernel series", selected_kernel_series)
        image = apply_kernel_series([test_image], selected_kernel_series) if selected_kernel_series != "None" else [test_image]
        image = image[0]
        test_filtered_image = image
    else:
        image = test_filtered_image
        
    if image is None or len(image) < 1:
        show_default_error("Error al aplicar los kernels")
        return
    
    width = image.shape[1]  
    height = image.shape[0]

    temp_image = image.copy()

    image_info = f"Resolución:\n Alto: {height} - Ancho: {width} - Canales de Color: {image.shape[2]}"
    
    if image.shape[1] > 500:
        aspect_ratio = 500 / float(image.shape[1])
        ne_height = int(image.shape[0] * aspect_ratio)
        temp_image = resize_image(image, 500, ne_height, "Bicubic")
        width = temp_image.shape[1]
        height = temp_image.shape[0]
        image_info += f"\n\nResolución ajustada (Solo para visualización):\n Alto: {height} - Ancho: {width} - Canales de Color: {temp_image.shape[2]}\n"
        
    temp_image = Image.fromarray(temp_image)
    temp_image = ctk.CTkImage(dark_image=temp_image, light_image=temp_image, size=(width, height))

    treated_image_lbl = ctk.CTkLabel(master=treated_image_frame, image=temp_image , text=image_info, compound="bottom", font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center")
    treated_image_lbl.grid(row=1, column=0, pady=10, padx=10, sticky="nsew", columnspan=12)
    
    # Resize the image for the model
    
    selected_width = width_entry.get()
    selected_height = height_entry.get()
    
    image_plained = plain_image(image)
    redimension = False
    
    
    tam_image_desired = len(weights_test_json["weights_h"][0])
    print("tam_image_desired: ", tam_image_desired)
    print("len(image_plained): ", len(image_plained))
    
    if len(image_plained) != tam_image_desired:
        show_default_error(f"La imagen será redimensionada porque debe tener un tamaño plano de {tam_image_desired} y tiene {len(image_plained)}")
        redimension = True
    
    if redimension and (selected_width == "" or selected_height == ""):
        show_default_error("Debe ingresar el ancho y el alto, no se redimensionará la imagen")
        return
    
    if redimension and not (selected_width.isdigit() or not selected_height.isdigit()):
        show_default_error("El ancho y el alto deben ser números enteros, no se redimensionará la imagen")
        return
    
    if redimension and interpolation_combobox.get() == "":
        show_default_error("Debe seleccionar una interpolación, no se redimensionará la imagen")
    
    if redimension:
        image = resize_image(image, int(width_entry.get()), int(height_entry.get()), interpolation_combobox.get())
    
    image_info += f"\nRedimensionada:\n Alto: {image.shape[0]} - Ancho: {image.shape[1]} - Canales de Color: {image.shape[2]}"
    temp_image = Image.fromarray(image)
    temp_image = ctk.CTkImage(dark_image=temp_image, light_image=temp_image, size=(image.shape[1], image.shape[0]))
    treated_image_lbl.configure(text=image_info, image=temp_image)
    
    image = plain_image(image)
        
    if len(image) != tam_image_desired:
        show_default_error(f"La imagen tratada debe tener un tamaño plano de {tam_image_desired} y tiene {len(image)} \n"+ 
                           "No se realizará la prueba, Cambie el tamaño de la imagen")
        return
    
    print("First 5 elements of the image: ", image[:5])
    
    image = normalize_image_vector(image)
    
    print("First 5 elements of the image normalized: ", image[:5])
    
    print("10 first weights of the first neuron in the h layer: ", weights_test_json["weights_h"][0][:10])
    print("10 first weights of the first neuron in the o layer: ", weights_test_json["weights_o"][0][:10])
    print("First 5 elements of the bias of the h layer: ", weights_test_json["bias_h"][:5])
    print("First 5 elements of the bias of the o layer: ", weights_test_json["bias_o"][:5])
    print("qty_neurons: ", weights_test_json["qty_neurons"])
    print("arquitecture: ", weights_test_json["arquitecture"])
    
    
    test_data = {
        "inputs": [image],
        "outputs": 0,
        "weights_h": weights_test_json["weights_h"],
        "weights_o": weights_test_json["weights_o"],
        "bias_h": weights_test_json["bias_h"],
        "bias_o": weights_test_json["bias_o"],
        "qty_neurons": weights_test_json["qty_neurons"],
        "function_h_name": weights_test_json["function_h_name"],
        "function_o_name": weights_test_json["function_o_name"],
        "arquitecture": weights_test_json["arquitecture"]
    }
    
    results, errors = test_neural_network(test_data, normalize = False, output = False)

    print("results: ", results)
    
    if results is None or len(results) < 1:
        show_default_error("Error durante la prueba")
        return
    
    
    result_category = test_categories[int(results[0][0])]
    result_lbl.configure(text=f"Resultado: {results[0]} = {result_category}")
# endregion

# region Top Level Windows for the Information of the Application
def show_info_image_training():
    """
    Show the information of the training images process in the application
    This function is called when the user clicks on the help button in the training images section
    """
    global info_window
    
    if info_window is not None:
        info_window.destroy()
    
    
    # Create a CtkToplevel window with some labels with info
    info_window = ctk.CTkToplevel()
    info_window.title("Información del Entrenamiento de Imágenes")
    icon_path = get_resource_path("Resources/brand_logo.ico")
    info_window.iconbitmap(icon_path)
    info_window = grid_setup(info_window)
    info_window.geometry("800x600")
    info_window.resizable(False, False)
    info_window.after(1, lambda: info_window.focus())
    info_window.attributes("-topmost", True)
    
    # Create scrollable frame
    info_frame = ctk.CTkScrollableFrame(master=info_window, corner_radius=0)
    info_frame.grid(row=0, column=0, sticky="nsew", columnspan=12, rowspan=12)
    info_frame = grid_setup(info_frame)
    
    
    # Title of the window
    title_txt = "Información del Entrenamiento de Imágenes"
    title = ctk.CTkLabel(master=info_frame, text=title_txt, font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 1
    step1_txt = "Paso 1: Cargar una categoría de imágenes"
    step1 = ctk.CTkLabel(master=info_frame, text=step1_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step1.grid(row=1, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 1
    desc1_txt = "Cargar una categoría de imágenes para entrenar el modelo de clasificación, se debe ingresar primero la etiqueta de la categoría y luego cargar las carpeta de imágenes."
    desc1 = ctk.CTkLabel(master=info_frame, text=desc1_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc1.grid(row=2, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 2
    step2_txt = "Paso 2: Seleccionar, Cargar una serie de Kernels o Selecionar imagen pretratada"
    step2 = ctk.CTkLabel(master=info_frame, text=step2_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step2.grid(row=3, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 2
    desc2_txt = ("En este paso se debe seleccionar la información para hacer el tratamiento de las imagenes"
                 " Hay varias opciones: Seleccionar de la lista desplegable alguna serie predefinida, cargar una serie de kernels en formato json para aplicar "
                 "O seleccionar la opción de Imagen pretratada, en este caso el software pasará las imagenes directamente a la sección de datos pretratados."
    "\n Ejemplo del json de kernels: \n")
    
    desc2 = ctk.CTkLabel(master=info_frame, text=desc2_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc2.grid(row=4, column=0, pady=10, sticky="new", columnspan=12)

    # Example of the json of kernels (Label with the image)
    kernels_example_path = get_resource_path("Resources/example_kernels_json.png")
    kernels_example_img = Image.open(kernels_example_path)
    kernels_example_photo = ctk.CTkImage(dark_image=kernels_example_img, light_image=kernels_example_img, size=(600, 400))
    kernels_example_label = ctk.CTkLabel(master=info_frame, image=kernels_example_photo, text="", compound="center")
    kernels_example_label.grid(row=5, column=0, pady=10, sticky="nsew", columnspan=12)
    
    # Step 2 Disclaimer
    disclaimer_txt = ("Nota: Si no se escribe una etiqueta no se pueden cargar las imagenes,"
                      "y si no se cargan las imagenes no se pueden cargar o seleccionar los kernels."
                      "Para cargar mas categorías debe hacer todo este proceso además de pretratar las imagenes y empezar de nuevo en el paso 1.")
    
    disclaimer = ctk.CTkLabel(master=info_frame, text=disclaimer_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    disclaimer.grid(row=6, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 3
    step3_txt = "Paso 3: Pretratar las imágenes"
    
    step3 = ctk.CTkLabel(master=info_frame, text=step3_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step3.grid(row=7, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 3
    desc3_txt = ("En este paso se debe  hacer click en el boton de 'Tratar Categoría de Imagenes' para pasar las imagenes a la sección de datos pretratados."
                 "Esto aplicará la serie de kernels por defecto en las opciones por defecto en la lista desplegable o las cargadas en el paso 2."
                 "\n Si la imagen ya ha sido pretratada este paso no se realizará y las imagenes pasaran a la sección de datos pretratados.")
    
    desc3 = ctk.CTkLabel(master=info_frame, text=desc3_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc3.grid(row=8, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 4
    step4_txt = "Paso 4: Aplanar las imágenes en las categorías"
    
    step4 = ctk.CTkLabel(master=info_frame, text=step4_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step4.grid(row=9, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 4
    desc4_txt = ("Una vez se hayan tratado todas las categorías de imagenes (Minimo 2) se desbloqueará el boton de 'Aplanar Imágenes'."
                 "Este debe ser clickeado para convertir a vector las imagenes y pasarlas a la sección de datos aplanados.")
    
    desc4 = ctk.CTkLabel(master=info_frame, text=desc4_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc4.grid(row=10, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 5
    step5_txt = "Paso 5: Llenar los datos de entrenamiento"
    
    step5 = ctk.CTkLabel(master=info_frame, text=step5_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step5.grid(row=11, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 5
    desc5_txt = ("En este paso se deben llenar los datos de entrenamiento, estos son los datos que se usarán para entrenar el modelo."
                 "En los datos de entrenamiento se debe ingresar alpha, maximas epocas posibles, numero de neuronas, funciones de activación"
                 ", si se quiere momentum y su valor Betha, la precisión deseada y el bias (0 son bias aleatorios).")
    
    desc5 = ctk.CTkLabel(master=info_frame, text=desc5_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc5.grid(row=12, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 6
    step6_txt = "Paso 6: Entrenar el modelo"
    step6 = ctk.CTkLabel(master=info_frame, text=step6_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step6.grid(row=13, column=0, pady=10, sticky="new", columnspan=12)  
    
    # Description of the step 6
    desc6_txt = ("En este paso se debe hacer click en el boton de 'Entrenar Modelo' para empezar el proceso de entrenamiento del modelo."
                 "Este proceso puede tardar dependiendo de la cantidad de datos, la cantidad de neuronas y la precisión deseada.")
    
    desc6 = ctk.CTkLabel(master=info_frame, text=desc6_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc6.grid(row=14, column=0, pady=10, sticky="new", columnspan=12)

def show_treatment_info(kernel_json, qty_images):
    """
    Show the information of the treatment images process in the application
    This function is called when the user clicks on the treat images button in the training images section
    This information is shown in a window that will be destroyed after 5 seconds
    
    Parameters:
    kernel_json: Json with the kernel series to apply to the images
    qty_images: Quantity of images to treat
    """
    global info_window
    
    if info_window is not None:
        info_window.destroy()
    
    
    # Create a CtkToplevel window with some labels with info
    info_window = ctk.CTkToplevel()
    info_window.title("Entrenamiento en proceso....")
    icon_path = get_resource_path("Resources/brand_logo.ico")
    info_window.iconbitmap(icon_path)
    info_window = grid_setup(info_window)
    info_window.geometry("600x500")
    info_window.resizable(False, False)
    info_window.attributes("-topmost", True)
    
    # Label for main info
    main_info = "Se empezo un proceso de tratamiento de imágenes, Esto puede tardar un poco, Sea paciente. \n\n PROCESO: \n"
    main_info_lbl = ctk.CTkLabel(master=info_window, text=main_info, font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    main_info_lbl.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)
    
    kernel_json_txt = f"Se aplicará la serie de kernels: \n {json.dumps(kernel_json, indent=2)}"
    kernel_json_lbl = ctk.CTkLabel(master=info_window, text=kernel_json_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    kernel_json_lbl.grid(row=1, column=0, pady=10, sticky="new", columnspan=12)   
    
    num_images_txt = f"Se tratarán {qty_images} imágenes"
    num_images_lbl = ctk.CTkLabel(master=info_window, text=num_images_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    num_images_lbl.grid(row=2, column=0, pady=10, sticky="new", columnspan=12)
    
    info_window.update()    
    info_window.after(5000, lambda: info_window.destroy())
    
    info_window.wait_window()
    
    # destroy the window after 5 seconds
    
def show_info_resize_images(error=None, main_info=True):
    """
    Show the information of the resize images process in the application
    This function is called when the user clicks on the resize images button in the training images section
    This information is shown in a window that will be destroyed after 5 seconds
    
    Parameters:
    error: Error message to show in the window
    main_info: Boolean to know if the main info is shown
    """
    global info_window
    
    if info_window is not None:
        info_window.destroy()
    
    # Create a CtkToplevel window with some labels with info
    info_window = ctk.CTkToplevel()
    info_window.title("Información de las Imagenes")
    icon_path = get_resource_path("Resources/brand_logo.ico")
    info_window.iconbitmap(icon_path)
    info_window = grid_setup(info_window)
    info_window.geometry("600x300")
    info_window.resizable(False, False)
    info_window.attributes("-topmost", True)
    
    # Create scrollable frame
    info_frame = ctk.CTkScrollableFrame(master=info_window, corner_radius=0)
    info_frame.grid(row=0, column=0, sticky="nsew", columnspan=12, rowspan=12)
    info_frame = grid_setup(info_frame)
    
    if main_info:
        txt = "Las imágenes no tienen la misma resolución, por favor redimensione las imágenes para poder continuar."
        info_lbl = ctk.CTkLabel(master=info_frame, text=txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
        info_lbl.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)

    if error is not None:
        error_lbl = ctk.CTkLabel(master=info_frame, text=error, font=("Arial", 16, "bold"), text_color="red", anchor="center", justify="center", wraplength=600)
        error_lbl.grid(row=1, column=0, pady=10, sticky="new", columnspan=12)

    info_window.update()

def show_info_image_test():
    """
    Show the information of the test images process in the application
    This function is called when the user clicks on the help button in the test images section
    """
    global info_window
    
    if info_window is not None:
        info_window.destroy()
    
    # Create a CtkToplevel window with some labels with info
    info_window = ctk.CTkToplevel()
    info_window.title("Información de la Prueba de Imágenes")
    icon_path = get_resource_path("Resources/brand_logo.ico")
    info_window.iconbitmap(icon_path)
    info_window = grid_setup(info_window)
    info_window.geometry("800x600")
    info_window.resizable(False, False)
    info_window.attributes("-topmost", True)
    
    # Create scrollable frame
    info_frame = ctk.CTkScrollableFrame(master=info_window, corner_radius=0)
    info_frame.grid(row=0, column=0, sticky="nsew", columnspan=12, rowspan=12)
    info_frame = grid_setup(info_frame)
    
    # Title of the window
    title_txt = "Información de la Prueba de Imágenes"
    title = ctk.CTkLabel(master=info_frame, text=title_txt, font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 1
    step1_txt = "Paso 1: Cargar una imagen para la prueba"
    step1 = ctk.CTkLabel(master=info_frame, text=step1_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step1.grid(row=1, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 1
    desc1_txt = "Cargar una imagen para realizar la prueba de clasificación, esta imagen será tratada con la serie de kernels seleccionada y se pasará por el modelo."
    desc1 = ctk.CTkLabel(master=info_frame, text=desc1_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc1.grid(row=2, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 2
    step2_txt = "Paso 2: Seleccionar o ingresar categorías y pesos"
    step2 = ctk.CTkLabel(master=info_frame, text=step2_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step2.grid(row=3, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 2
    desc2_txt = ("En este paso se debe seleccionar las categorías de la prueba, estas categorías deben ser las mismas que se usaron para entrenar el modelo."
                 "También se debe seleccionar los pesos del modelo, estos pesos son los que se usaron para entrenar el modelo."
                 "Si no se seleccionan las categorías o los pesos no se podrá realizar la prueba."
                 "Se pueden seleccionar las categorías por defecto o ingresar nuevas categorías por si se quieren cambiar los nombres."
                 "También se puede cargar los pesos del modelo o seleccionar los pesos por defecto.")
    desc2 = ctk.CTkLabel(master=info_frame, text=desc2_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc2.grid(row=4, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 3
    step3_txt = "Paso 3: Seleccionar un filtro o configuración de tratamiento"
    step3 = ctk.CTkLabel(master=info_frame, text=step3_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step3.grid(row=5, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 3
    desc3_txt = ("En este paso se tienen 3 opciones posibles: \n"
                 "1. Seleccionar una serie de kernels predefinida en la lista desplegable. (Para el ejercicio de los peces)\n"
                 "2. Cargar una serie de kernels en formato json para aplicar.\n"
                 "3. Seleccionar la opción de Imagen pretratada, en este caso el software pasará las imagenes directamente al modelo o pedirá redimensionarlas."
                 "Si se selecciona la opción de imagen pretratada no se aplicarán kernels.")
    desc3 = ctk.CTkLabel(master=info_frame, text=desc3_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc3.grid(row=6, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 4
    step4_txt = "Paso 4 (Opcional): Redimensionar la imagen"
    step4 = ctk.CTkLabel(master=info_frame, text=step4_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step4.grid(row=7, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 4
    desc4_txt = ("Si la imagen no tiene el tamaño plano necesario para el modelo se debe redimensionar la imagen."
                 "Se debe ingresar el ancho y el alto de la imagen, además de seleccionar una interpolación para el redimensionamiento."
                 "el sistema se encargará de recopilar la arquitectura puesta en el json de pesos y sabrá cual es el tamaño plano necesario.")
    desc4 = ctk.CTkLabel(master=info_frame, text=desc4_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc4.grid(row=8, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 5
    step5_txt = "Paso 5: Realizar la prueba"
    step5 = ctk.CTkLabel(master=info_frame, text=step5_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step5.grid(row=9, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 5
    desc5_txt = ("En este paso se debe hacer click en el boton de 'Realizar Prueba' para empezar el proceso de prueba de la imagen."
                 "Este proceso pasará la imagen, se tratará y se detendrá si algo falta por ser específicado, después de que la imagen"
                 "Este correctamente procesada, se aplana y se pasa al modelo, Finalmente se dará una predicción de la categoría de la imagen.")
    desc5 = ctk.CTkLabel(master=info_frame, text=desc5_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc5.grid(row=10, column=0, pady=10, sticky="new", columnspan=12)    
    
def show_default_error(error = ""):
    """
    Show a default error message in a window
    This function is called when an error occurs in the application
    
    Parameters:
    error: Error message to show in the window
    """
    global info_window
    
    if info_window is not None:
        info_window.destroy()
    
    # Create a CtkToplevel window with some labels with info
    info_window = ctk.CTkToplevel()
    info_window.title("Error")
    icon_path = get_resource_path("Resources/brand_logo.ico")
    info_window.iconbitmap(icon_path)
    info_window = grid_setup(info_window)
    info_window.geometry("400x300")
    info_window.resizable(False, False)
    info_window.attributes("-topmost", True)
    
    # Create scrollable frame
    info_frame = ctk.CTkScrollableFrame(master=info_window, corner_radius=0)
    info_frame.grid(row=0, column=0, sticky="nsew", columnspan=12, rowspan=12)
    info_frame = grid_setup(info_frame)
    
    # Label for the error
    error_lbl = ctk.CTkLabel(master=info_frame, text=error, font=("Arial", 16, "bold"), text_color="red", anchor="center", justify="center", wraplength=400)
    error_lbl.grid(row=0, column=0, pady=10, sticky="nsew", columnspan=12)
    
    info_window.after(5000, lambda: info_window.destroy())
    info_window.wait_window()
    
    info_window.update()

def show_info_treatment():
    """
    Show the information of the treatment images process in the application
    This function is called when the user clicks on the help button in the treatment images section
    """
    global info_window
    
    if info_window is not None:
        info_window.destroy()
    
    # Create a CtkToplevel window with some labels with info
    info_window = ctk.CTkToplevel()
    info_window.title("Información del Tratamiento de Imágenes")
    icon_path = get_resource_path("Resources/brand_logo.ico")
    info_window.iconbitmap(icon_path)
    info_window = grid_setup(info_window)
    info_window.geometry("800x600")
    info_window.resizable(False, False)
    info_window.attributes("-topmost", True)
    
    # Create scrollable frame
    info_frame = ctk.CTkScrollableFrame(master=info_window, corner_radius=0)
    info_frame.grid(row=0, column=0, sticky="nsew", columnspan=12, rowspan=12)
    info_frame = grid_setup(info_frame)
    
    # Title of the window
    title_txt = "Información del Tratamiento de Imágenes"
    title = ctk.CTkLabel(master=info_frame, text=title_txt, font=("Arial", 20, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    title.grid(row=0, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 1
    step1_txt = "Paso 1: Cargar una imagen o una carpeta de imagenes para el tratamiento"
    step1 = ctk.CTkLabel(master=info_frame, text=step1_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step1.grid(row=1, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 1
    desc1_txt = ("Se debe seleccionar la opción de cargar carpeta si se desea mas de una imagen, para cargar una imagen se debe clickear en el boton de 'Cargar Datos'."
                 "Se debe cargar al menos una imagen para poder continuar con el proceso de tratamiento.")
    desc1 = ctk.CTkLabel(master=info_frame, text=desc1_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc1.grid(row=2, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 2
    step2_txt = "Paso 2: Seleccionar un filtro o configuración de tratamiento"
    step2 = ctk.CTkLabel(master=info_frame, text=step2_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step2.grid(row=3, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 2
    desc2_txt = ("En este paso se debe seleccionar un filtro o una configuración de tratamiento para las imagenes cargadas."
                 "Se puede seleccionar un filtro predefinido en la lista desplegable o tambien llenar los valores para las"
                 " otras configuraciónes. Existen varias opciones como: Redimensionar, Rotar, Escalar, etc."
                 " Cada una de estas opciones tener sus valores llenos."
                 " Para la aplicación de kernels se debe tener en cuenta que un el stride no puede ser 0"
                 " \n Por otro lado el valor de porcentaje de color solo es utilizado en las opciones de aumentar o disminuir algún canal de color.")
    desc2 = ctk.CTkLabel(master=info_frame, text=desc2_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc2.grid(row=4, column=0, pady=10, sticky="new", columnspan=12)    

    # Step 3
    step3_txt = "Paso 3: Tratar las imagenes"
    step3 = ctk.CTkLabel(master=info_frame, text=step3_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step3.grid(row=5, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 3
    desc3_txt = ("Solo se puede hacer un tratamiento a la vez, y cada tratamiento tiene su botón en específico."
                 "para los kernels se debe clickear el boton de 'Aplicar Kernels', para rotar las imagenes el boton de 'Rotar'"
                 ", para redimensionar el boton de 'Redimensionar' y así sucesivamente. \n"
                 "Una vez que se realize 1 tratamiento se debe hacer click en el boton de cargar imágenes filtradas"
                 "para que la imagen tratada sea guardada y se pueda continuar con otro tratamiento.")
    desc3 = ctk.CTkLabel(master=info_frame, text=desc3_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc3.grid(row=6, column=0, pady=10, sticky="new", columnspan=12)
    
    # Optional remove kernel step
    step4_txt = "Paso Opcional: Quitar un filtro o configuración de tratamiento"
    step4 = ctk.CTkLabel(master=info_frame, text=step4_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step4.grid(row=7, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 4
    desc4_txt = ("Este paso es opcional si se desea eliminar un tratamiento hecho anteriormente, para esto se debe clickear el boton de 'Deshacer Tratamiento'."
                 "Este boton eliminará el último tratamiento hecho y se podrá continuar con otro tratamiento con la imagen que estaba anteriormente.")
    desc4 = ctk.CTkLabel(master=info_frame, text=desc4_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc4.grid(row=8, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 5
    step5_txt = "Guardar las imagenes tratadas"
    step5 = ctk.CTkLabel(master=info_frame, text=step5_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step5.grid(row=9, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 5
    desc5_txt = ("Una vez se hayan tratado todas las imagenes se debe hacer click en el boton de 'Descargar Imágenes'."
                 " Para que este proceso pueda ser realizado se debe poner un nombre para las imagenes, el software se"
                 " encargará de poner esa etiqueta al nombre de la imagen junto con un número de serie."
                 " Al clickear el boton de descargar la imagen se debe seleccionar la carpeta donde se desea guardar las imagenes.")
    desc5 = ctk.CTkLabel(master=info_frame, text=desc5_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc5.grid(row=10, column=0, pady=10, sticky="new", columnspan=12)
    
    # Step 6
    step6_txt = "Paso 6: Descargar los Kernels Aplicados"
    step6 = ctk.CTkLabel(master=info_frame, text=step6_txt, font=("Arial", 16, "bold"), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center")
    step6.grid(row=11, column=0, pady=10, sticky="new", columnspan=12)
    
    # Description of the step 6
    desc6_txt = ("Si se quiere replicar el proceso en alguna otra sección del software o si se desea conocer los tratamientos realizados"
                 " se puede descargar un archivo json con la información de los kernels aplicados. Esto se hace clickeando el boton de 'Descargar Kernels'.")
    desc6 = ctk.CTkLabel(master=info_frame, text=desc6_txt, font=("Arial", 14), text_color=GUI_ELEMENTS["colors"]["foreground"], anchor="center", justify="center", wraplength=600)
    desc6.grid(row=12, column=0, pady=10, sticky="new", columnspan=12)    
# endregion
if __name__ == "__main__":
    GUI_creation()