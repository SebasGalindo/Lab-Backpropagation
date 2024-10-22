# Archivo para crear el algoritmo de backpropagation de nuevo ya que el otro no funciona
import math 
import random
import datetime
import numpy as np

stop_training = False
weights_json, graph_json = None, None

def set_stop_training(value = True):
    global stop_training
    stop_training = value

def get_weights_json():
    global weights_json
    return weights_json

def get_graph_json():
    global graph_json
    return graph_json

# region Funciones de activación y sus derivadas
def sigmoid(x):
    # Clip para evitar problemas numéricos
    x = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - math.tanh(x)**2

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    return 1 if x > 0 else alpha

def linear(x):
    return x

def linear_derivative(x):
    return 1

def softplus(x):
    # Clip para evitar problemas numéricos
    x = np.clip(x, -100, 100)
    return np.log(1 + np.exp(x))


def softplus_derivative(x):
    # Limitar el valor de x para evitar problemas con valores extremadamente grandes o pequeños
    x_clipped = np.clip(x, -100, 100)
    return 1 / (1 + math.exp(-x_clipped))


def elu(x, alpha=1.0):
    return x if x > 0 else alpha * (math.exp(x) - 1)

def elu_derivative(x, alpha=1.0):
    return 1 if x > 0 else alpha * math.exp(x)

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)


def softmax(x):
    exp_values = np.exp(x - np.max(x))  # Evitar overflow numérico
    return exp_values / np.sum(exp_values)


def softmax_derivative(softmax_output):
    # softmax_output es la salida del softmax para el input correspondiente
    return [s * (1 - s) for s in softmax_output]


# Función switch 
def switch_function_output(fuction_name, derivada=False):
    # Diccionario para obtener la función y su derivada
    switcher = {
        "sigmoid": {
            "function": sigmoid,
            "derivative": sigmoid_derivative
        },
        "tanh": {
            "function": tanh,
            "derivative": tanh_derivative
        },
        "ReLU": {
            "function": relu,
            "derivative": relu_derivative
        },
        "Leaky ReLU": {
            "function": leaky_relu,
            "derivative": leaky_relu_derivative
        },
        "Lineal": {
            "function": linear,
            "derivative": linear_derivative
        },
        "Softplus": {
            "function": softplus,
            "derivative": softplus_derivative
        },
        "ELU": {
            "function": elu,
            "derivative": elu_derivative
        },
        "Swish": {
            "function": swish,
            "derivative": swish_derivative
        }
    }

    if derivada:
        return switcher.get(fuction_name, {"derivative": tanh_derivative})["derivative"]
    else:
        return switcher.get(fuction_name, {"function": tanh})["function"]

# endregion

def normalize_data(entradas, salidas):
    # Convertir entradas y salidas a numpy arrays para facilitar las operaciones
    entradas = np.array(entradas)
    salidas = np.array(salidas)

    # Obtener el máximo y mínimo de todas las entradas y salidas
    maximo_entrada = np.max(entradas)
    minimo_entrada = np.min(entradas)
    entradas_n = (entradas - minimo_entrada) / (maximo_entrada - minimo_entrada)

    maximo_salida = np.max(salidas)
    minimo_salida = np.min(salidas)
    salidas_n = (salidas - minimo_salida) / (maximo_salida - minimo_salida)

    # Retornar las entradas y salidas normalizadas junto con los máximos y mínimos
    return entradas_n, salidas_n, maximo_entrada, minimo_entrada, maximo_salida, minimo_salida

def normalize_data_normal(entradas, salidas):
    # Entrada es una matriz [[]] y salida es una matriz [[]]
    # Normalizar las entradas y las salidas
    # hallar el valor máximo y el valor mínimo de las entradas
    maximo_entrada = max([max(entrada) for entrada in entradas])
    minimo_entrada = min([min(entrada) for entrada in entradas])

    # hallar el valor máximo y el valor mínimo de las salidas
    maximo_salida = max([max(salida) for salida in salidas])
    minimo_salida = min([min(salida) for salida in salidas])

    # realizar copia de las entradas y salidas para no afectar las originales
    entradas_n = [entrada[:] for entrada in entradas]
    salidas_n = [salida[:] for salida in salidas]

    # Normalizar las entradas y las salidas
    for i in range(len(entradas)):
        for j in range(len(entradas[i])):
            entradas_n[i][j] = (entradas[i][j] - minimo_entrada) / (maximo_entrada - minimo_entrada)
    
    for i in range(len(salidas)):
        for j in range(len(salidas[i])):
            salidas_n[i][j] = (salidas[i][j] - minimo_salida) / (maximo_salida - minimo_salida)

    return entradas_n, salidas_n, maximo_entrada, minimo_entrada, maximo_salida, minimo_salida

def backpropagation_training_normal(train_data = None, errors_text = None , status_label=None, download_weights_btn = None, download_training_data_btn = None, results_btn = None, main_window=None, normalize=True):
    from app import update_errors_ui, changue_status_training
    global weights_json, graph_json, stop_training
    
    # Booleano para saber si se debe aumentar la cantidad de neuronas en la capa oculta
    aumentar_neuronas = False
    
    # Inicializar listas para almacenar pesos, bias y errores
    bias_h = []
    bias_o = []
    pesos_h_registro = []
    pesos_o_registro = []
    bias_h_registro = []
    bias_o_registro = []
    pesos_h_momentum = []
    pesos_o_momentum = []
    errores_totales = []
    errores_patrones = []
    errores_patrones_registro = []

    funcion_h_nombre = train_data["function_h_name"]
    funcion_o_nombre = train_data["function_o_name"]
    neuronas_h_cnt = train_data["qty_neurons"]
    alpha = train_data["alpha"]
    precision = train_data["precision"]
    iteraciones_max = train_data["max_epochs"]
    MOMENTUM = train_data["momentum"]
    beta = train_data["betha"]
    bias = train_data["bias"]
    entradas = train_data["inputs"]
    salidas_d = train_data["outputs"]

    # Obtener la cantidad de neuronas de salida
    neuronas_o_cnt = len(salidas_d[0])

    # Obtener las funciones de activación y sus derivadas
    funcion_h = switch_function_output(funcion_h_nombre)
    funcion_o = switch_function_output(funcion_o_nombre)
    funcion_h_derivada = switch_function_output(funcion_h_nombre, True)
    funcion_o_derivada = switch_function_output(funcion_o_nombre, True)

    # Normalizar entradas y salidas si es necesario
    if normalize:
        entradas, salidas_d, *_ = normalize_data(entradas, salidas_d) 

    # Prevenir entradas extremas (1 -> 0.999 y 0 -> 0.001)
    for entrada in entradas:
        for i, valor in enumerate(entrada):
            entrada[i] = 0.999 if valor == 1 else 0.001 if valor == 0 else valor

    # Crear pesos y bias
    pesos_h = [[random.uniform(0.1, 1) for _ in entradas[0]] for _ in range(neuronas_h_cnt)]
    bias_h = [bias if bias != 0 else random.uniform(0.1, 1) for _ in range(neuronas_h_cnt)]
    
    pesos_o = [[random.uniform(0.1, 1) for _ in range(neuronas_h_cnt)] for _ in range(neuronas_o_cnt)]
    bias_o = [bias if bias != 0 else random.uniform(0.1, 1) for _ in range(neuronas_o_cnt)]

    # Inicializar los errores de los patrones en un valor mayor a la precisión
    errores_patrones = [precision + 0.9 for _ in entradas]

    # Bucle de entrenamiento principal
    epoca = 0
    B = beta
    momentum = 0

    while any(error > precision for error in errores_patrones):
        if epoca % 10 == 0:
            # Guardar pesos, bias y errores en registros
            pesos_h_registro.append([row[:] for row in pesos_h])  
            pesos_o_registro.append([row[:] for row in pesos_o])  
            bias_h_registro.append(bias_h[:])  
            bias_o_registro.append(bias_o[:])  
            errores_patrones_registro.append(errores_patrones[:])
            error_total = sum(errores_patrones) / len(entradas)
            errores_totales.append(error_total)

            # Actualización de la interfaz con los errores
            main_window.after(0, update_errors_ui, epoca, errores_patrones, error_total, precision, errors_text, main_window, False)

        # Entrenamiento por patrón
        for p, (x, yd) in enumerate(zip(entradas, salidas_d)):
            
            if MOMENTUM:
                pesos_h_momentum = pesos_h_momentum[-1:]
                pesos_o_momentum = pesos_o_momentum[-1:]
                pesos_h_momentum.append([row[:] for row in pesos_h])
                pesos_o_momentum.append([row[:] for row in pesos_o])

            # Inicializar Nethj y Yh
            Nethj = [sum(x[i] * pesos_h[j][i] for i in range(len(x))) + bias_h[j] for j in range(neuronas_h_cnt)]
            Yh = [funcion_h(Nethj[j]) for j in range(neuronas_h_cnt)]

            # Inicializar Netok y Yk
            Netok = [sum(Yh[j] * pesos_o[k][j] for j in range(neuronas_h_cnt)) + bias_o[k] for k in range(neuronas_o_cnt)]
            Yk = [funcion_o(Netok[k]) for k in range(neuronas_o_cnt)]

            # Calcular el error de salida (delta_o)
            delta_o = [(yd[k] - Yk[k]) * funcion_o_derivada(Yk[k]) for k in range(neuronas_o_cnt)]

            # Calcular el error de la capa oculta (delta_h)
            delta_h = [
                [funcion_h_derivada(Nethj[j]) * sum(delta_o[k] * pesos_o[k][j] for k in range(neuronas_o_cnt)) for _ in x] 
                for j in range(neuronas_h_cnt)
            ]

            # Actualizar los pesos de la capa de salida
            for k in range(neuronas_o_cnt):
                for j in range(neuronas_h_cnt):
                    momentum = B * (pesos_o[k][j] - pesos_o_momentum[-1][k][j]) if MOMENTUM else 0
                    pesos_o[k][j] += alpha * delta_o[k] * Yh[j] + momentum

                bias_o[k] += alpha * delta_o[k] 

            # Actualizar los pesos de la capa oculta
            for j in range(neuronas_h_cnt):
                for i in range(len(x)):
                    momentum = B * (pesos_h[j][i] - pesos_h_momentum[-1][j][i]) if MOMENTUM else 0
                    pesos_h[j][i] += alpha * delta_h[j][i] * x[i] + momentum

                bias_h[j] += alpha * sum(delta_h[j]) / len(delta_h[j]) 

            # Calcular el error total del patrón
            error_patron = 0.5 * sum((yd[k] - Yk[k]) ** 2 for k in range(neuronas_o_cnt))
            errores_patrones[p] = error_patron

        error_total = sum(errores_patrones) / len(entradas)
        epoca += 1
        
        if epoca >= iteraciones_max:
            #changue_status_training(status_label, "Límite de épocas alcanzado", "red", download_weights_btn, download_training_data_btn, results_btn)
            main_window.after(0, changue_status_training, status_label, "Límite de épocas alcanzado", "red", download_weights_btn, download_training_data_btn, results_btn)
            return

        if stop_training:
            #changue_status_training(status_label, "Entrenamiento detenido", "red", download_weights_btn, download_training_data_btn, results_btn)
            main_window.after(0, changue_status_training, status_label, "Entrenamiento detenido", "red", download_weights_btn, download_training_data_btn, results_btn)
            return

    errores_totales.append(error_total)
    pesos_h_registro.append([row[:] for row in pesos_h])  
    pesos_o_registro.append([row[:] for row in pesos_o])  
    bias_h_registro.append(bias_h[:])  
    bias_o_registro.append(bias_o[:])  
    errores_patrones_registro.append(errores_patrones[:])  

    main_window.after(0, update_errors_ui, epoca, errores_patrones, error_total, precision, errors_text, main_window , True)    



    print("||-------------------------------------------------------------------||")
    print("         SE TERMINO EL ENTRENAMIENTO CORRECTAMENTE                  ")
    print("Epoca Final: ", epoca)
    for i in range(len(entradas)):
        if errores_patrones[i] < precision and i % 4 != 0:
        # imprimir con solo 10 decimales
            print("\033[32m#", i, "|E| ", "{:.10f}".format(errores_patrones[i]), "\033[0m", end=" ")
        elif errores_patrones[i] <= precision and i % 4 == 0:
            print("\033[32m#", i, "|E| ", "{:.10f}".format(errores_patrones[i]), "\033[0m")
        elif errores_patrones[i] > precision and i % 4 != 0:
            print("\033[91m#", i, "|E| ", "{:.10f}".format(errores_patrones[i]), "\033[0m", end=" ")
        else:
            print("\033[91m#", i, "|E| ", "{:.10f}".format(errores_patrones[i]), "\033[0m")
    print("\n-------------------------------------------------------------------||")
    print("RESULTADOS CAPA OCULTA: \n")
    for j in range(len(pesos_h)):
        print(f"\nNeurona J = {j+1}") 
        print("Pesos:")
        for i in range(len(pesos_h[0])):
            if (i+1) % 5 !=0:
                print("\033[32m",f"W[{j+1}][{i+1}]:","{:.10f}".format(pesos_h[j][i]), "\033[0m", end=" | ")
            else: 
                print("\033[32m",f"W[{j+1}][{i+1}]:","{:.10f}".format(pesos_h[j][i]), "\033[0m")
        print("\033[32m","Bias:", bias_h[j], "\033[0m")
    print("RESULTADOS CAPA SALIDA: \n")
    for j in range(len(pesos_o)):
        print("\033[32m",f"\nNeurona J = {j+1}", "\033[0m",) 
        print("\033[32m","Pesos:", "\033[0m")
        for i in range(len(pesos_o[0])):
            if (i+1) % 5 !=0:
                print("\033[32m",f"W[{j+1}][{i+1}]:","{:.10f}".format(pesos_o[j][i]), "\033[0m", end=" | ")
            else: 
                print("\033[32m",f"W[{j+1}][{i+1}]:","{:.10f}".format(pesos_o[j][i]), "\033[0m")
        print("\033[32m","Bias:", bias_o[j], "\033[0m")

    print("||-------------------------------------------------------------------||")

    graph_json = {
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epoca,
        "max_epochs": iteraciones_max,
        "initial_bias": bias,
        "qty_neurons": neuronas_h_cnt,
        "arquitecture": [len(entradas[0]), neuronas_h_cnt, len(salidas_d[0])],
        "function_h": funcion_h_nombre,
        "function_o": funcion_o_nombre,
        "momentum": MOMENTUM,
        "b": beta,
        "alpha": alpha,
        "precision": precision,
        "totals_errors": errores_totales,
        "patterns_errors": errores_patrones_registro,
        "weights_h": pesos_h_registro,
        "weights_o": pesos_o_registro,
        "bias_h": bias_h_registro,
        "bias_o": bias_o_registro
    }

    weights_json = {
            "weights_h": pesos_h,
            "weights_o": pesos_o,
            "bias_h": bias_h,
            "bias_o": bias_o,
            "qty_neurons": neuronas_h_cnt,
            "function_h_name": funcion_h_nombre,
            "function_o_name": funcion_o_nombre
        }

    if not stop_training:
        changue_status_training(status_label, "Entrenamiento finalizado", "green", download_weights_btn, download_training_data_btn, results_btn)

def backpropagation_training(train_data=None, errors_text=None, status_label=None, download_weights_btn=None, download_training_data_btn=None, results_btn=None, main_window=None, normalize=True):
    from app import update_errors_ui, changue_status_training
    global weights_json, graph_json, stop_training

    # Inicializar listas para almacenar pesos, bias y errores
    errores_totales = []
    errores_patrones_registro = []

    # Extraer parámetros de configuración
    funcion_h_nombre = train_data["function_h_name"]
    funcion_o_nombre = train_data["function_o_name"]
    neuronas_h_cnt = train_data["qty_neurons"]
    alpha = train_data["alpha"]
    precision = train_data["precision"]
    iteraciones_max = train_data["max_epochs"]
    MOMENTUM = train_data["momentum"]
    beta = train_data["betha"]
    bias = train_data["bias"]
    entradas = np.array(train_data["inputs"])
    salidas_d = np.array(train_data["outputs"])

    # Obtener las funciones de activación y sus derivadas
    funcion_h = np.vectorize(switch_function_output(funcion_h_nombre))
    funcion_o = np.vectorize(switch_function_output(funcion_o_nombre))
    funcion_h_derivada = np.vectorize(switch_function_output(funcion_h_nombre, True))
    funcion_o_derivada = np.vectorize(switch_function_output(funcion_o_nombre, True))

    # Normalizar entradas y salidas si es necesario
    if normalize:
        entradas, salidas_d, *_ = normalize_data(entradas, salidas_d) 

    # Inicializar pesos y bias con numpy
    pesos_h = np.random.uniform(0.1, 1, (neuronas_h_cnt, entradas.shape[1]))
    bias_h = np.full(neuronas_h_cnt, bias if bias != 0 else np.random.uniform(-1, 1))

    pesos_o = np.random.uniform(0.1, 1, (salidas_d.shape[1], neuronas_h_cnt))
    bias_o = np.full(salidas_d.shape[1], bias if bias != 0 else np.random.uniform(-1, 1))

    # Inicializar errores de patrones
    errores_patrones = np.full(entradas.shape[0], precision + 0.9)

    # Make inputs = 1 -> 0.999 and inputs = 0 -> 0.001
    for i in range(len(entradas)):
        for j in range(len(entradas[i])):
            if entradas[i][j] == 1:
                entradas[i][j] = 0.999
            elif entradas[i][j] == 0:
                entradas[i][j] = 0.001

    epoca = 0
    momentum_h = np.zeros_like(pesos_h)
    momentum_o = np.zeros_like(pesos_o)

    while np.any(errores_patrones > precision):
        if epoca % 10 == 0:
            error_total = np.mean(errores_patrones)
            errores_totales.append(error_total)
            errores_patrones_registro.append(errores_patrones.copy().tolist())
            main_window.after(0, update_errors_ui, epoca, errores_patrones, error_total, precision, errors_text, main_window, False)

        for p in range(entradas.shape[0]):
            x = entradas[p]
            yd = salidas_d[p]

            # Forward propagation
            Nethj = np.dot(pesos_h, x) + bias_h
            Yh = funcion_h(Nethj)

            Netok = np.dot(pesos_o, Yh) + bias_o
            Yk = funcion_o(Netok)

            # Backward propagation
            delta_o = (yd - Yk) * funcion_o_derivada(Yk)
            delta_h = funcion_h_derivada(Nethj) * np.dot(pesos_o.T, delta_o)

            # Actualización de pesos capa de salida
            pesos_o += alpha * np.outer(delta_o, Yh) + beta * (pesos_o - momentum_o)
            bias_o += alpha * delta_o
            momentum_o = pesos_o.copy()

            # Actualización de pesos capa oculta
            pesos_h += alpha * np.outer(delta_h, x) + beta * (pesos_h - momentum_h)
            bias_h += alpha * delta_h
            momentum_h = pesos_h.copy()

            # Error del patrón
            errores_patrones[p] = 0.5 * np.sum((yd - Yk) ** 2)

        epoca += 1

        if epoca >= iteraciones_max:
            main_window.after(0, changue_status_training, status_label, "Límite de épocas alcanzado", "red", download_weights_btn, download_training_data_btn, results_btn)
            return

        if stop_training:
            main_window.after(0, changue_status_training, status_label, "Entrenamiento detenido", "red", download_weights_btn, download_training_data_btn, results_btn)
            return

    errores_totales.append(np.mean(errores_patrones))
    errores_patrones_registro.append(errores_patrones.copy().tolist())

    main_window.after(0, update_errors_ui, epoca, errores_patrones, np.mean(errores_patrones), precision, errors_text, main_window, True)

    # save only the 10k weights in graph json if the weights are more than 10k
    weights_h_graph = pesos_h if pesos_h.shape[0] <= 10000 else pesos_h[:10000]
    weights_o_graph = pesos_o if pesos_o.shape[0] <= 10000 else pesos_o[:10000]

    # Almacenar pesos y errores
    graph_json = {
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epoca,
        "max_epochs": iteraciones_max,
        "initial_bias": bias,
        "qty_neurons": neuronas_h_cnt,
        "arquitecture": [entradas.shape[1], neuronas_h_cnt, salidas_d.shape[1]],
        "function_h": funcion_h_nombre,
        "function_o": funcion_o_nombre,
        "momentum": MOMENTUM,
        "b": beta,
        "alpha": alpha,
        "precision": precision,
        "totals_errors": errores_totales,
        "patterns_errors": errores_patrones_registro,
        "weights_h": weights_h_graph.tolist(),
        "weights_o": weights_o_graph.tolist(),
        "bias_h": bias_h.tolist(),
        "bias_o": bias_o.tolist()
    }

    weights_json = {
        "weights_h": pesos_h.tolist(),
        "weights_o": pesos_o.tolist(),
        "bias_h": bias_h.tolist(),
        "bias_o": bias_o.tolist(),
        "qty_neurons": neuronas_h_cnt,
        "function_h_name": funcion_h_nombre,
        "function_o_name": funcion_o_nombre,
        "arquitecture": [entradas.shape[1], neuronas_h_cnt, salidas_d.shape[1]]
    }

    if not stop_training:
        changue_status_training(status_label, "Entrenamiento finalizado", "green", download_weights_btn, download_training_data_btn, results_btn)

def test_neural_network_normal(test_data, normalize=True):
    
    entradas = test_data["inputs"]
    salidas = test_data["outputs"]
    pesos_h = test_data["weights_h"]
    pesos_o = test_data["weights_o"]
    bias_h = test_data["bias_h"]
    bias_o = test_data["bias_o"]
    funcion_h_nombre = test_data["function_h_name"]
    funcion_o_nombre = test_data["function_o_name"]
    neuronas_h_cnt = test_data["qty_neurons"]


    # Obtener la cantidad de neuronas de salida
    neuronas_o_cnt = len(salidas[0])

    # Obtener la funcion para la capa oculta (h) y la capa de salida
    # También obtener las funciones para la derivada

    funcion_h = switch_function_output(funcion_h_nombre)
    funcion_o = switch_function_output(funcion_o_nombre)

    entradas_n = salidas_n = maximo_entrada = minimo_entrada = maximo_salida = minimo_salida = None
    # Normalizar las entradas y las salidas
    if normalize:
        entradas_n, salidas_n, maximo_entrada, minimo_entrada, maximo_salida, minimo_salida = normalize_data(entradas, salidas)

    # Verificar si alguna entrada tiene valor de 1 y volver su valor a 0.999
    # Verificar si alguna entrada tiene valor de 0 y volver su valor a 0.001
    for i in range(len(entradas_n)):
        for j in range(len(entradas_n[i])):
            if entradas_n[i][j] == 1:
                entradas_n[i][j] = 0.999
            elif entradas_n[i][j] == 0:
                entradas_n[i][j] = 0.001

    # Realizar la suma de las entradas y los pesos de la capa oculta
    # Despues se le suma el bias de la capa oculta
    # Realizar la salida de la capa oculta
    # Realizar la suma de las salidas de la capa oculta y los pesos de la capa de salida
    # Despues se le suma el bias de la capa de salida
    # Realizar la salida de la capa de salida
    # Guardar el resultado en una lista 
    # Imprimir el resultado de forma -> Patron #, Entradas, Salidas deseadas, Salidas obtenidas
    Y_resultados = []
    for p in range(len(entradas_n)):
        x = entradas_n[p]
        Nethj = [0 for i in range(neuronas_h_cnt)]
        Yh = [0 for i in range(neuronas_h_cnt)]

        for j in range(neuronas_h_cnt):
            for i in range(len(x)):
                Nethj[j] += x[i] * pesos_h[j][i]

            Nethj[j] += bias_h[j]
            Yh[j] = funcion_h(Nethj[j])

        Netok = [0 for i in range(neuronas_o_cnt)]
        Yk = [0 for i in range(neuronas_o_cnt)]

        for k in range(neuronas_o_cnt):
            for j in range(neuronas_h_cnt):
                Netok[k] += Yh[j] * pesos_o[k][j]

            Netok[k] += bias_o[k]
            Yk[k] = funcion_o(Netok[k])
            
        Y_resultados.append(Yk)

    if normalize:
        Y_resultados = [[(Y_resultados[i][j] * (maximo_salida - minimo_salida)) + minimo_salida for j in range(neuronas_o_cnt)] for i in range(len(Y_resultados))]
    else:
        Y_resultados = [[Y_resultados[i][j] for j in range(neuronas_o_cnt)] for i in range(len(Y_resultados))]
    
    errores = []

    # Impresion del resultado de las pruebas
    #print("\n\n\n\n")
    #print("||----------------------------------------------------------------||")
    #print("||                        PRUEBA DEL CASO                         ||")
    #print("||----------------------------------------------------------------||")
    #print(f"|| Cantidad de neuronas en la capa oculta: {neuronas_h_cnt}")
    #print("||----------------------------------------------------------------||")
    #print("||                        PESOS DE LA RED                         ||")
    #print("||----------------------------------------------------------------||")
    #print("|| Pesos de la capa oculta:                                       ||")
    #print("||----------------------------------------------------------------||")
    #for i in range(len(pesos_h)):
    #    print("|| ", pesos_h[i])
    #print("||----------------------------------------------------------------||")
    #print("|| Bias de la capa oculta:                                        ||")
    #print("||----------------------------------------------------------------||")
    #print("|| ", bias_h)
    #print("||----------------------------------------------------------------||")
    #print("|| Pesos de la capa de salida:                                    ||")
    #print("||----------------------------------------------------------------||")
    #for i in range(len(pesos_o)):
    #    print("|| ", pesos_o[i])
    #print("||----------------------------------------------------------------||")
    #print("|| Bias de la capa de salida:                                     ||")
    #print("||----------------------------------------------------------------||")
    #print("|| ", bias_o)
    #print("||----------------------------------------------------------------||")
    #print("||                        RESULTADOS OBTENIDOS                    ||")
    #print("||----------------------------------------------------------------||")
    #print("|| Patrón # | Entradas | Salidas deseadas | Salidas obtenidas | MSE     ||")
    #print("||----------------------------------------------------------------||")
    for i in range(len(entradas)):
        for j in range(len(salidas[i])):
            error = 0.5 * math.pow(salidas[i][j] - Y_resultados[i][j],2)
            errores.append(f"{error:.10f}")
    #    print("|| ", i, "      | ", entradas[i], " | ", "\033[34m" ,salidas[i], "\033[0m", " | ", "\033[33m" , Y_resultados[i],"\033[0m", " | ", "\033[32m", errores, "\033[0m")
    #print("||----------------------------------------------------------------||")
    #print("||                        FIN DE LA PRUEBA                        ||")
    #print("||----------------------------------------------------------------||")

    return Y_resultados, errores

def test_neural_network(test_data, normalize=True, output = True):
    entradas = np.array(test_data["inputs"])
    
    salidas = None
    if output: 
        salidas = np.array(test_data["outputs"])
    pesos_h = np.array(test_data["weights_h"])
    pesos_o = np.array(test_data["weights_o"])
    bias_h = np.array(test_data["bias_h"])
    bias_o = np.array(test_data["bias_o"])
    funcion_h_nombre = test_data["function_h_name"]
    funcion_o_nombre = test_data["function_o_name"]
    neuronas_h_cnt = test_data["qty_neurons"]

    # Obtener la cantidad de neuronas de salida
    neuronas_o_cnt = salidas.shape[1] if output else test_data["arquitecture"][2]
        
    # Obtener las funciones para la capa oculta (h) y la capa de salida
    funcion_h = np.vectorize(switch_function_output(funcion_h_nombre))
    funcion_o = np.vectorize(switch_function_output(funcion_o_nombre))

    maximo_entrada = minimo_entrada = maximo_salida = minimo_salida = None
    # Normalizar las entradas y las salidas
    if normalize:
        entradas, salidas, maximo_entrada, minimo_entrada, maximo_salida, minimo_salida = normalize_data(entradas, salidas)

    # Verificar si alguna entrada tiene valor de 1 o 0, ajustarlo a 0.999 o 0.001
    entradas = np.clip(entradas, 0.001, 0.999)

    # Cálculos para la capa oculta
    Nethj = np.dot(entradas, pesos_h.T) + bias_h
    Yh = funcion_h(Nethj)

    # Cálculos para la capa de salida
    Netok = np.dot(Yh, pesos_o.T) + bias_o
    Yk = funcion_o(Netok)

    # Calcular errores
    errores_patrones = np.full(entradas.shape[0], 0.0)
    
    if output:
        for i in range(len(salidas)):
            errores_patrones[i] = 0.5 * np.sum((salidas[i] - Yk[i]) ** 2)

    # Desnormalizar si es necesario
    if normalize:
        Yk = Yk * (maximo_salida - minimo_salida) + minimo_salida

    
    for i in range(Yk.shape[0]):
        if output:
            print("\nSalidas Esperadas:")
            print(salidas[i])
        print("Salidas Obtenidas:")
        print(Yk[i])
        if output:
            print("Error:")
            print(errores_patrones[i])

    return Yk.tolist() , errores_patrones.tolist()
   