# Archivo para crear el algoritmo de backpropagation de nuevo ya que el otro no funciona
import math 
import random
from utils import get_resource_path, secondCase_Data, add_or_replace_secon_case
import json
import datetime
import math

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
    return 1 / (1 + math.exp(-x))

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
    return math.log(1 + math.exp(x))

def softplus_derivative(x):
    return 1 / (1 + math.exp(-x))

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
        }
    }

    if derivada:
        return switcher.get(fuction_name, {"derivative": tanh_derivative})["derivative"]
    else:
        return switcher.get(fuction_name, {"function": tanh})["function"]

# endregion

def normalize_data(entradas, salidas):
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

def backpropagation_training(train_data = None, epoch_label=None, total_error_label=None, labels_error=None, status_label=None, download_weights_btn = None, download_training_data_btn = None, results_btn = None):
    from app import update_training_process, changue_status_training
    # Json para guardar los datos del entrenamiento
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

    # Obtener la función para la capa oculta (h) y la capa de salida
    # También obtener las funciones para la derivada
    funcion_h = switch_function_output(funcion_h_nombre)
    funcion_o = switch_function_output(funcion_o_nombre)
    funcion_h_derivada = switch_function_output(funcion_h_nombre, True)
    funcion_o_derivada = switch_function_output(funcion_o_nombre, True)

    # Normalizar las entradas y las salidas
    entradas, salidas_d, *_ = normalize_data(entradas, salidas_d)

    # Verificar si alguna entrada tiene valor de 1 y volver su valor a 0.999
    # Verificar si alguna entrada tiene valor de 0 y volver su valor a 0.001
    for i in range(len(entradas)):
        for j in range(len(entradas[i])):
            if entradas[i][j] == 1:
                entradas[i][j] = 0.999
            elif entradas[i][j] == 0:
                entradas[i][j] = 0.001

    # Crear los pesos para las conexiones entre entrada y capa oculta
    # Cada neurona tiene una cantidad de pesos igual a la cantidad de entradas 
    # También se calcula el bias aparte en otra lista
    pesos_h = [[random.uniform(0.1, 1) for i in range(len(entradas[0]))] for j in range(neuronas_h_cnt)]
    if bias != 0:
        bias_h = [bias for i in range(neuronas_h_cnt)]
    else:
        bias_h = [random.uniform(0.1, 1) for i in range(neuronas_h_cnt)]

    # Crear los pesos para las conexiones entre capa oculta y salida
    # Cada neurona tiene una cantidad de pesos igual a la cantidad de neuronas en la capa oculta
    # También se calcula el bias aparte en otra lista 
    pesos_o = [[random.uniform(0.1, 1) for i in range(neuronas_h_cnt)] for j in range(neuronas_o_cnt)]
    if bias != 0:
        bias_o = [bias for i in range(neuronas_o_cnt)]
    else:
        bias_o = [random.uniform(0.1, 1) for i in range(neuronas_o_cnt)]

    # Bucle While que se ejecuta hasta que todos los errores de patrones sean menores a la precisión
    # o hasta que se llegue al máximo de épocas
    epoca = 0
    delta_o = []

    # Valor B para calcular el momentum
    B = beta
    momentum = 0

    # Inicializar los errores de los patrones en un valor igual a la precisión + 5
    for i in range(len(entradas)):
        errores_patrones.append(precision + 0.9)

    while any([error > precision for error in errores_patrones]):
        
        if epoca % 1000 == 0:
            # Guardar los pesos iniciales en los registros
            pesos_h_registro.append([row[:] for row in pesos_h])  
            pesos_o_registro.append([row[:] for row in pesos_o])  
            bias_h_registro.append(bias_h[:])  
            bias_o_registro.append(bias_o[:])  
            errores_patrones_registro.append(errores_patrones[:])  
            error_total = sum(errores_patrones) / len(entradas)
            errores_totales.append(error_total)

            print("Epoca: ", epoca, "Error Total: ", error_total)
            for i in range(len(entradas)):
                if errores_patrones[i] < precision and i % 4 != 0:
                    # imprimir con solo 10 decimales y color verde
                    print("\033[32m#", i, "|E| ", "{:.10f}".format(errores_patrones[i]), "\033[0m", end=" ")
                elif errores_patrones[i] <= precision and i % 4 == 0:
                    print("\033[32m#", i, "|E| ", "{:.10f}".format(errores_patrones[i]), "\033[0m")
                elif errores_patrones[i] > precision and i % 4 != 0:
                    print("\033[91m#", i, "|E| ", "{:.10f}".format(errores_patrones[i]), "\033[0m", end=" ")
                else:
                    print("\033[91m#", i, "|E| ", "{:.10f}".format(errores_patrones[i]), "\033[0m")

            print("\n-------------------------------------------------")
            update_training_process(epoca, errores_patrones, error_total, precision, epoch_label, total_error_label, labels_error)

        # Empezar el recorrido de los patrones
        for p in range(len(entradas)):
            
            if MOMENTUM:
                # Eliminar los pesos anteriores para mantener la lista con solo 1 elemento
                pesos_h_momentum = pesos_h_momentum[-1:]
                pesos_o_momentum = pesos_o_momentum[-1:]
                # Guardar los pesos iniciales en las listas del momentum
                pesos_h_momentum.append([row[:] for row in pesos_h])
                pesos_o_momentum.append([row[:] for row in pesos_o])

            # Sacar las entradas en ese patrón
            x = entradas[p]

            # Sacar las salidas deseadas en ese patrón
            yd = salidas_d[p]

            # Inicializar Neth y Yh
            Nethj = [0 for i in range(neuronas_h_cnt)]
            Yh = [0 for i in range(neuronas_h_cnt)]

            # Realizar la sumatoria entre las entradas y los pesos de la capa oculta
            # Net^h(p,j)=Net^h(p,j)+X(p,i)*Wh(j,i)
            # A esa sumatoria se le suma el bias de la capa oculta
            # Net^h(p,j)=Net^h(p,j)+ Th(j,0)
            for j in range(neuronas_h_cnt):
                for i in range(len(x)):
                    Nethj[j] += x[i] * pesos_h[j][i]

                Nethj[j] += bias_h[j]
                # Después se realiza la salida de la capa oculta
                # Yh(p,j)=Funcion_activacion_h(Neth(p,j))
                Yh[j] = funcion_h(Nethj[j])

            # Inicializar Net^o(p,k) y Yk(p,k)
            Netok = [0 for i in range(neuronas_o_cnt)]
            Yk = [0 for i in range(neuronas_o_cnt)]
            
            # Realizar la sumatoria entre las salidas de la capa oculta y los pesos de la capa de salida
            # Net^o(p,k)=Neto(p,k)+Yh(p,j)*Wo(k,j)
            # A esa sumatoria se le suma el bias de la capa de salida
            # Net^o(p,k)=Neto(p,k)+To(k,0)
            for k in range(neuronas_o_cnt):
                for j in range(neuronas_h_cnt):
                    Netok[k] += Yh[j] * pesos_o[k][j]

                Netok[k] += bias_o[k]
                # Después se realiza la salida de la capa de salida
                # Yk(p,k)=Funcion_activacion_o(Neto(p,k))
                Yk[k] = funcion_o(Netok[k])

            # Inicializar la lista de los errores de salida
            delta_o = [0 for i in range(neuronas_o_cnt)]

            # Calcular el error de salida
            # ∂^o = (d(p,k) - Yk(p,k)) * Funcion_derivada_o(Neto(p,k))
            for k in range(neuronas_o_cnt):
                delta_o[k] = (salidas_d[p][k] - Yk[k]) * funcion_o_derivada(Yk[k])

            # Inicializar la matriz de los errores de la capa oculta
            delta_h = [[0 for i in range(len(x))] for j in range(neuronas_h_cnt)]

            # Calcular el error de la capa oculta
            # ∂^h(p,j) = Funcion_derivada_h(Neth(p,j)) * Σ(∂^o(p,k) * Wo(k,j))
            # primero se calcula la sumatoria en una variable llamada backpropagation
            # Después se calcula el error de la capa oculta
            for j in range(neuronas_h_cnt):
                backpropagation = 0
                for k in range(neuronas_o_cnt):
                    backpropagation += delta_o[k] * pesos_o[k][j]

                for i in range(len(x)):
                    delta_h[j][i] = funcion_h_derivada(Nethj[j]) * backpropagation

            # Actualizar los pesos de la capa de salida
            # W^o_(k,j)(t + 1) = W^o_(k,j)(t) + α * ∂^o(p,k) * Yh(p,j) + momentum
            for k in range(neuronas_o_cnt):
                for j in range(neuronas_h_cnt):
                    # Calcular el momentum B * (w^o(t) - w^o(t - 1))
                    momentum = B * (pesos_o[k][j] - pesos_o_momentum[-1][k][j]) if MOMENTUM else 0
                    # Calcular el nuevo peso
                    pesos_o[k][j] += alpha * delta_o[k] * Yh[j] 
                    pesos_o[k][j] += momentum

                # Actualizar el bias de la capa de salida
                # Cálculo del momentum para el bias
                # To(k,0)(t + 1) = To(k,0)(t) + α * ∂^o(p,k) + momentum
                bias_o[k] += alpha * delta_o[k] 

            # Actualizar los pesos de la capa oculta
            # W^h_(j,i)(t + 1) = W^h_(j,i)(t) + α * ∂^h(p,j,i) * X(p,i) + momentum
            for j in range(neuronas_h_cnt):
                for i in range(len(x)):
                    # Calcular el momentum B * (w^h(t) - w^h(t - 1))
                    momentum = B * (pesos_h[j][i] - pesos_h_momentum[-1][j][i]) if MOMENTUM else 0
                    # Calcular el nuevo peso
                    pesos_h[j][i] += alpha * delta_h[j][i] * x[i]
                    pesos_h[j][i] += momentum

                # Actualizar el bias de la capa oculta
                # Cálculo del momentum para el bias
                # Th(j,0)(t + 1) = Th(j,0)(t) + α * Σ(∂^h(p,j,i)/len(∂^h(p,j,i))) + momentum
                bias_h[j] += alpha * sum(delta_h[j]) / len(delta_h[j]) 

            # Calcular el error total del patrón
            # E^p = 1/2 * Σ(d(p,k) - Yk(p,k))^2
            error_patron = 0.5 * sum((yd[k] - Yk[k]) ** 2 for k in range(neuronas_o_cnt))

            # Guardar el error del patrón
            errores_patrones[p] = error_patron

        error_total = sum(errores_patrones) / len(entradas)

        epoca += 1

        if epoca >= iteraciones_max:
            changue_status_training(status_label, "Límite de épocas alcanzado", "red", download_weights_btn, download_training_data_btn, results_btn)
            return

        if stop_training:
            changue_status_training(status_label, "Entrenamiento detenido", "red", download_weights_btn, download_training_data_btn, results_btn)
            return


    errores_totales.append(error_total)
    pesos_h_registro.append([row[:] for row in pesos_h])  
    pesos_o_registro.append([row[:] for row in pesos_o])  
    bias_h_registro.append(bias_h[:])  
    bias_o_registro.append(bias_o[:])  
    errores_patrones_registro.append(errores_patrones[:])  

    update_training_process(epoca, errores_patrones, error_total, precision, epoch_label, total_error_label, labels_error)


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


def test_neural_network(test_data):
    
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

    # Normalizar las entradas y las salidas
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

    Y_resultados = [[(Y_resultados[i][j] * (maximo_salida - minimo_salida)) + minimo_salida for j in range(neuronas_o_cnt)] for i in range(len(Y_resultados))]
    errores = []

    # Impresion del resultado de las pruebas
    print("\n\n\n\n")
    print("||----------------------------------------------------------------||")
    print("||                        PRUEBA DEL CASO                         ||")
    print("||----------------------------------------------------------------||")
    print(f"|| Cantidad de neuronas en la capa oculta: {neuronas_h_cnt}")
    print("||----------------------------------------------------------------||")
    print("||                        PESOS DE LA RED                         ||")
    print("||----------------------------------------------------------------||")
    print("|| Pesos de la capa oculta:                                       ||")
    print("||----------------------------------------------------------------||")
    for i in range(len(pesos_h)):
        print("|| ", pesos_h[i])
    print("||----------------------------------------------------------------||")
    print("|| Bias de la capa oculta:                                        ||")
    print("||----------------------------------------------------------------||")
    print("|| ", bias_h)
    print("||----------------------------------------------------------------||")
    print("|| Pesos de la capa de salida:                                    ||")
    print("||----------------------------------------------------------------||")
    for i in range(len(pesos_o)):
        print("|| ", pesos_o[i])
    print("||----------------------------------------------------------------||")
    print("|| Bias de la capa de salida:                                     ||")
    print("||----------------------------------------------------------------||")
    print("|| ", bias_o)
    print("||----------------------------------------------------------------||")
    print("||                        RESULTADOS OBTENIDOS                    ||")
    print("||----------------------------------------------------------------||")
    print("|| Patrón # | Entradas | Salidas deseadas | Salidas obtenidas | MSE     ||")
    print("||----------------------------------------------------------------||")
    for i in range(len(entradas)):
        for j in range(len(salidas[i])):
            error = 0.5 * math.pow(salidas[i][j] - Y_resultados[i][j],2)
            errores.append(f"{error:.10f}")
        print("|| ", i, "      | ", entradas[i], " | ", "\033[34m" ,salidas[i], "\033[0m", " | ", "\033[33m" , Y_resultados[i],"\033[0m", " | ", "\033[32m", errores, "\033[0m")
    print("||----------------------------------------------------------------||")
    print("||                        FIN DE LA PRUEBA                        ||")
    print("||----------------------------------------------------------------||")

    return Y_resultados, errores
