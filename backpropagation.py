# Archivo para crear el algoritmo de backpropagation de nuevo ya que el otro no funciona
import math 
import random
from utils import get_resource_path
import os
import scipy.special as sp
import json
import datetime

def switch_function_output(fuction_name, derivada = False):
    # Function for return the equation used for calculate the output
    switcher = {
        "sigmoid": {
            "function": sigmoid,
            "derivative": sigmoid_derivative
            },
        "tanh": {
            "function": tanh,
            "derivative": tanh_derivative
        }
    }

    if derivada:
        return switcher.get(fuction_name, sigmoid).get("derivative", sigmoid_derivative)
    else:
        return switcher.get(fuction_name, sigmoid).get("function", sigmoid)

def sigmoid(x):
    return sp.expit(x)

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2

    
def backpropagation_training(entradas, salidas_d, alpha=0.1, funcion_h_nombre="sigmoid", funcion_o_nombre="sigmoid", neuronas_h_cnt=2, precision=0.001, iteraciones_max=50000, MOMENTUM = True):
    
    # Json para guardar los datos del entrenamiento
    graph_json = {}
    
    # Booleano para saber si se debe aumentar la cantidad de neuronas en la capa oculta
    aumentar_neuronas = False
    
    # Inicializar listas para almacenar pesos, bias y errores
    pesos_h_registro = []
    pesos_o_registro = []
    bias_h_registro = []
    bias_o_registro = []
    errores_totales = []
    errores_patrones = []
    errores_patrones_registro = []

    # Obtener la cantidad de neuronas de salida
    neuronas_o_cnt = len(salidas_d[0])

    # Obtener la función para la capa oculta (h) y la capa de salida
    # También obtener las funciones para la derivada
    funcion_h = switch_function_output(funcion_h_nombre)
    funcion_o = switch_function_output(funcion_o_nombre)
    funcion_h_derivada = switch_function_output(funcion_h_nombre, True)
    funcion_o_derivada = switch_function_output(funcion_o_nombre, True)

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
    bias_h = [random.uniform(0.1, 1) for i in range(neuronas_h_cnt)]

    # Crear los pesos para las conexiones entre capa oculta y salida
    # Cada neurona tiene una cantidad de pesos igual a la cantidad de neuronas en la capa oculta
    # También se calcula el bias aparte en otra lista 
    pesos_o = [[random.uniform(0.1, 1) for i in range(neuronas_h_cnt)] for j in range(neuronas_o_cnt)]
    bias_o = [random.uniform(0.1, 1) for i in range(neuronas_o_cnt)]

    # Bucle While que se ejecuta hasta que todos los errores de patrones sean menores a la precisión
    # o hasta que se llegue al máximo de épocas
    epoca = 0
    delta_o = []

    # Valor B para calcular el momentum
    B = 0.1
    momentum = 0

    # Inicializar los errores de los patrones en un valor igual a la precisión + 5
    for i in range(len(entradas)):
        errores_patrones.append(precision + 5)

    while any([error > precision for error in errores_patrones]):

        # Guardar los pesos iniciales en los registros
        pesos_h_registro.append([row[:] for row in pesos_h])  # Copia profunda
        pesos_o_registro.append([row[:] for row in pesos_o])  # Copia profunda
        bias_h_registro.append(bias_h[:])  # Copia profunda
        bias_o_registro.append(bias_o[:])  # Copia profunda

        # Empezar el recorrido de los patrones
        for p in range(len(entradas)):
            
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
                    momentum = B * (pesos_o[k][j] - pesos_o_registro[-1][k][j]) if MOMENTUM else 0
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
                    momentum = B * (pesos_h[j][i] - pesos_h_registro[-1][j][i]) if MOMENTUM else 0
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

        print("Epoca: ", epoca, " Error por patrón: ", errores_patrones)
        print("Pesos de la capa oculta: ", pesos_h)
        print("Pesos de la capa de salida: ", pesos_o)
        print("-------------------------------------------------")

        # Guardar los errores de los patrones en los registros
        errores_patrones_registro.append(errores_patrones[:])
        # Calcular el error total
        error_total = sum(errores_patrones) / len(entradas)
        errores_totales.append(error_total)

        epoca += 1

        if epoca >= iteraciones_max:
            print("Se llegó al máximo de épocas")
            aumentar_neuronas = True
            break
    
    if aumentar_neuronas:
        print("Se aumentará la cantidad de neuronas en la capa oculta a ", neuronas_h_cnt + 1)
        input()
        pesos_h, pesos_o, bias_h, bias_o, neuronas_h_cnt, graph_json = backpropagation_training(entradas, salidas_d, alpha, funcion_h_nombre, funcion_o_nombre, neuronas_h_cnt + 1, precision, iteraciones_max, MOMENTUM)
        return pesos_h, pesos_o, bias_h, bias_o, neuronas_h_cnt, graph_json

    print("||-------------------------------------------------------------------||")
    print("         SE TERMINO EL ENTRENAMIENTO CORRECTAMENTE                  ")
    print("Pesos resultantes de la capa oculta: \n", pesos_h)
    print("Pesos resultantes de la capa de salida: \n", pesos_o)
    print("Pesos de la capa de salida: \n", pesos_o)
    print("Bias de la capa oculta: \n", bias_h)
    print("||-------------------------------------------------------------------||")

    graph_json = {
        "fecha_entrenamiento": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epocas": epoca,
        "cantidad_neuronas": neuronas_h_cnt,
        "funcion_h": funcion_h_nombre,
        "funcion_o": funcion_o_nombre,
        "momentum": MOMENTUM,
        "b": B,
        "alpha": alpha,
        "errores_totales": errores_totales,
        "errores_patrones": errores_patrones_registro,
        "pesos_h": pesos_h_registro,
        "pesos_o": pesos_o_registro,
        "bias_h": bias_h_registro,
        "bias_o": bias_o_registro
    }


    return pesos_h, pesos_o, bias_h, bias_o, neuronas_h_cnt, graph_json

def prueba_backpropagation(entradas, salidas, pesos_h, pesos_o, bias_h, bias_o, neuronas_h_cnt = 2, funcion_h_nombre = "sigmoid", funcion_o_nombre = "sigmoid"):
    
    # Obtener la cantidad de neuronas de salida
    neuronas_o_cnt = len(salidas[0])

    # Obtener la funcion para la capa oculta (h) y la capa de salida
    # También obtener las funciones para la derivada

    funcion_h = switch_function_output(funcion_h_nombre)
    funcion_o = switch_function_output(funcion_o_nombre)

    # Verificar si alguna entrada tiene valor de 1 y volver su valor a 0.999
    # Verificar si alguna entrada tiene valor de 0 y volver su valor a 0.001
    for i in range(len(entradas)):
        for j in range(len(entradas[i])):
            if entradas[i][j] == 1:
                entradas[i][j] = 0.999
            elif entradas[i][j] == 0:
                entradas[i][j] = 0.001

    # Realizar la suma de las entradas y los pesos de la capa oculta
    # Despues se le suma el bias de la capa oculta
    # Realizar la salida de la capa oculta
    # Realizar la suma de las salidas de la capa oculta y los pesos de la capa de salida
    # Despues se le suma el bias de la capa de salida
    # Realizar la salida de la capa de salida
    # Guardar el resultado en una lista 
    # Imprimir el resultado de forma -> Patron #, Entradas, Salidas deseadas, Salidas obtenidas
    Y_resultados = []
    for p in range(len(entradas)):
        x = entradas[p]
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

    # Impresion del resultado de las pruebas
    print("\n\n\n\n")
    print("||----------------------------------------------------------------||")
    print("||                        PRUEBA DEL CASO                         ||")
    print("||----------------------------------------------------------------||")
    print(f"|| Cantidad de neuronas en la capa oculta: {neuronas_h_cnt}                ||")
    print("||----------------------------------------------------------------||")
    print("||                        PESOS DE LA RED                         ||")
    print("||----------------------------------------------------------------||")
    print("|| Pesos de la capa oculta:                                       ||")
    print("||----------------------------------------------------------------||")
    for i in range(len(pesos_h)):
        print("|| ", pesos_h[i], " ||")
    print("||----------------------------------------------------------------||")
    print("|| Bias de la capa oculta:                                        ||")
    print("||----------------------------------------------------------------||")
    print("|| ", bias_h, " ||")
    print("||----------------------------------------------------------------||")
    print("|| Pesos de la capa de salida:                                    ||")
    print("||----------------------------------------------------------------||")
    for i in range(len(pesos_o)):
        print("|| ", pesos_o[i], " ||")
    print("||----------------------------------------------------------------||")
    print("|| Bias de la capa de salida:                                     ||")
    print("||----------------------------------------------------------------||")
    print("|| ", bias_o, " ||")
    print("||----------------------------------------------------------------||")
    print("||                        RESULTADOS OBTENIDOS                    ||")
    print("||----------------------------------------------------------------||")
    print("|| Patrón # | Entradas | Salidas deseadas | Salidas obtenidas      ||")
    print("||----------------------------------------------------------------||")
    for i in range(len(entradas)):
        print("|| ", i, "      | ", entradas[i], " | ", salidas[i], " | ", Y_resultados[i], " ||")
    print("||----------------------------------------------------------------||")
    print("||                        FIN DE LA PRUEBA                        ||")
    print("||----------------------------------------------------------------||")


def realizar_caso(casos_json, numero_caso, funcion_h="tanh", funcion_o="tanh"):
    # Obtener el caso de entrenamiento # 0
    caso = casos_json[f"case_{numero_caso}"]

    # Obtener las entradas y las salidas deseadas
    entradas = caso["inputs"]
    salidas = caso["outputs"]

    # Entrenar la red
    pesos_h, pesos_o, bias_h, bias_o, neuronas_cnt, grap_json = backpropagation_training(entradas, salidas, alpha=0.2, neuronas_h_cnt = 2, precision=0.001, iteraciones_max=50000, funcion_h_nombre=funcion_h, funcion_o_nombre=funcion_o, MOMENTUM = True)

    # Probar la red
    if pesos_h != None and pesos_o != None and bias_h != None and bias_o != None:
        prueba_backpropagation(entradas, salidas, pesos_h, pesos_o, bias_h, bias_o, neuronas_h_cnt = neuronas_cnt, funcion_h_nombre=funcion_h, funcion_o_nombre=funcion_o)
    else:
        print("No se pudo entrenar la red")

    # Guardar los pesos y bias en un archivo
    with open(get_resource_path(f"Data/case_{numero_caso}_weights.json"), "w") as file:
        json.dump({
            "pesos_h": pesos_h,
            "pesos_o": pesos_o,
            "bias_h": bias_h,
            "bias_o": bias_o,
            "neuronas_cnt": neuronas_cnt,
            "funcion_h_nombre": funcion_h,
            "funcion_o_nombre": funcion_o
        }, file)

    # Guardar los datos del entrenamiento en un archivo
    with open(get_resource_path(f"Data/case_{numero_caso}_training.json"), "w") as file:
        json.dump(grap_json, file)

if __name__ == "__main__":
    # cargar los casos de pruebas
    casos_json = {}
    with open(get_resource_path("Data/cases.json"), "r") as file:
        casos_json = json.load(file)
    
    # Realizar el caso 0
    realizar_caso(casos_json, 0, "tanh", "tanh", neuronas_h = 2, precision = 0.001, iteraciones_max = 50000, MOMENTUM = True, alpha = 0.2)
    input("Presione enter para continuar con el caso 1")

    # Realizar el caso 1
    realizar_caso(casos_json, 1, "tanh", "tanh", neuronas_h = 6, precision = 0.01, iteraciones_max = 30000, MOMENTUM = True, alpha = 0.2)
    input("Presione enter para continuar con el caso 2")
    # Realizar el caso 2
    realizar_caso(casos_json, 2, "tanh", "tanh")
