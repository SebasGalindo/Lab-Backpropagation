import random
import math
import json
from utils import get_resource_path
import os
import datetime
import scipy.special as sp

def switch_function_output(fuction_name):
    # Function for return the equation used for calculate the output
    switcher = {
        "sigmoid": sigmoid,
    }

    return switcher.get(fuction_name, sigmoid)

def switch_function_derivative(fuction_name):
    # Function for return the equation used for calculate the derivative of the output
    switcher = {
        "sigmoid": sigmoid_derivative,
    }

    return switcher.get(fuction_name, sigmoid_derivative)

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

def backpropagation_trianing(inputs, outputs, alpha, function_h_name, function_o_name, bias = [], neurons_hidden_layer = 1, precision = 0.001, max_iterations = 100000):
    
    # If inputs are 1 or 0, the training is not possible
    # then for the 1 values x = 0.9999 and for the 0 values x = 0.0001
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if inputs[i][j] == 1:
                inputs[i][j] = 0.99
            elif inputs[i][j] == 0:
                inputs[i][j] = 0.01

    more_neurons = False
    weights_json, graph_json = {}, {}

    # epoch counter
    epoch = 0
    epochs_register = []
    # For the momentum implementation
    b = random.uniform(0, 1)

    # 1. Weights initialization
    weights_h, weights_o = weights_initialization(inputs, outputs, neurons_hidden_layer)
    weights_h_history, weights_o_history = [], []

    # Add the initial weights to the history for future implementation of momentum
    weights_h_history.append(weights_h)
    weights_o_history.append(weights_o)
    # 2. Bias initialization
    # Bias is an array of 2 arrays[[bias for the hidden layer], [bias for the output layer]]
    if bias == []:
        bias = [[0.99 for _ in range(neurons_hidden_layer)], [0.99 for _ in range(len(outputs[0]))]]

    # list of Errors by pattern initialized in precision value + 5
    errors_p = [precision + 5] * len(inputs)

    # Error total
    error_total = 0
    errors_by_epoch = []
    # Start the training
    # Training stops when the errors of each pattern is less than the precision value 
    while any([error > precision for error in errors_p]):

        # 0. Epoch counter increment
        epoch += 1
        # Pattern by pattern training
        for p in range(len(inputs)):
            if p == 0:
                epoch_r = ("***--------------------------------INICIO EPOCA--------------------------------***\n")
            epoch_r += (f"Patrón: {p}\n")
            # 3. Y calculation for the hidden layer
            y_h = output_hidden_calculation(inputs[p], bias[0], weights_h, neurons_hidden_layer, function_h_name)
            epoch_r += (f"y_h (Resultados obtenidos de la capa oculta):\n {y_h}\n")

            # 4. Y calculation for the output layer and partials error calculation for the output layer
            y_o, d_o = output_calculation(outputs[p],bias[1],y_h, weights_o, len(outputs[0]), function_o_name)
            epoch_r += (f"y_o (Resultados obtenidos de la capa de salida):\n {y_o}\n")
            epoch_r += (f"Errores de salida: {d_o}\n")
            # 5. Partials Error calculation for the hidden layer
            d_h = hidden_partial_error_calculation(d_o, weights_o, inputs[p], bias[0] ,function_h_name, neurons_hidden_layer)
            epoch_r += (f"Errores parciales de la capa oculta:\n {d_h}\n")

            # 6. Weights update for the output layer
            weights_o = update_output_weights(weights_o, bias[1] , d_o, y_h, alpha, b, weights_o_history)
            epoch_r += (f"Actualización de pesos de la capa de salida:\n {weights_o}\n")

            # 7. Weights update for the hidden layer
            weights_h = update_hidden_layer_weights(weights_h, bias[0], d_h, inputs[p], alpha, b, weights_h_history, neurons_hidden_layer)
            epoch_r += (f"Actualización de pesos de la capa oculta:\n {weights_h}\n")

            # 8. Cuadratic mean error calculation
            error_p = sum([d_o[k] ** 2 for k in range(len(d_o))]) * 1 / 2
            errors_p[p] = error_p
            epoch_r += (f"Error cuadrático medio del patron:\n {error_p}\n")

        # 9. Error total calculation
        error_total = sum(errors_p) / len(inputs)
        errors_by_epoch.append(error_total)
        epoch_r += (f"Errores de los patrones: {errors_p}\n")
        epoch_r += (f"Error total: {error_total}\n")
        epoch_r += ("***--------------------------------FINAL EPOCA--------------------------------***\n")

        if epoch > max_iterations:
            more_neurons = True
            break

        # Register the epoch
        epochs_register.append(epoch_r)
    
    if more_neurons:
        print("Numero maximo de iteraciones alcanzado, se aumentará el número de neuronas en la capa oculta")
        neurons_hidden_layer += 1
        bias[0].append(1)
        weights_json, graph_json = backpropagation_trianing(inputs, outputs, alpha, function_h_name, function_o_name, bias, neurons_hidden_layer, precision, max_iterations)
        return None
    
    weights_json = {
        "weights_h": weights_h,
        "weights_o": weights_o,
        "bias": bias,
        "alpha": alpha,
        "function_h_name": function_h_name,
        "function_o_name": function_o_name,
        "neurons_hidden_layer": neurons_hidden_layer,
        }

    graph_json = {
        "epochs": epoch,
        "errors_by_epoch": errors_by_epoch,
        "weights_h_history": weights_h_history,
        "weights_o_history": weights_o_history,
        "date_training": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs_register": epochs_register,
    }

    return weights_json, graph_json

def weights_initialization(inputs, outputs, neurons_hidden_layer):
    # 1.1 Random initialization of the weights for the inputs and hidden neurons connections
    # for _ in range(len(inputs[0]) + 1) is for the assigment of connections between the inputs 
    # and one neuron in the hidden layer
    # for _ in range(neurons_hidden_layer) is for each neuron in the hidden layer
    # Estructure [neuron hidden 1 -> [x_0, x_1, x_n], neuron hidden n -> [x_0, x_1, x_n]]
    # result: Example [[ 0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]

    weights_h = []
    for _ in range(neurons_hidden_layer):
        weights_one_neuron = []
        for _ in range(len(inputs[0]) + 1):
            w = random.uniform(-1, 1)
            w = w if w > 0.1 or w < -0.1 else 0.5
            weights_one_neuron.append(w)
        weights_h.append(weights_one_neuron)

    # 1.2 Random initialization of the weights for the hidden and output neurons connections
    qty_outputs = len(outputs[0])
    weights_o = []
    for _ in range(qty_outputs):
        weights_one_neuron = []
        for _ in range(neurons_hidden_layer + 1):
            w = random.uniform(-1, 1)
            w = w if w != 0 else 0.1
            weights_one_neuron.append(w)
        weights_o.append(weights_one_neuron)
    
    return weights_h, weights_o

def output_hidden_calculation(inputs, bias, weights_h, neurons_hidden_layer, function_name):

    # Select the function to calculate the output
    function = switch_function_output(function_name)

    # List of outputs for the hidden layer
    y_h = []

    for j in range(neurons_hidden_layer):
        # Calculate the net value
        net_h = sum([inputs[i] * weights_h[j][i] for i in range(1, len(inputs))]) + (bias[j] * weights_h[j][0])
        # Calculate the output value
        y = function(net_h)
        y_h.append(y)
    
    return y_h

def output_calculation(y_desired, bias, y_h, weights_o, qty_outputs, function_name):

    # Select the function to calculate the output
    function = switch_function_output(function_name)
    # Select the function to calculate the derivative of the output
    function_derivative = switch_function_derivative(function_name)

    # List of outputs for the output layer
    y_out = []
    # List of outputs for the hidden layer
    d_out = []

    for k in range(qty_outputs):
        # Calculate the net value
        net_o = sum([y_h[j-1] * weights_o[k][j] for j in range(1,len(weights_o[k]))])
        net_o += (bias[k] * weights_o[k][0])
        # Calculate the output value
        y_obtained = function(net_o)
        y_out.append(y_obtained)

        # Calculate the error
        error = y_desired[k] - y_obtained
        d = function_derivative(y_obtained) * error
        d_out.append(d)

    return y_out, d_out

def hidden_partial_error_calculation(d_out, weights_o, x, bias, function_name,qty_hidden_neurons):
    
    # Select the function to calculate the derivative of the output
    function_derivative = switch_function_derivative(function_name)
    
    # List of partial errors for the hidden layer
    d_h = []
    
    for j in range(qty_hidden_neurons):

        # Calculate the backpropagation = d_o(p,k) * w_o(k,j)
        backpropagation = sum([d_out[k] * weights_o[k][j+1] for k in range(len(d_out))])

        d_h_j = []

        # Calculate the partial error for each weight
        # First calculate the partial error for the bias
        d = function_derivative(bias[j]) * backpropagation
        d_h_j.append(d)

        for i in range(len(x)):
            d = function_derivative(x[i]) * backpropagation
            d_h_j.append(d)
        
        d_h.append(d_h_j)
            
    return d_h

def update_output_weights(weights_o, bias, d_o, y_h, alpha, b, weights_history):
    # List of new weights for the output layer
    weights = []

    #𝑊^𝑜 (𝑘,𝑗)= 𝑊^𝑜 (𝑘,𝑗) + alfa * 𝑑^𝑜 (𝑝,𝑘)*Y(p,j)
    for k in range(len(d_o)):
        # Update the weights
        w_k = []
        for j in range(len(y_h) + 1): # +1 for the bias
            # Calculate the momentum
            # momentum = b * (weights_history[-2][k][j] - weights_history[-1][k][j]) if len(weights_history) > 1 else 0
            
            # If j = 0, the weight is for the bias, then the calculation is using the bias value instead of the y_h value
            if j == 0:
                w = weights_o[k][j] + (alpha * d_o[k] * bias[k]) 
            # Otherwise, the calculation is using the y_h value 
            else:
                w = weights_o[k][j] + (alpha * d_o[k] * y_h[j-1]) 

            w_k.append(w)
        weights.append(w_k)

    return weights

def update_hidden_layer_weights(weights_h, bias, d_h, x, alpha, b, weights_history, qty_hidden_neurons):
    # List of new weights for the output layer
    weights = []

    #𝑊^ℎ (𝑗,𝑖)= 𝑊^ℎ (𝑗,𝑖) + alfa * 𝑑^ℎ (𝑝,j,i)*X(p,i)
    for j in range(qty_hidden_neurons):
        # Update the weights
        w_j = []
        for i in range(len(x) + 1):
            # Calculate the momentum
            #momentum = b * (weights_history[-2][j][i] - weights_history[-1][j][i]) if len(weights_history) > 1 else 0

            # If i = 0, the weight is for the bias, then the calculation is using the bias value instead of the x value
            if i == 0:
                w = weights_h[j][i] + (alpha * (sum(d_h[j])/len(d_h[j])) * bias[j]) 
            # Otherwise, the calculation is using the x value 
            else:
                w = weights_h[j][i] + (alpha * d_h[j][i] * x[i-1])

            w_j.append(w)

        weights.append(w_j)
    return weights

def sigmoid(x):
    return sp.expit(x)

def sigmoid_derivative(x):
    return x * (1 - x)

def aplication_backpropagation(weights_h, weights_o, bias, inputs, function_h_name, function_o_name, neurons):
    # Select the function to calculate the output
    function = switch_function_output(function_h_name)
    function_o = switch_function_output(function_o_name)

    y_output = [] 

    for p in range(len(inputs)):
        print(f"Patrón {p}")

        x = inputs[p]
        print(f"Entradas: {x}")

        # List of outputs for the hidden layer
        y_h = []

        for j in range(neurons):
            # Calculate the net value
            net_h = sum([x[i] * weights_h[j][i] for i in range(1, len(x))]) + (bias[0][j] * weights_h[j][0])
            # Calculate the output value
            y = function(net_h)
            y_h.append(y)
    
        # List of outputs for the output layer
        y_out = []

        for k in range(len(outputs[0])):
            # Calculate the net value
            net_o = sum([y_h[j-1] * weights_o[k][j] for j in range(1,len(weights_o[k]))])
            net_o += (bias[1][k] * weights_o[k][0])

            # Calculate the output value
            y_obtained = function_o(net_o)
            y_out.append(y_obtained)

        y_output.append(y_out)

        print(f"Salida deseada: {outputs[p]}")
        print(f"Salida: {y_out}")

    return y_obtained


if __name__ == "__main__":
    # Generate the second case data
    second_case_json = secondCase_Data()

    # Add the second case to the cases.json file
    add_or_replace_secon_case(second_case_json)

    # Read the data from the cases.json file
    with open(get_resource_path("cases.json"), "r") as file:
        cases_json = json.load(file)
    
    # Get the first case data
    case_0 = cases_json["case_0"]

    # Get the inputs and outputs from the first case
    inputs = case_0["inputs"]
    outputs = case_0["outputs"]

    # Training the neural network
    weights_json, graph_json = backpropagation_trianing(inputs, outputs, alpha=0.2,function_h_name="sigmoid",function_o_name="sigmoid", precision=0.001, max_iterations=25000, neurons_hidden_layer=2)

    # Print the epochs_register
    for epoch in graph_json["epochs_register"]:
        print(epoch)

    # Print the number of epochs
    print(f"Number of epochs: {graph_json['epochs']}")

    # Aplication of the neural network
    y_obtained = aplication_backpropagation(weights_json["weights_h"],
                                weights_json["weights_o"], weights_json["bias"], inputs,
                                  weights_json["function_h_name"], weights_json["function_o_name"],
                                    weights_json["neurons_hidden_layer"])



    # Save the weights and graph data in the files weights.json and graph.json
    with open(get_resource_path("weights.json"), "w") as file:
        json.dump(weights_json, file)
    
    with open(get_resource_path("graph.json"), "w") as file:
        json.dump(graph_json, file)
    