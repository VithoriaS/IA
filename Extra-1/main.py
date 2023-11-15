import numpy as np

# Função de ativação sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Função de ativação tangente hiperbólica (tanh) e sua derivada
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

# Classe para a rede neural
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # Bias para a camada oculta e de saída
        self.bias1 = np.random.rand(1, hidden_nodes)
        self.bias2 = np.random.rand(1, output_nodes)
      
        # Pesos para as conexões entre camada de entrada e camada oculta, e entre camada oculta e camada de saída
        self.weights1 = np.random.rand(self.input_nodes, self.hidden_nodes)
        self.weights2 = np.random.rand(self.hidden_nodes, self.output_nodes)
      
        self.learning_rate = learning_rate

         # Bias
    # def feedforward(self, input_data):
    #     self.layer1 = sigmoid(np.dot(input_data, self.weights1) + self.bias1)
    #     self.output = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
    #     return self.output

    # def feedforward(self, input_data):
    #   self.layer1 = sigmoid(np.dot(input_data, self.weights1))
    #   self.output = sigmoid(np.dot(self.layer1, self.weights2))
    #   return self.output

    def feedforward(self, input_data):
        # Calcula a ativação da camada oculta usando a função de ativação tanh
        self.layer1 = tanh(np.dot(input_data, self.weights1) + self.bias1)
        # Calcula a ativação da camada de saída usando a função de ativação tanh
        self.output = tanh(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.output

    def backprop(self, input_data, output_data):
        # Calcula o erro da camada de saída e o delta associado
        self.output_error = output_data - self.output
        self.output_delta = self.output_error * sigmoid_derivative(self.output)

        # Calcula o erro da camada oculta e o delta associado
        self.layer1_error = self.output_delta.dot(self.weights2.T)
        self.layer1_delta = self.layer1_error * sigmoid_derivative(self.layer1)

        # Atualiza os pesos e os bias usando o gradiente descendente
        self.weights1 += self.learning_rate * input_data.T.dot(self.layer1_delta)
        self.weights2 += self.learning_rate * self.layer1.T.dot(self.output_delta)

# Funções para gerar dados lógicos (AND, OR, XOR)
def generate_AND_data(input_nodes):
    inputs = np.array([np.array([int(x) for x in format(i, f'0{input_nodes}b')]) for i in range(2 ** input_nodes)])
    outputs = np.array([[int(np.all(input_))] for input_ in inputs])
    return inputs, outputs

def generate_OR_data(input_nodes):
    inputs = np.array([np.array([int(x) for x in format(i, f'0{input_nodes}b')]) for i in range(2 ** input_nodes)])
    outputs = np.array([[int(np.any(input_))] for input_ in inputs])
    return inputs, outputs

def generate_XOR_data(input_nodes):
    inputs = np.array([np.array([int(x) for x in format(i, f'0{input_nodes}b')]) for i in range(2 ** input_nodes)])
    outputs = np.array([[int(sum(input_) % 2 == 1)] for input_ in inputs])
    return inputs, outputs

# Entrada do usuário para configurar a rede neural
input_nodes = int(input("Enter the number of inputs for the logic function (2 or 10): "))
hidden_nodes = 6 if input_nodes == 2 else 20
output_nodes = 1

learning_rate = float(input("Enter the learning rate (e.g., 0.1, 0.01): "))
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

logic_function = input("Enter 'AND', 'OR', or 'XOR' to train the network: ").strip().upper()
if logic_function == 'AND':
    X, y = generate_AND_data(input_nodes)
elif logic_function == 'OR':
    X, y = generate_OR_data(input_nodes)
elif logic_function == 'XOR':
    X, y = generate_XOR_data(input_nodes)
else:
    raise ValueError("Invalid logic function. Choose 'AND', 'OR', or 'XOR'.")

# Treinamento da rede neural
for i in range(3000):
    output = nn.feedforward(X)
    nn.backprop(X, y)

# Teste da rede neural
print("Test:")
for x in X:
    print(f"Input: {x}, Output: {nn.feedforward(x)}")
