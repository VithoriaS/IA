import numpy as np

class Perceptron:
    def __init__(self, n_inputs, epochs=1000, learning_rate=0.01):
        self.weights = np.zeros(n_inputs + 1) 
        self.epochs = epochs
        self.learning_rate = learning_rate

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_fn(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

def create_data(n, func):
    training_inputs = np.array([list(map(int, "{:0{width}b}".format(i, width=n))) for i in range(2**n)])
    if func == "AND":
        labels = np.all(training_inputs, axis=1).astype(int)
    elif func == "OR":
        labels = np.any(training_inputs, axis=1).astype(int)
    else:
        raise ValueError("Function not recognized. Use 'AND' or 'OR'.")
    return training_inputs, labels
  
#função create_data para incluir a operação XOR
def create_data_xor(n):
    training_inputs = np.array([list(map(int, "{:0{width}b}".format(i, width=n))) for i in range(2**n)])
    labels = np.array([0 if np.sum(x) % 2 == 0 else 1 for x in training_inputs])
    return training_inputs, labels
training_inputs, labels = create_data_xor(2)

perceptron_xor = Perceptron(2)
perceptron_xor.train(training_inputs, labels)
predictions_xor = np.array([perceptron_xor.predict(x) for x in training_inputs])
print("__XOR___")
print("Inputs\tExpected\tPredicted")
print("-" * 30)
for i, (inp, label, prediction) in enumerate(zip(training_inputs, labels, predictions_xor)):
    inputs_str = ' '.join(map(str, inp))
    print(f"{inputs_str}\t\t{label}\t\t{prediction}")

# Vamos verificar também se todas as previsões são corretas
all_predictions_correct = np.all(predictions_xor == labels)
all_predictions_correct, predictions_xor, labels

# Criar dados de treinamento para a funçãoa OR com 4 entradas
training_inputs, labels = create_data(4, "OR")

# Instanciar e treinar o Perceptron
perceptron = Perceptron(4)
perceptron.train(training_inputs, labels)

# Testar o Perceptron
predictions = np.array([perceptron.predict(x) for x in training_inputs])

# Imprimir cabeçalho da tabela
print("__OR___")
print("Inputs\t\tExpected\tPredicted")
print("-" * 30)

# Iterar sobre cada conjunto de entrada, rótulo e previsão e imprimi-los
for i, (inp, label, prediction) in enumerate(zip(training_inputs, labels, predictions)):
    # Converter a entrada de array para string para impressão
    inputs_str = ' '.join(map(str, inp))
    print(f"{inputs_str}\t\t{label}\t\t{prediction}")

# Criar dados de treinamento para a função AND com 3 entradas
training_inputs, labels = create_data(3, "AND")

# Instanciar e treinar o Perceptron
perceptron = Perceptron(3)
perceptron.train(training_inputs, labels)

# Testar o Perceptron
predictions = np.array([perceptron.predict(x) for x in training_inputs])

# Imprimir cabeçalho da tabela
print("\n__AND___")
print("Inputs\t\tExpected\tPredicted")
print("-" * 33)

# Iterar sobre cada conjunto de entrada, rótulo e previsão e imprimi-los
for i, (inp, label, prediction) in enumerate(zip(training_inputs, labels, predictions)):
    # Converter a entrada de array para string para impressão
    inputs_str = ' '.join(map(str, inp))
    print(f"{inputs_str}\t\t\t{label}\t\t\t{prediction}")
