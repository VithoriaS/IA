import numpy as np

class Perceptron:
    # Inicializa o perceptron com o número de entradas, épocas e taxa de aprendizado
    def __init__(self, num_entradas, epocas=1000, taxa_aprendizado=0.01):
        self.pesos = np.zeros(num_entradas + 1)  # Inicializa os pesos com zeros
        self.epocas = epocas  # Define o número de épocas para treinamento
        self.taxa_aprendizado = taxa_aprendizado  # Define a taxa de aprendizado

    # Função de ativação (degrau) que retorna 1 se x >= 0, senão retorna 0
    def funcao_ativacao(self, x):
        return 1 if x >= 0 else 0

    # Função para prever a saída para entradas dadas
    def prever(self, entradas):
        somatorio = np.dot(entradas, self.pesos[1:]) + self.pesos[0]  # Calcula o somatório ponderado
        return self.funcao_ativacao(somatorio)  # Aplica a função de ativação

    # Função para treinar o perceptron usando as entradas e rótulos fornecidos
    def treinar(self, entradas_treino, rotulos):
        for _ in range(self.epocas):  # Itera sobre o número definido de épocas
            for entradas, rotulo in zip(entradas_treino, rotulos):  # Itera sobre cada exemplo de treino
                previsao = self.prever(entradas)  # Faz a previsão
                # Ajusta os pesos com base no erro (diferença entre rótulo e previsão)
                self.pesos[1:] += self.taxa_aprendizado * (rotulo - previsao) * entradas
                self.pesos[0] += self.taxa_aprendizado * (rotulo - previsao)

def criar_dados(n, func):
    # Gera dados de treinamento para operações lógicas (AND, OR, XOR)
    entradas_treino = np.array([list(map(int, "{:0{width}b}".format(i, width=n))) for i in range(2**n)])
    # Determina os rótulos com base na função lógica escolhida
    if func == "AND":
        rotulos = np.all(entradas_treino, axis=1).astype(int)
    elif func == "OR":
        rotulos = np.any(entradas_treino, axis=1).astype(int)
    elif func == "XOR":
        rotulos = np.array([0 if np.sum(x) % 2 == 0 else 1 for x in entradas_treino])
    else:
        raise ValueError("Função não reconhecida. Use 'AND', 'OR' ou 'XOR'.")
    return entradas_treino, rotulos

def imprimir_resultados(nome_funcao, entradas_treino, rotulos, previsoes):
    # Imprime os resultados do treinamento em um formato legível
    print(f"__{nome_funcao}___")
    print("Entradas\tEsperado\tPrevisto")
    print("-" * 30)
    for entrada, rotulo, previsao in zip(entradas_treino, rotulos, previsoes):
        entrada_str = ' '.join(map(str, entrada))
        print(f"{entrada_str}\t\t{rotulo}\t\t{previsao}")

# Abaixo estão os exemplos de uso do Perceptron para as operações lógicas XOR, OR e AND.
# Cada bloco cria dados de treinamento para a operação lógica correspondente,
# treina um Perceptron e imprime os resultados.

# Testando o Perceptron para XOR
entradas_treino, rotulos = criar_dados(2, "XOR")
perceptron_xor = Perceptron(2)
perceptron_xor.treinar(entradas_treino, rotulos)
previsoes_xor = np.array([perceptron_xor.prever(x) for x in entradas_treino])
imprimir_resultados("XOR", entradas_treino, rotulos, previsoes_xor)

# Testando o Perceptron para OR
entradas_treino, rotulos = criar_dados(4, "OR")
perceptron_or = Perceptron(4)
perceptron_or.treinar(entradas_treino, rotulos)
previsoes_or = np.array([perceptron_or.prever(x) for x in entradas_treino])
imprimir_resultados("OR", entradas_treino, rotulos, previsoes_or)

# Testando o Perceptron para AND
entradas_treino, rotulos = criar_dados(3, "AND")
perceptron_and = Perceptron(3)
perceptron_and.treinar(entradas_treino, rotulos)
previsoes_and = np.array([perceptron_and.prever(x) for x in entradas_treino])
imprimir_resultados("AND", entradas_treino, rotulos, previsoes_and)
