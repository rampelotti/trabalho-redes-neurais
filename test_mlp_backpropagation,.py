import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

# Carregar os dados de treino e teste
X_train = pd.read_csv('X_train_scaled.csv', header=None, delimiter=' ')
y_train = pd.read_csv('x_train.csv', header=None, delimiter=' ')
X_test = pd.read_csv('X_test_scaled.csv', header=None, delimiter=' ')
y_test = pd.read_csv('y_test.csv', header=None, delimiter=' ')

# Ajustar a estrutura de y_train e y_test para formato correto
y_train = y_train.values.ravel()  # Transformar de DataFrame para array
y_test = y_test.values.ravel()    # Transformar de DataFrame para array

print(X_train.shape)  # Verifique as dimensões do conjunto de treinamento
print(X_test.shape)   # Verifique as dimensões do conjunto de teste

# Definir a rede neural (MLP) sem sample_weight
mlp = MLPClassifier(hidden_layer_sizes=(50, 30), 
                    max_iter=2000,         # Aumentar o número de iterações
                    random_state=42, 
                    solver='adam',         # Usar o otimizador Adam
                    learning_rate_init=0.001)  # Definir a taxa de aprendizado

# Treinar o modelo
mlp.fit(X_train, y_train)

# Prever os resultados
y_pred = mlp.predict(X_test)

# Relatório de classificação
print(classification_report(y_test, y_pred))

# Acurácia
accuracy = mlp.score(X_test, y_test)
print("Acurácia do modelo:", accuracy)

#
# Exibir a topologia da rede neural
print("Topologia da rede neural:")
print("Camadas ocultas (neurônios por camada):", mlp.hidden_layer_sizes)

# Exibir os coeficientes (pesos) das camadas da rede neural
print("\nPesos das camadas:")
for i, coef in enumerate(mlp.coefs_):
    print(f"Camada {i + 1}: {coef.shape}")