import matplotlib.pyplot as plt
import networkx as nx

# Definir uma topologia simples com menos neurônios
layers = [3, 4, 1]  # 3 neurônios na camada de entrada, 4 na camada oculta, 1 na saída

# Criar o gráfico direcionado (fluxograma)
G = nx.DiGraph()

# Adicionar nós para cada neurônio, com rótulos indicando a camada e o número do neurônio
for i in range(len(layers)):
    for j in range(layers[i]):
        G.add_node(f"Layer {i+1} - Neuron {j+1}")

# Conectar os nós (camadas anteriores para camadas seguintes)
for i in range(len(layers) - 1):
    for j in range(layers[i]):
        for k in range(layers[i+1]):
            G.add_edge(f"Layer {i+1} - Neuron {j+1}", f"Layer {i+2} - Neuron {k+1}")

# Visualizar o gráfico no estilo fluxograma
plt.figure(figsize=(8, 8))

# Organizar os nós de maneira sequencial para dar a aparência de um fluxograma
pos = {}
layer_positions = [0, 1, 2]  # Posições horizontais das camadas
for i, layer_size in enumerate(layers):
    # Coloca os neurônios de cada camada em posições verticais
    pos.update({f"Layer {i+1} - Neuron {j+1}": (layer_positions[i], -j) for j in range(layer_size)})

# Desenhando os nós e as conexões
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='black', arrows=True)

# Configurações de título e visualização
plt.title("Topologia da Rede Neural (Diagrama de Fluxo)", size=15)
plt.axis("off")
plt.show()
