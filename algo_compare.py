import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from node2vec import Node2Vec
import torch
from torch_geometric.datasets import Planetoid
from graphSAGE import GraphSAGE
import matplotlib
matplotlib.use('TkAgg')


def time_execution(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


# Prepare a graph dataset (using PyTorch Geometric for this example)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Convert to NetworkX graph for Node2Vec
G = nx.Graph()
G.add_edges_from(data.edge_index.numpy().T)
G.add_nodes_from(range(data.num_nodes))

# Generate Node2Vec embeddings and time the execution
node2vec_time, model_n2v = time_execution(
    Node2Vec, G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Generate GraphSAGE embeddings and time the execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_sage = GraphSAGE(dataset.num_node_features,
                       hidden_channels=64, out_channels=dataset.num_classes)
model_sage = model_sage.to(device)
data = data.to(device)
graphsage_time, _ = time_execution(
    model_sage.model_training, model_sage, data, epoches=200)

# Performance data (dummy data for demonstration purposes, please replace with actual timings)
# Create a similar linear relationship for the graph as seen in your uploaded image
graph_sizes = [100, 200, 300, 400, 500]
# Assuming linear growth similar to your image
node2vec_performance = [t * (node2vec_time/35.0) for t in graph_sizes]
# Constant time as seen in your image
graphsage_performance = [t * (graphsage_time/5.0) for t in graph_sizes]

# Plotting the performance comparison
plt.figure(figsize=(10, 6))
plt.plot(graph_sizes, node2vec_performance, label='Node2Vec', color='blue')
plt.plot(graph_sizes, graphsage_performance, label='GraphSAGE', color='orange')
plt.xlabel('Graph Size (Number of Nodes)')
plt.ylabel('Execution Time (Seconds)')
plt.title('Performance Comparison of Node2Vec and GraphSAGE')
plt.legend()
plt.show()
