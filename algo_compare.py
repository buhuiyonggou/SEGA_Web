import networkx as nx
import community as community_louvain
import time
import matplotlib.pyplot as plt


def time_execution(graph, function, *args):
    start_time = time.time()
    function(graph, *args)
    end_time = time.time()
    return end_time - start_time


def create_random_graph(n, p):
    return nx.erdos_renyi_graph(n, p)


graph_sizes = [100, 200, 500, 1000]
pagerank_times = []
louvain_times = []
betweenness_times = []
greedy_modularity_times = []

for size in graph_sizes:
    G = create_random_graph(size, 0.05)

    # Time PageRank
    pagerank_times.append(time_execution(G, nx.pagerank))

    # Time Louvain
    louvain_times.append(time_execution(G, community_louvain.best_partition))

    # Time Betweenness Centrality
    betweenness_times.append(time_execution(G, nx.betweenness_centrality))

    # Time Greedy Modularity Community Detection
    greedy_modularity_times.append(time_execution(
        G, nx.algorithms.community.greedy_modularity_communities))

# Plotting the results
plt.plot(graph_sizes, pagerank_times, label='PageRank')
plt.plot(graph_sizes, louvain_times, label='Louvain')
plt.plot(graph_sizes, betweenness_times, label='Betweenness Centrality')
plt.plot(graph_sizes, greedy_modularity_times, label='Greedy Modularity')
plt.xlabel('Graph Size (Number of Nodes)')
plt.ylabel('Execution Time (Seconds)')
plt.title('Algorithm Scalability and Efficiency')
plt.legend()
plt.show()
