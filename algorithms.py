import networkx as nx


def calculate_centrality(G, algorithm='pagerank'):
    if algorithm == 'pagerank':
        return nx.pagerank(G, weight='weight')
    elif algorithm == 'betweenness':
        return nx.betweenness_centrality(G, weight='weight')
    elif algorithm == 'closeness':
        return nx.closeness_centrality(G, distance='weight')
    elif algorithm == 'degree':
        return nx.degree_centrality(G)
    elif algorithm == 'eigenvector':
        return nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, weight='weight')
    # elif algorithm == 'katz':
    #     return nx.katz_centrality(G, weight='weight', max_iter=1000, alpha=0.005)
    elif algorithm == 'load':
        return nx.load_centrality(G, weight='weight')
    # elif algorithm == 'harmonic':
    #     return nx.harmonic_centrality(G, distance='weight')
    else:
        raise ValueError(f"Unsupported centrality algorithm: {algorithm}")


def detect_communities(G, algorithm='louvain'):
    if algorithm == 'louvain':
        return nx.community.louvain_communities(G)
    elif algorithm == 'greedy_modularity':
        return list(nx.algorithms.community.greedy_modularity_communities(G))
    elif algorithm == 'label_propagation':
        return list(nx.algorithms.community.label_propagation_communities(G))
    elif algorithm == 'fluid':

        k = int(len(G.nodes()) ** 0.5)
        return list(nx.algorithms.community.asyn_fluidc(G, k))
    elif algorithm == 'girvan_newman':
        comp = nx.algorithms.community.girvan_newman(G)
        return list(next(comp))
    else:
        raise ValueError(f"Unsupported community detection algorithm: {algorithm}")