from torch_geometric.nn import Node2Vec as PyGNode2Vec
import torch

class Node2VecProcessor(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, walk_length, context_size, walks_per_node, p, q):
        super(Node2VecProcessor, self).__init__()
        self.node2vec = PyGNode2Vec(
            num_nodes, embedding_dim, walk_length, context_size, walks_per_node, p, q)

    def forward(self, edge_index):
        return self.node2vec(edge_index)

    def train_model(self, edge_index, num_epochs):
        self.train()
        optimizer = torch.optim.Adam(self.parameters())
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.node2vec.loss(edge_index)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        return self.node2vec.embedding.weight.data

    def process_data_for_node2vec(self, hr_data, edge_filepath):
        # Apply similar preprocessing as in GraphSAGEProcessor
        hr_data = self.rename_columns_to_standard(
            hr_data, self.COLUMN_ALIGNMENT)
        index_to_name_mapping = self.create_index_id_name_mapping(hr_data)

        # Generate features and edges similar to GraphSAGE
        features = self.features_generator(hr_data, self.NODE_FEATURES)
        feature_index = self.feature_index_generator(features)
        edges = self.edges_generator(hr_data, edge_filepath)
        edge_index = self.edge_index_generator(edges)

        return feature_index, edge_index, index_to_name_mapping