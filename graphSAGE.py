import pandas as pd
import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    def contrastive_loss(self, out, edge_index, num_neg_samples=None):
        # Positive samples: directly connected nodes
        pos_loss = F.pairwise_distance(
            out[edge_index[0]], out[edge_index[1]]).pow(2).mean()

        # Negative sampling: randomly select pairs of nodes that are not directly connected
        num_nodes = out.size(0)
        # Default to the same number of negative samples as positive
        num_neg_samples = num_neg_samples or edge_index.size(1)
        neg_edge_index = torch.randint(
            0, num_nodes, (2, num_neg_samples), dtype=torch.long, device=out.device)

        # Compute loss for negative samples
        neg_loss = F.relu(
            1 - F.pairwise_distance(out[neg_edge_index[0]], out[neg_edge_index[1]])).pow(2).mean()

        # Combine positive and negative loss
        loss = pos_loss + neg_loss
        return loss

    def model_training(self, feature_index, edge_index, epoches):
        # Initialize the GraphSAGE model
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        data = Data(x=feature_index, edge_index=edge_index)

        # Training loop
        for epoch in range(epoches):  # Adjust the number of epochs as needed
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)

            # Use the contrastive loss function here
            loss = self.contrastive_loss(out, data.edge_index)

            loss.backward()
            optimizer.step()

        # Generate embeddings for nodes without gradient calculations
        self.eval()  # Switch to evaluation mode
        with torch.no_grad():
            embeddings = self(data.x, data.edge_index)

            new_weights = torch.norm(
                embeddings[data.edge_index[0]] - embeddings[data.edge_index[1]], dim=1)

        # Initialize the scaler
        scaler = MinMaxScaler()

        # Reshape new_weights for scaling - sklearn's MinMaxScaler expects a 2D array
        weights_reshaped = new_weights.numpy().reshape(-1, 1)

        # Apply the scaler to the weights
        scaled_weights = scaler.fit_transform(weights_reshaped).flatten()

        return scaled_weights

    def data_reshape(self, scaled_weights, edge_index, index_to_name_mapping):
        # Create a DataFrame to export
        edges_with_weights = pd.DataFrame(
            edge_index.t().numpy(), columns=['source', 'target'])

        # Update the DataFrame with scaled weights
        edges_with_weights['weight'] = scaled_weights

        # Use id to map names
        edges_with_weights['source'] = edges_with_weights['source'].apply(
            lambda x: index_to_name_mapping.loc[x, 'name'])
        edges_with_weights['target'] = edges_with_weights['target'].apply(
            lambda x: index_to_name_mapping.loc[x, 'name'])

        return edges_with_weights