import pandas as pd
import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

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
        num_neg_samples = num_neg_samples or edge_index.size(1)
        neg_edge_index = torch.randint(
            0, num_nodes, (2, num_neg_samples), dtype=torch.long, device=out.device)

        neg_loss = F.relu(
            1 - F.pairwise_distance(out[neg_edge_index[0]], out[neg_edge_index[1]])).pow(2).mean()

        loss = pos_loss + neg_loss
        return loss

    def model_training(self, model, device, feature_index, edge_index, epoches):
        data = Data(x=feature_index, edge_index=edge_index)
        
        data = data.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()
        
        # Training loop
        for epoch in range(epoches):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = model.contrastive_loss(out, data.edge_index)

            # backup for supervised learning
            # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward() 
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)

        return embeddings

    def data_reshape(self, scaled_weights, edge_index, index_to_name_mapping):
        # Create a DataFrame to exports
        edges_with_weights = pd.DataFrame(
            edge_index.t().numpy(), columns=['Source', 'Target'])

        # Update the DataFrame with scaled weights
        edges_with_weights['Weight'] = scaled_weights

        # Use id to map names
        edges_with_weights['Source'] = edges_with_weights['Source'].apply(
            lambda x: index_to_name_mapping.loc[x, 'name'])
        edges_with_weights['Target'] = edges_with_weights['Target'].apply(
            lambda x: index_to_name_mapping.loc[x, 'name'])

        return edges_with_weights