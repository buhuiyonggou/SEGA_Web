import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import torch.nn.functional as F
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

    def generate_tsne_plot(self, embeddings, departments_list, file_path='tsne_plot_graphSAGE.png'):
        departments_array = np.array(departments_list)

        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(16, 10))
        for dept in set(departments_array):
            idx = departments_array == dept
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=dept)
        plt.legend()
        plt.title("GraphSAGE Embeddings Visualized by Department")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        
        plt.savefig(file_path)
        plt.close()