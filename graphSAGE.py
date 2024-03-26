import matplotlib
import pandas as pd
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

    def model_training(self, model, graph_data, num_labels,epoches):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        graph_data = graph_data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        def train():
                model.train()
                optimizer.zero_grad()
                out = model(graph_data.x, graph_data.edge_index)
                # Only use nodes with labels available for loss calculation --> mask
                # unique_labels = torch.unique(graph_data.y[graph_data.train_mask])
                # print("Unique labels:", unique_labels)
                # print("Expected range of labels:", list(range(num_labels)))
                loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
                loss.backward()
                optimizer.step()
                return loss
        
        def test():
            model.eval()
            out = model(graph_data.x, graph_data.edge_index)
            # Check against ground-truth labels.
            pred = out.argmax(dim=1)

            test_correct = pred[graph_data.test_mask] == graph_data.y[graph_data.test_mask]
            # Derive ratio of correct predictions.
            test_acc = int(test_correct.sum()) / int(graph_data.test_mask.sum())
            return test_acc

        losses = []
        for epoch in range(0, epoches + 1):
            loss = train()
            losses.append(loss)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {(epoch + 1):03d}, Loss: {loss:.6f}')
        test_acc = test()
        print(f'Test Accuracy: {test_acc:.6f}')
        
        model.eval()
        with torch.no_grad():
            embeddings = model(graph_data.x, graph_data.edge_index)

        return embeddings