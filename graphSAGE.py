import logging
import os
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from sklearn.manifold import TSNE

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def model_training(self, model, graph_data,epoches):
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
        print(f'Test Accuracy: {test_acc * 100:.4f} %')
        
        model.eval()
        with torch.no_grad():
            embeddings = model(graph_data.x, graph_data.edge_index)

        return embeddings
    
    def visualize_embeddings(self, embeddings, tensor_labels, labels, plot_folder):
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())
        
        plt.figure(figsize=(12, 10))
        unique_labels = torch.unique(tensor_labels).cpu().numpy()
        # if more than 10 labels, use only first 10
        if len(unique_labels) > 10:
            unique_labels = unique_labels[:10]

        colors = plt.cm.tab20.colors
        for i, l in enumerate(unique_labels):
            plt.scatter(embeddings_2d[tensor_labels == l, 0], embeddings_2d[tensor_labels == l, 1], c=[colors[i]], label=labels[i])
        plt.legend(loc='lower left', title='Label Names')
        plt.title('t-SNE visualization of embeddings')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.savefig(plot_folder + '/tsne_plot_graphSAGE.png')
        plt.close()

            
    