import pandas as pd
import numpy as np
from itertools import combinations
import torch
from flask import session
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from graphSAGE import GraphSAGE

class GraphSAGEProcessor:
    def __init__(self, nodes_folder, edges_folder = None):
        self.nodes_folder = nodes_folder
        self.edges_folder = edges_folder
        
        self.CONNECT_AMONG_SUB_DEPART = 0.25
        self.CONNECT_AMONG_DEPARTMENT = 0.05
        self.CONNECT_AMONG_ORGANIZATION= 0.01
        self.COLUMN_ALIGNMENT = {
            'id':['id','emp_id', 'employeeid'],
            'name': ['name', 'full_name', "employee_name", 'first_name'],
            'department': ['department', 'depart', 'sector'],
            'sub-depart':['sub-depart', 'sub-department', 'sub-sector','second-department'],
            'manager': ['manager', 'supervisor', 'manager_name','managername']
        }
        self.NODE_FEATURES = []
        self.epoches = 200

    def load_data(self, file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')  # Handles BOM if present
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

        df.columns = df.columns.str.lower()
        return df

    def fetch_data_from_user(self, file_path):
        if file_path is None:
            return
        hr_data = self.load_data(file_path)
        hr_data.fillna("Missing", inplace=True)
        
        return hr_data

    # Function to rename columns based on expected variations
    def rename_columns_to_standard(self, df, column_alignment):
        renamed_columns = df.columns.tolist()  # Start with the original column names
        for standard_name, variations in column_alignment.items():
            for variation in variations:
                # Find which variation is in the DataFrame columns (case-insensitive match)
                found_columns = [col for col in df.columns if col.lower() == variation.lower()]
                if found_columns:
                    # Assume only one match is found; rename it to the standard name
                    index_of_found = renamed_columns.index(found_columns[0])
                    renamed_columns[index_of_found] = standard_name
                    break  # Stop looking for other variations if one is found
        return renamed_columns

    def create_index_id_name_mapping(self, hr_data):
        index_to_id_name_mapping = [{
            'index': i,
            'id': row['id'],
            'name': row['name'].title()
        } for i, row in hr_data.iterrows()]

        mapping_df = pd.DataFrame(index_to_id_name_mapping)
        return mapping_df

    def preprocess_data(self, df, node_features):
        # Initialize LabelEncoder and MinMaxScaler
        le = LabelEncoder()
        scaler = MinMaxScaler()
        imputer = SimpleImputer(strategy='mean')  
        for column in node_features:
            if df[column].dtype == 'object':
                df[column] = le.fit_transform(df[column].astype(str))  # Convert categorical data to numerical
            else:
                # Reshape the column data to a 2D array for imputer and scaler
                column_data_reshaped = df[column].values.reshape(-1, 1)  # Reshape data
                # Apply imputer to the reshaped data
                imputed_data = imputer.fit_transform(column_data_reshaped)
                # Apply scaler to the imputed and reshaped data
                df[column] = scaler.fit_transform(imputed_data)

        return df

    # Re-defining the custom function to adapt to the new logic
    def manage_edge_probability(self, edges, hr_data, dept_indices, sub_dept_info=False):
        for i, j in combinations(dept_indices, 2):
            emp_i = hr_data.iloc[i]
            emp_j = hr_data.iloc[j]
            if sub_dept_info:  # When there's detail on 'sub-depart'
                if (emp_i['department'] == emp_j['department']) and (emp_i['sub-depart'] == emp_j['sub-depart']):
                    if np.random.rand() < self.CONNECT_AMONG_SUB_DEPART:  # Same department and 'sub-depart'
                        edges.append([i, j])
                elif emp_i['department'] == emp_j['department']:
                    if np.random.rand() < self.CONNECT_AMONG_DEPARTMENT:  # Same department but not 'sub-depart'
                        edges.append([i, j])
            else:  # Defaults to handling by 'department' with chance
                if emp_i['department'] == emp_j['department']:
                    if np.random.rand() < self.CONNECT_AMONG_DEPARTMENT:
                        edges.append([i, j])


    # if edges are given by user, relationship_data is not None, mapping current users to edges
    def edges_generator(self, hr_data, edge_filepath = None):
        edges = []
        mapping_df = self.create_index_id_name_mapping(hr_data)

        if edge_filepath:
            # mapping name to index and generating edges
            hr_edge = self.fetch_data_from_user(edge_filepath)
            for _, row in hr_edge.iterrows():
                source_id = row['source']
                target_id = row['target']
                source_index = mapping_df[mapping_df['id'] == source_id].index.item()
                target_index = mapping_df[mapping_df['id'] == target_id].index.item()

                edges.append([source_index, target_index])
        else:
            # if edges are not given, infer the edges
            if 'sub-depart' in hr_data.columns and hr_data['sub-depart'].notnull().any():
                for dept in hr_data['sub-depart'].unique():
                    dept_indices = hr_data[hr_data['sub-depart'] == dept].index.tolist()
                    self.manage_edge_probability(edges, hr_data,dept_indices, sub_dept_info=True)
            else:
                for dept in hr_data['department'].unique():
                    dept_indices = hr_data[hr_data['department'] == dept].index.tolist()
                    self.manage_edge_probability(edges, hr_data, dept_indices, sub_dept_info=False)
        return edges

    def edge_index_generator(self,edges):
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def features_generator(self, hr_data, node_features):
        hr_data_parsed = self.preprocess_data(hr_data, node_features)
        # Exclude 'id' and 'name' columns from features
        feature_columns = [col for col in hr_data.columns if col in node_features]
        
        features_data = hr_data_parsed[feature_columns].values
        
        return features_data

    def feature_index_generator(self, features):
        feature_index = torch.tensor(features, dtype=torch.float)
        
        return feature_index

    def nanCheck(self, hr_data, feature_index):
        # Check for NaN values in features
        if torch.isnan(feature_index).any():
           nan_columns = hr_data.columns[hr_data.isnull().any()].tolist()
           raise ValueError(f"NaN values detected in columns: {', '.join(nan_columns)}")
        return "No NaN values detected."

    def contrastive_loss(self, out, edge_index, num_neg_samples=None):
        # Positive samples: directly connected nodes
        pos_loss = F.pairwise_distance(out[edge_index[0]], out[edge_index[1]]).pow(2).mean()

        # Negative sampling: randomly select pairs of nodes that are not directly connected
        num_nodes = out.size(0)
        num_neg_samples = num_neg_samples or edge_index.size(1)  # Default to the same number of negative samples as positive
        neg_edge_index = torch.randint(0, num_nodes, (2, num_neg_samples), dtype=torch.long, device=out.device)

        # Compute loss for negative samples
        neg_loss = F.relu(1 - F.pairwise_distance(out[neg_edge_index[0]], out[neg_edge_index[1]])).pow(2).mean()

        # Combine positive and negative loss
        loss = pos_loss + neg_loss
        return loss
    
    
    def model_training(self, feature_index, edge_index):
        # Initialize the GraphSAGE model
        model = GraphSAGE(in_channels=feature_index.shape[1], hidden_channels=16, out_channels=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        data = Data(x=feature_index, edge_index=edge_index)

        # Training loop
        for epoch in range(self.epoches):  # Adjust the number of epochs as needed
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            
            # Use the contrastive loss function here
            loss = self.contrastive_loss(out, data.edge_index)
            
            loss.backward()
            optimizer.step()

            # if epoch == self.epochs - 1:  # Check if it's the last epoch
            #     session['training_progress'] = "complete"
            # else:
            #     session['training_progress'] = f"Epoch {epoch + 1}/{self.epochs}"
            # session.modified = True

        # Generate embeddings for nodes without gradient calculations
        model.eval()  # Switch to evaluation mode
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)

            new_weights = torch.norm(embeddings[data.edge_index[0]] - embeddings[data.edge_index[1]], dim=1)

        # Initialize the scaler
        scaler = MinMaxScaler()

        # Reshape new_weights for scaling - sklearn's MinMaxScaler expects a 2D array
        weights_reshaped = new_weights.numpy().reshape(-1, 1)

        # Apply the scaler to the weights
        scaled_weights = scaler.fit_transform(weights_reshaped).flatten()
        
        return scaled_weights

    def data_reshape(self, scaled_weights,edge_index, index_to_name_mapping):
        # Create a DataFrame to export
        edges_with_weights = pd.DataFrame(edge_index.t().numpy(), columns=['source', 'target'])

        # Update the DataFrame with scaled weights
        edges_with_weights['weight'] = scaled_weights

        # Use id to map names
        edges_with_weights['source'] = edges_with_weights['source'].apply(lambda x: index_to_name_mapping.loc[x, 'name'])
        edges_with_weights['target'] = edges_with_weights['target'].apply(lambda x: index_to_name_mapping.loc[x, 'name'])

        return edges_with_weights