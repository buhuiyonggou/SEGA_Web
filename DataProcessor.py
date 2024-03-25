import pandas as pd
import numpy as np
from itertools import combinations
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

class DataProcessor:
    def __init__(self, nodes_folder, edges_folder=None):
        self.nodes_folder = nodes_folder
        self.edges_folder = edges_folder
        
        self.UPPER_RATIO = 0.25
        self.LOWER_RATIO = 0.03
        self.COLUMN_ALIGNMENT = {
            'id': ['id', 'emp_id', 'employeeid'],
            'name': ['name', 'full_name', "employee_name", 'first_name'],
        }
        self.NODE_FEATURES = []
        self.epoches = 200

    def fetch_data_from_user(self, file_path):
        if file_path is None:
            raise ValueError("File path is None")

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

        df.columns = df.columns.str.lower()
        df.fillna("Missing", inplace=True)  # Handle missing values

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Loaded data is not a DataFrame")

        return df

    def rename_columns_to_standard_graphSAGE(self, df, column_alignment):
        df_copy = df.copy()    
        
        for standard_name, variations in column_alignment.items():
            for variation in variations:
                found_columns = [col for col in df_copy.columns if col.lower() == variation.lower()]
                if found_columns:
                    df_copy.rename(columns={found_columns[0]: standard_name}, inplace=True)
                    break  # Stop looking for other variations if one is found
        if not isinstance(df_copy, pd.DataFrame):
            logging.error(
                "rename_columns_to_standard_1 is not returning a DataFrame")
        return df_copy

    # Function to rename columns based on expected variations
    def rename_columns_to_standard_node2vec(self, df, column_alignment):
        # Dictionary to hold new column names
        new_column_names = {}

        for standard_name, variations in column_alignment.items():
            for variation in variations:
                if variation in df.columns:
                    new_column_names[variation] = standard_name
                    break  # once a variation is found, break out of the loop

        # Apply the renaming
        df.rename(columns=new_column_names, inplace=True)

        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':  # For non-numeric columns
                df[col] = df[col].fillna('Missing')
            else:  # For numeric columns
                df[col] = df[col].fillna(df[col].mean())

        return df

    def create_index_id_name_mapping(self, node_data, edge_infer_column=None):
        index_to_id_name_mapping = []

        for i, row in node_data.iterrows():
            mapping_entry = {
                'index': i,
                'id': row['id'],
                'name': row['name'].title() if 'name' in row else 'Unknown'
            }
            if edge_infer_column and edge_infer_column in node_data.columns:
                mapping_entry[edge_infer_column] = row[edge_infer_column]
            index_to_id_name_mapping.append(mapping_entry)

        mapping_df = pd.DataFrame(index_to_id_name_mapping)
        return mapping_df
    # Re-defining the custom function to adapt to the new logic
    def manage_edge_probability(self, edges, node_data, dept_indices, edge_infer):
        for i, j in combinations(dept_indices, 2):
            emp_i = node_data.iloc[i]
            emp_j = node_data.iloc[j]
            if emp_i[edge_infer] == emp_j[edge_infer]:
                if np.random.rand() < self.UPPER_RATIO:  # Same department but not 'sub-depart'
                    edges.append([i, j])
            else:  # Defaults to handling by 'department' with chance
                if np.random.rand() < self.LOWER_RATIO:
                        edges.append([i, j])

    # if edges are given by user, relationship_data is not None, mapping current users to edges

    def edges_generator(self, node_data, edge_infer, edge_filepath=None):
        edges = []
        mapping_df = self.create_index_id_name_mapping(node_data, edge_infer)

        if edge_filepath:
            # mapping name to index and generating edges
            edge_data = self.fetch_data_from_user(edge_filepath)
            for _, row in edge_data.iterrows():
                source_id = row['source']
                target_id = row['target']
                source_index = mapping_df[mapping_df['id']
                                          == source_id].index.item()
                target_index = mapping_df[mapping_df['id']
                                          == target_id].index.item()

                edges.append([source_index, target_index])
        else:
            if edge_infer is None:
                logging.error("Edge infer column is None")
            for each in node_data[edge_infer].unique():
                column_indices = node_data[node_data[edge_infer]
                                        == each].index.tolist()
                self.manage_edge_probability(
                    edges, node_data, column_indices, edge_infer)
        return edges

    def edge_index_generator(self, edges):
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def numeric_dataset(self, df, node_features, label_column):
        # Initialize LabelEncoder and MinMaxScaler
        le = LabelEncoder()
        scaler = MinMaxScaler()
        imputer = SimpleImputer(strategy='mean')
        for column in node_features:
            if df[column].dtype == 'object':
                df[column] = le.fit_transform(df[column].astype(str))
            else:
                column_data_reshaped = df[column].values.reshape(-1, 1)
                imputed_data = imputer.fit_transform(column_data_reshaped)
                df[column] = scaler.fit_transform(imputed_data)
                
        # Encode labels
        if df[label_column].dtype == 'object':
            df[label_column] = le.fit_transform(df[label_column].astype(str))
            
        labels = torch.tensor(df[label_column].values, dtype=torch.long)
        x = torch.tensor(df[node_features].values, dtype=torch.float)

        return x, labels

    def nanCheck(self, node_data, feature_index):
        # Check for NaN values in features
        if torch.isnan(feature_index).any():
            nan_columns = node_data.columns[node_data.isnull().any()].tolist()
            raise ValueError(
                f"NaN values detected in columns: {', '.join(nan_columns)}")
        return "No NaN values detected."

    def construct_graph_data(self, num_rows, edge_index, x, y):
        train_mask = torch.zeros(num_rows, dtype=torch.bool)
        val_mask = torch.zeros(num_rows, dtype=torch.bool)
        test_mask = torch.zeros(num_rows, dtype=torch.bool)
        
        indices = np.random.permutation(num_rows)
        train_size = int(0.7 * num_rows)
        val_size = int(0.2 * num_rows)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        graph_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        return graph_data