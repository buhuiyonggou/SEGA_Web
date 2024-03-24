import pandas as pd
import numpy as np
from itertools import combinations
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

class DataProcessor:
    def __init__(self, nodes_folder, edges_folder=None):
        self.nodes_folder = nodes_folder
        self.edges_folder = edges_folder
        
        self.CONNECT_AMONG_SUB_DEPART = 0.25
        self.CONNECT_AMONG_DEPARTMENT = 0.05
        self.CONNECT_AMONG_ORGANIZATION = 0.01
        self.COLUMN_ALIGNMENT = {
            'id': ['id', 'emp_id', 'employeeid'],
            'name': ['name', 'full_name', "employee_name", 'first_name'],
            'department': ['department', 'depart', 'sector'],
            'sub-depart': ['sub-depart', 'sub-department', 'sub-sector', 'second-department'],
            'manager': ['manager', 'supervisor', 'manager_name', 'managername']
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

    def create_index_id_name_mapping(self, hr_data):

        index_to_id_name_mapping = [{
            'index': i,
            'id': row['id'],
            'name': row['name'].title() if 'name' in row else 'Unknown',
            'department': row['department']
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
                # Convert categorical data to numerical
                df[column] = le.fit_transform(df[column].astype(str))
            else:
                # Reshape the column data to a 2D array for imputer and scaler
                # Reshape data
                column_data_reshaped = df[column].values.reshape(-1, 1)
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

    def edges_generator(self, hr_data, edge_filepath=None):
        edges = []
        mapping_df = self.create_index_id_name_mapping(hr_data)

        if edge_filepath:
            # mapping name to index and generating edges
            hr_edge = self.fetch_data_from_user(edge_filepath)
            for _, row in hr_edge.iterrows():
                source_id = row['source']
                target_id = row['target']
                source_index = mapping_df[mapping_df['id']
                                          == source_id].index.item()
                target_index = mapping_df[mapping_df['id']
                                          == target_id].index.item()

                edges.append([source_index, target_index])
        else:
            # if edges are not given, infer the edges
            if 'sub-depart' in hr_data.columns and hr_data['sub-depart'].notnull().any():
                for dept in hr_data['sub-depart'].unique():
                    dept_indices = hr_data[hr_data['sub-depart']
                                           == dept].index.tolist()
                    self.manage_edge_probability(
                        edges, hr_data, dept_indices, sub_dept_info=True)
            else:
                for dept in hr_data['department'].unique():
                    dept_indices = hr_data[hr_data['department']
                                           == dept].index.tolist()
                    self.manage_edge_probability(
                        edges, hr_data, dept_indices, sub_dept_info=False)
        return edges

    def edge_index_generator(self, edges):
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def features_generator(self, hr_data, node_features):
        hr_data_parsed = self.preprocess_data(hr_data, node_features)
        # Exclude 'id' and 'name' columns from features
        feature_columns = [
            col for col in hr_data.columns if col in node_features]

        features_data = hr_data_parsed[feature_columns].values

        return features_data

    def feature_index_generator(self, features):
        feature_index = torch.tensor(features, dtype=torch.float)

        return feature_index

    def nanCheck(self, hr_data, feature_index):
        # Check for NaN values in features
        if torch.isnan(feature_index).any():
            nan_columns = hr_data.columns[hr_data.isnull().any()].tolist()
            raise ValueError(
                f"NaN values detected in columns: {', '.join(nan_columns)}")
        return "No NaN values detected."

    
