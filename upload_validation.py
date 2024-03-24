import os
from flask import flash, session
import pandas as pd
import networkx as nx
from werkzeug.utils import secure_filename

def adjacency_to_edgelist(adj_matrix_df):
    edges = []
    for i, row in adj_matrix_df.iterrows():
        for j, weight in row.items(): 
            if weight != 0 and i != j: 
                edges.append((i, j, weight))
    return pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])

def load_graph_data(filepath, file_extension):
    try:
        if file_extension.lower() == '.csv':
            df = pd.read_csv(filepath)
        elif file_extension.lower() == '.xlsx':
            adj_matrix_df = pd.read_excel(filepath, index_col=0)
            df = adjacency_to_edgelist(adj_matrix_df)
        else:
            raise ValueError("Unsupported file type.")

        if not {'Source', 'Target', 'Weight'}.issubset(df.columns):
            raise ValueError(
                "Dataframe must contain 'Source', 'Target', and 'Weight' columns.")

        # Sort by 'Weight' in descending order and select top 5000 rows
        # df = df.sort_values(by='Weight', ascending=False).head(3000)
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

        return df, G
    except Exception as e:
        flash(str(e)) 
        return None, None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

def process_and_validate_files(node_file, edge_file, raw_data_folder):
    infer_required = False
    if node_file and allowed_file(node_file.filename):
        filename = secure_filename(node_file.filename)
        filepath = os.path.join(raw_data_folder, filename)
        node_file.save(filepath)
        session['node_filepath'] = filepath
    else:
        flash('No participant file detected or unsupported file type.', 'error')
        return False, infer_required

    if edge_file and allowed_file(edge_file.filename):
        filename = secure_filename(edge_file.filename)
        filepath = os.path.join(raw_data_folder, filename)
        edge_file.save(filepath)
        session['edge_filepath'] = filepath
    elif not edge_file:
        infer_required = True
        flash('No edge file detected. Proceeding with inferred graph.', 'warning')
    else:
        flash('Unsupported file type for relationship file.', 'error')
        return False, infer_required
    
    return True, infer_required

