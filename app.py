import os
import pandas as pd
import networkx as nx
import numpy as np
from flask import Flask, request, session, render_template, request, redirect, flash, jsonify, redirect, url_for, send_file, current_app
import torch
from werkzeug.utils import secure_filename
from algorithms import calculate_centrality, detect_communities
from upload_validation import column_validation, load_graph_data, process_and_validate_files
from graph_utils import draw_graph_with_pyvis, draw_shortest_path_graph, invert_weights
from pyecharts import options as opts
from pyecharts.charts import Tree
from pyecharts.globals import ThemeType
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, to_tree
import json
from DataProcessor import DataProcessor
from graphSAGE import GraphSAGE
import logging
from node2vec import Node2Vec


app = Flask(__name__)

# Configuration for the file upload folder and allowed file types
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RAW_DATA_FOLDER = os.path.join(BASE_DIR, 'raw_data')
PLOT_FOLDER = os.path.join(BASE_DIR, 'plots')
PROCESSED_GRAPH_FOLDER = os.path.join(BASE_DIR, 'processed_graph')
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
EPOCHES = 300
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RAW_DATA_FOLDER'] = RAW_DATA_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER
app.config['PROCESSED_GRAPH_FOLDER'] = PROCESSED_GRAPH_FOLDER
app.secret_key = 'BabaYaga'
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    session.clear()
    return render_template('Home.html')


@app.route('/user_data_adoper', methods=['GET'])
def upload_user_data():
    eb_algorithm = request.args.get('eb_algorithm', 'graphSAGE')
    return render_template('UserDataAdopter.html', eb_algorithm=eb_algorithm)


@app.route('/user_upload', methods=['GET', 'POST'])
def upload_data_store():
    if request.method == 'POST':
        node_file = request.files.get('participantFile')
        edge_file = request.files.get('relationshipFile')
        success, infer_required = process_and_validate_files(
            node_file, edge_file, app.config['RAW_DATA_FOLDER'])

        if not success:
            return redirect(url_for('upload_user_data'))

        # Column validation
        node_filepath = session.get('node_filepath')
        edge_filepath = session.get('edge_filepath', None)

        # def clear_flashed_messages(category_filter=['error']):
        #     session.pop('_flashes', None)

        column_validation_success, message = column_validation(
            node_filepath, edge_filepath)

        if not column_validation_success:
            # clear_flashed_messages()
            flash(f"Error: {message}", 'error')
            return redirect(url_for('upload_user_data'))

        return redirect(url_for('graph_infer', infer='True' if infer_required else 'False'))
    return redirect(url_for('upload_user_data'))


@app.route('/graph_infer')
def graph_infer():
    infer_required = request.args.get(
        'infer', default='False', type=str) == 'True'
    node_filepath = session.get('node_filepath')

    if not node_filepath:
        return redirect(url_for('upload_user_data'))

    if node_filepath.endswith('.csv'):
        node_df = pd.read_csv(node_filepath, encoding='utf-8')
    elif node_filepath.endswith('.xlsx'):
        node_df = pd.read_excel(node_filepath)

    columns = node_df.columns.tolist()
    # Exclude 'id' and 'name' from the columns
    selectable_columns = [col for col in columns if col.lower() not in [
        'id', 'name']]

    return render_template('inferSelector.html', infer_required=infer_required, columns=selectable_columns)


@app.route('/perform_inference', methods=['POST'])
def perform_inference():
    session['selected_label_column'] = request.form.get('selectedLabelColumn')
    session['selected_edge_infer_column'] = request.form.get(
        'selectedEdgeInferColumn', None)
    session['infer_required'] = 'True' if session['selected_edge_infer_column'] is None else 'False'

    return redirect(url_for('data_process_panel'))


@app.route('/data_panel')
def data_process_panel():
    # initial data
    graph_data_file = None
    graph_data = None
    unique_label = None
    edge_infer_column = None
    num_features = 0
    num_infers = 0
    num_labels = 0

    label_column = session.get('selected_label_column').lower()
    if session.get('selected_edge_infer_column'):
        edge_infer_column = session.get('selected_edge_infer_column').lower()

    try:
        node_filepath = session.get('node_filepath')
        edge_filepath = session.get('edge_filepath')

        if node_filepath:
            if edge_filepath:
                flash("Upload Status: upload successful!")
            else:
                flash("Upload Status: node file upload successful!")
        else:
            logging.error(
                "Upload Status: Sorry, there is something wrong with uploading...")

        processor = DataProcessor(
            node_filepath, edge_filepath if edge_filepath else None)

        nodes_data = processor.fetch_data_from_user(node_filepath)

        if nodes_data.empty:
            logging.error("Sorry, document data cannot be found.")
            return redirect(url_for('upload_page'))

        # process features
        nodes_data = processor.rename_columns_to_standard_graphSAGE(
            nodes_data, processor.COLUMN_ALIGNMENT)
        # store index map
        index_to_name_mapping = processor.create_str_index_mapping(
            nodes_data, label_column, edge_infer_column)

        # save index map to processed graph
        try:
            mapping_path = os.path.join(
                PROCESSED_GRAPH_FOLDER, 'edge_mapping.csv')
            index_to_name_mapping.to_csv(mapping_path, index=False)
            session['edge_mapping_file'] = mapping_path
        except Exception as message:
            logging.error(f'Error: output {str(message)}')
            flash(f'Error: output {str(message)}')

        if edge_infer_column:
            unique_infer_categories = nodes_data[edge_infer_column].unique(
            ).tolist()
        else:
            unique_infer_categories = None

        unique_label = nodes_data[label_column].unique().tolist()

        # prepare num_features and num_classes for visualization of validation
        if label_column == edge_infer_column or edge_infer_column is None:
            columns_to_exclude = ['id', 'name', edge_infer_column]
        else:
            columns_to_exclude = ['id', 'name',
                                  label_column, edge_infer_column]
        node_features = [
            col for col in nodes_data.columns if col not in columns_to_exclude]

        # column numbers and category of infer indicator
        num_features, num_labels = len(node_features), len(unique_label)

        if edge_infer_column:
            num_infers = len(unique_infer_categories)

        # generate edges
        edges_data = processor.edges_generator(
            nodes_data, edge_infer_column, edge_filepath)

        edge_index = processor.edge_index_generator(edges_data)

        # generate features
        x, labels = processor.numeric_dataset(
            nodes_data, node_features, label_column)

        # check if nan value exists
        flash(processor.nanCheck(nodes_data, x))

        # create train and test mask
        num_rows = nodes_data.shape[0]

        graph_data = processor.construct_graph_data(
            num_rows, edge_index, x, labels)

        # save graph data
        graph_data_file = PROCESSED_GRAPH_FOLDER + '/graph_data.pt'
        torch.save(graph_data, graph_data_file)
        session['graph_data_file'] = graph_data_file
        session['num_features'] = num_features
        session['num_labels'] = num_labels
        session["num_infers"] = num_infers
        session['label_names'] = unique_label
        session['enable_process'] = False
        session['enable_download'] = False
        session['enable_analyze'] = False
    except Exception as e:
        flash(f'Error: {str(e)}')
    finally:
        # Clear the session after processing is complete
        session.pop('node_filepath', None)
        session.pop('edge_filepath', None)

    return render_template('dataProcess.html',
                           graph_file=graph_data_file,
                           mapping_file=mapping_path,
                           num_features=num_features,
                           num_labels=num_labels,
                           num_infers=num_infers,
                           label_names=unique_label,
                           process_success=session.get('enable_process'),
                           enable_download=session.get('enable_download'),
                           enable_analyze=session.get('enable_analyze'))


# Implementation of graphSAGE
@app.route('/process_graphsage', methods=['GET', 'POST'])
def process_with_graphsage():

    graph_data_file = session.get('graph_data_file')
    mapping_file = session.get('edge_mapping_file')
    num_features = session.get('num_features')
    num_labels = session.get('num_labels')
    num_infers = session.get('num_infers')
    labels = session.get('label_names')

    if not graph_data_file or num_features is None or num_labels is None:
        logging.error('Missing data for processing.')
        return redirect(url_for('data_process_panel'))
    if not mapping_file:
        logging.error('Missing mapping file for processing.')
        return redirect(url_for('data_process_panel'))

    try:
        graph_data = torch.load(graph_data_file)
        flash('GraphSAGE processing completed successfully', 'success')
        mapping_df = pd.read_csv(mapping_file)

        session.pop('graph_data_file', None)
        session.pop('edge_mapping_file', None)
        session.pop('num_features', None)
        session.pop('num_labels', None)
        session.pop('num_infers', None)
        session.pop('label_names', None)

        graphSAGEProcessor = GraphSAGE(
            in_channels=num_features, hidden_channels=16, out_channels=num_labels)

        embeddings = graphSAGEProcessor.model_training(
            graphSAGEProcessor, graph_data, EPOCHES)
        if embeddings is None:
            logging.error('Error in generating embeddings.')
            return redirect(url_for('data_process_panel'))

        # generating validation plot
        graphSAGEProcessor.visualize_embeddings(
            embeddings, graph_data.y, labels, PLOT_FOLDER)

        # mapping weighted graph and save to csv
        edge_embeddings_start = embeddings[graph_data.edge_index[0]]
        edge_embeddings_end = embeddings[graph_data.edge_index[1]]

        raw_weights = torch.norm(
            edge_embeddings_start - edge_embeddings_end, dim=1).cpu().numpy()
        edges_with_weights = pd.DataFrame(
            graph_data.edge_index.t().cpu().numpy(), columns=['Source', 'Target'])
        edges_with_weights['Weight'] = raw_weights

        index_to_name_dict = mapping_df.set_index('index')['name'].to_dict()

        edges_with_weights['Source'] = edges_with_weights['Source'].map(
            index_to_name_dict)
        edges_with_weights['Target'] = edges_with_weights['Target'].map(
            index_to_name_dict)

        # Save the DataFrame to a CSV file
        try:
            output_path = UPLOAD_FOLDER + '/graphSAGE_edges.csv'
            edges_with_weights.to_csv(output_path, index=False)
            message = "Process Status: Congragulations! You data has successfully processed."
        except Exception as message:
            flash(f'Error: output {str(message)}')
        finally:
            flash(message)
            session['enable_process'] = True
            session['enable_download'] = True
            session['enable_analyze'] = True
            session['processed_file'] = 'graphSAGE_edges.csv'
    except Exception as e:
        session['enable_process'] = False
        session['enable_download'] = False
        session['enable_analyze'] = False
        flash(f'Error: {str(e)}')
        # return redirect(url_for('graph_infer'))
    finally:
        # Clear the session after processing is complete
        session.pop('node_filepath', None)
        session.pop('edge_filepath', None)
        os.remove(mapping_file)
        os.remove(graph_data_file)
    return render_template('dataProcess.html', process_success=session.get('enable_process'),
                           download_success=session.get('enable_download'),
                           analyze_success=session.get('enable_analyze'))


@app.route('/process_node2vec', methods=['GET', 'POST'])
def data_process_node2vec():
    graph_data_file = session.get('graph_data_file')
    mapping_file = session.get('edge_mapping_file')
    num_features = session.get('num_features')
    num_labels = session.get('num_labels')
    labels = session.get('label_names')

    if not graph_data_file or num_features is None or num_labels is None:
        logging.error('Missing data for processing.')
        return redirect(url_for('data_process_panel'))
    if not mapping_file:
        logging.error('Missing mapping file for processing.')
        return redirect(url_for('data_process_panel'))

    try:
        graph_data = torch.load(graph_data_file)
        mapping_df = pd.read_csv(mapping_file)

        session.pop('graph_data_file', None)
        session.pop('edge_mapping_file', None)
        session.pop('num_features', None)
        session.pop('num_labels', None)
        session.pop('label_names', None)

        name_dict = mapping_df.set_index('index')['name'].to_dict()
        edges = graph_data.edge_index.t().cpu().numpy()

        G = nx.Graph()
        G.add_edges_from(edges)

        # Process with Node2Vec
        node2vec = Node2Vec(G, dimensions=64, walk_length=30,
                            num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = model.wv

        # Save Node2Vec embeddings to a DataFrame
        edge_list = []
        for node in G.nodes():
            if node in embeddings:
                for neighbor in G.neighbors(node):
                    if neighbor in embeddings:
                        weight = np.dot(embeddings[node], embeddings[neighbor])
                        source_name = name_dict.get(node, f"Unknown-{node}")
                        target_name = name_dict.get(
                            neighbor, f"Unknown-{neighbor}")
                        edge_list.append((source_name, target_name, weight))

        edges_df = pd.DataFrame(
            edge_list, columns=['Source', 'Target', 'Weight'])

        output_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'node2vec_edges.csv')
        edges_df.to_csv(output_path, index=False)

        flash("Node2Vec process completed and CSV file generated.", "success")
        session['enable_process'] = True
        session['enable_download'] = True
        session['enable_analyze'] = True
        session['processed_file'] = 'node2vec_edges.csv'

    except Exception as e:
        logging.exception("Error in Node2Vec processing: " + str(e))
        session['enable_process'] = False
        session['enable_download'] = False
        session['enable_analyze'] = False
        flash(f"Error in Node2Vec processing: {str(e)}", "error")
    finally:
        session.pop('node_filepath', None)
        session.pop('edge_filepath', None)
        os.remove(mapping_file)
        os.remove(graph_data_file)

    return render_template('dataProcess.html', process_success=session.get('enable_process'),
                           download_success=session.get('enable_download'),
                           analyze_success=session.get('enable_analyze'))

@app.route('/download_processed_file')
def download_processed_file():
    processed_file = session.get('processed_file')
    if processed_file:
        file_path = os.path.join(
            current_app.root_path, app.config['UPLOAD_FOLDER'], processed_file)
        return send_file(file_path, as_attachment=True)
    else:
        flash('No processed file available for download.')
        return redirect(url_for('data_process'))


@app.route('/analyze')
def analyze():
    filename = session.get('processed_file', None)
    return render_template('analyze.html', filename=filename)


@app.route('/upload_to_vis', methods=['GET', 'POST'])
def upload_file():
    """Handle file uploads."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            return render_template('analyze.html', filename=filename)
    return render_template('upload.html')


@app.route('/show_graph/<filename>')
def network_graph(filename):
    """Display the network graph based on selected algorithms."""
    centrality_algo = request.args.get('centrality', 'pagerank')
    community_algo = request.args.get('community', 'louvain')

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)

    _, G = load_graph_data(filepath, file_extension)

    if G is None:  # Check if G is None, indicating an error occurred
        # flash('File format error. Please upload a valid file.', 'danger')
        # Redirect the user to upload page
        return redirect(url_for('upload_file'))

    # remove_low_degree_nodes(G)

    centrality = calculate_centrality(G, centrality_algo)
    communities = detect_communities(G, community_algo)
    community_map = {node: i for i, community in enumerate(
        communities) for node in community}

    graph_html_path = draw_graph_with_pyvis(G, centrality, community_map)

    return render_template('index.html', graph_html_path=graph_html_path, filename=filename, community_algo=community_algo, centrality_algo=centrality_algo)


@app.route('/show_top_communities/<filename>', methods=['GET', 'POST'])
def show_top_communities(filename):
    """Show top communities based on the selected algorithms."""
    if request.method == 'POST':
        top_n = int(request.form.get('topN', 10))
        # Use request.form for POST data
        centrality_algo = request.form.get('centrality', 'pagerank')
        # Use request.form for POST data
        community_algo = request.form.get('community', 'louvain')
    else:
        top_n = 10
        centrality_algo = 'pagerank'
        community_algo = 'louvain'

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)
    _, G = load_graph_data(filepath, file_extension)

    if G is None:
        return redirect(url_for('upload_file'))

    # remove_low_degree_nodes(G)

    centrality = calculate_centrality(G, centrality_algo)
    communities = detect_communities(G, community_algo)
    community_map = {node: i for i, community in enumerate(
        communities) for node in community}

    community_scores = {
        i: sum(centrality[node] for node in com) for i, com in enumerate(communities)}
    top_communities = sorted(
        community_scores, key=community_scores.get, reverse=True)[:top_n]

    top_nodes = set().union(*(communities[i] for i in top_communities))
    H = G.subgraph(top_nodes)

    graph_html_path = draw_graph_with_pyvis(H, centrality, community_map)

    return render_template('index.html', graph_html_path=graph_html_path, filename=filename, community_algo=community_algo, centrality_algo=centrality_algo)


@app.route('/find_shortest_path', methods=['POST'])
def find_shortest_path():
    """Find and display the shortest path between two nodes."""
    data = request.get_json()
    filename = data['filename']
    nodeStart = data['nodeStart']
    nodeEnd = data['nodeEnd']

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)
    _, G = load_graph_data(filepath, file_extension)

    # Check if the specified nodes exist in the graph
    if nodeStart not in G or nodeEnd not in G:
        return jsonify({'error': 'One or both of the specified nodes do not exist in the graph.'})

    # Invert weights to reflect closeness instead of distance
    H = invert_weights(G)

    path = nx.shortest_path(H, source=nodeStart,
                            target=nodeEnd, weight='weight')
    unique_filename = draw_shortest_path_graph(H, path)

    return jsonify({'graph_html_path': f'/static/{unique_filename}'})


@app.route('/show_dendrogram/<filename>')
def show_dendrogram(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    _, file_extension = os.path.splitext(filename)

    # Load the graph data from the file
    df, G = load_graph_data(file_path, file_extension)
    # If the graph is not loaded successfully, redirect to upload page
    if G is None:
        flash('Error loading graph data.', 'danger')
        return redirect(url_for('upload_file'))

    def inverse_weight(u, v, d):
        weight = d.get('weight', 1.0)
        epsilon = 1e-4
        if weight > epsilon:
            return 1.0 / weight
        else:
            return 1e4

    # Compute the distance matrix directly from graph G
    # distance_matrix = nx.floyd_warshall_numpy(G, weight='weight')
    distance_matrix = nx.floyd_warshall_numpy(G, weight=inverse_weight)
    distance_matrix[np.isinf(distance_matrix)] = 1e4
    # Convert the numpy array returned by floyd_warshall_numpy to a format suitable for the linkage function
    Z = linkage(squareform(distance_matrix, checks=False), method='complete')

    dendrogram_json = convert_to_dendrogram_json(Z, list(G.nodes()))

    dendrogram_json_path = os.path.join('static', 'dendrogram.json')
    with open(dendrogram_json_path, 'w') as f:
        json.dump(dendrogram_json, f)

    tree_chart = (
        Tree(init_opts=opts.InitOpts(width="1200px",
             height="900px", theme=ThemeType.LIGHT))
        .add("", [dendrogram_json],
             collapse_interval=10,
             initial_tree_depth=10,
             is_roam=True,
             symbol="circle",
             symbol_size=8, 
             label_opts=opts.LabelOpts(
                 font_size=10,
                 color="#fa8072",
                 font_style="normal",
                 font_weight="bold",
                 position="right" 
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Dendrogram",
                subtitle="Hierarchical Clustering",
                title_textstyle_opts=opts.TextStyleOpts(color="black"),
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{b}")
        )
    )

    # Save the dendrogram as an HTML file
    dendrogram_html_filename = 'dendrogram_chart.html'
    tree_chart.render(path=os.path.join('static', dendrogram_html_filename))

    # Redirect to the page displaying the dendrogram
    return render_template('dendrogram.html', dendrogram_html_filename=dendrogram_html_filename, filename=filename)


def convert_to_dendrogram_json(Z, labels):
    # Convert the linkage matrix into a tree structure.
    tree = to_tree(Z, rd=False)

    def count_leaves(node):
        # Recursively count the leaves under a node
        if node.is_leaf():
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    # Recursive function to build the JSON structure
    def build_json(node):
        if node.is_leaf():
            # For leaf nodes, use the provided labels
            return {"name": labels[node.id]}
        else:
            # For internal nodes, generate a name that includes the cluster size
            size = count_leaves(node)
            name = f"Cluster of {size}"
            # Recursively build the JSON for children
            return {
                "name": name,
                "children": [build_json(node.left), build_json(node.right)]
            }

    return build_json(tree)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(RAW_DATA_FOLDER):
        os.makedirs(RAW_DATA_FOLDER)
    if not os.path.exists(PROCESSED_GRAPH_FOLDER):
        os.makedirs(PROCESSED_GRAPH_FOLDER)
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)
    
    app.run(debug=True)
