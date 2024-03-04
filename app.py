import os
import pandas as pd
import networkx as nx
from flask import Flask, session, render_template, request, redirect, flash, jsonify, redirect, url_for, send_file, current_app
from werkzeug.utils import secure_filename
from algorithms import calculate_centrality, detect_communities
from graph_utils import draw_graph_with_pyvis, draw_shortest_path_graph, invert_weights
from pyecharts import options as opts
from pyecharts.charts import Tree
from pyecharts.globals import ThemeType
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
import json
from DataProcessor import GraphSAGEProcessor

app = Flask(__name__)

# Configuration for the file upload folder and allowed file types
UPLOAD_FOLDER = 'uploads'
RAW_DATA_FOLDER = 'raw_data'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RAW_DATA_FOLDER'] = RAW_DATA_FOLDER
app.secret_key = 'BabaYaga'


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
# Convert adjacency matrix to edge list
def adjacency_to_edgelist(adj_matrix_df):
    edges = []
    for i, row in adj_matrix_df.iterrows():
        for j, weight in row.items():  # Changed from iteritems() to items()
            if weight != 0 and i != j:  # Assuming no self-loops and non-zero weight
                edges.append((i, j, weight))
    return pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])


# Load graph data from a file
def load_graph_data(filepath, file_extension):
    try:
        if file_extension.lower() == '.csv':
            df = pd.read_csv(filepath)
        elif file_extension.lower() == '.xlsx':
            adj_matrix_df = pd.read_excel(filepath, index_col=0)
            df = adjacency_to_edgelist(adj_matrix_df)
        else:
            raise ValueError("Unsupported file type.")

        # Check if required columns are present
        if not {'Source', 'Target', 'Weight'}.issubset(df.columns):
            raise ValueError("Dataframe must contain 'Source', 'Target', and 'Weight' columns.")

        # Sort by 'Weight' in descending order and select top 5000 rows
        # df = df.sort_values(by='Weight', ascending=False).head(3000)

        # Create graph from dataframe
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

        return df, G
    except Exception as e:
        # Handle any errors that occur during data loading
        flash(str(e))  # Display the error message to the user
        return None, None  # Return None values to indicate failure

def get_data_with_weight(processor, feature_index, edge_index, index_to_name_mapping):
    
    scaled_weights = processor.model_training(feature_index, edge_index)
    
    # if scaled_weights.size > 0:
    edges_with_weights = processor.data_reshape(scaled_weights, edge_index, index_to_name_mapping)
    
    # Save the DataFrame to a CSV file
    try: 
        output_path = UPLOAD_FOLDER + '/weighted_graph.csv'
        edges_with_weights.to_csv(output_path, index=False)
        message = "Process Status: Congragulations! You data has successfully processed."
    except Exception as message:
        flash(f'Error: {str(message)}')
    finally:
        flash(message) 
        
@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/user_data_adoper', methods=['GET'])
def upload_user_data():
    eb_algorithm = request.args.get('eb_algorithm', 'graphSAGE') 
    return render_template('UserDataAdopter.html', eb_algorithm=eb_algorithm)

@app.route('/user_upload', methods=['GET', 'POST'])
def upload_data_store():
    if request.method == 'POST':
        node_file = request.files.get('participantFile')
        if node_file and allowed_file(node_file.filename):
            node_filename = secure_filename(node_file.filename)
            
        edge_file = request.files.get('relationshipFile')
        if edge_file and allowed_file(edge_file.filename):
            edge_filename = secure_filename(edge_file.filename)
            
        # Save files and process
        if node_file:
            session['node_filepath'] = os.path.join(app.config['RAW_DATA_FOLDER'], secure_filename(node_file.filename))
            node_file.save(session['node_filepath'])
            session['upload_success'] = True
        if edge_file:
            session['edge_filepath'] = os.path.join(app.config['RAW_DATA_FOLDER'], secure_filename(edge_file.filename))
            edge_file.save(session['edge_filepath'])
        session['process_success'] = False
        
        return redirect(url_for('confirm_edge_upload'))

@app.route('/confirm_edge_upload')
def confirm_edge_upload():
    if 'node_filepath' not in session:
        flash('No participant file detected. Please upload the required files.', 'error')
        return redirect(url_for('upload_user_data'))
    elif 'node_filepath' in session and 'edge_filepath' not in session:
        flash('No edge file detected. Proceeding with inferred graph. You can upload an edge file to improve model accuracy.', 'warning')
    else:
        flash('Files successfully uploaded.', 'success')
    return render_template('dataProcess.html', edge_file_provided='edge_filepath' in session)

@app.route('/process_graphsage')
def data_process():
    try:
        node_filepath = session.get('node_filepath')
        edge_filepath = session.get('edge_filepath')

        if node_filepath:
            flash("Upload Status: upload successful!")
        else:
            flash("Upload Status: Sorry, there is something wrong with uploading...")

        processor = GraphSAGEProcessor(node_filepath, edge_filepath if edge_filepath else None)
            
        hr_data = processor.fetch_data_from_user(node_filepath)
            
        if hr_data.empty:
            flash("Sorry, document data cannot be found.")
        
        # process features
        hr_data.columns = processor.rename_columns_to_standard(hr_data, processor.COLUMN_ALIGNMENT)
        
        # store index map
        index_to_name_mapping = processor.create_index_id_name_mapping(hr_data)
        
        # align name of columns 
        # target embedding attributes for this instance, used for creating node_index
        columns_to_exclude = ['id', 'name']
        node_features = [col for col in hr_data.columns if col not in columns_to_exclude]
        
        # get features with number
        features = processor.features_generator(hr_data, node_features)
        
        feature_index = processor.feature_index_generator(features)
        
        # process edges 
        if edge_filepath:
            edges = processor.edges_generator(hr_data, edge_filepath)
        else:
            edges = processor.edges_generator(hr_data)
            
        edge_index = processor.edge_index_generator(edges)
        # check if nan value exists
        processor.nanCheck(hr_data,feature_index)
        
        get_data_with_weight(processor, feature_index, edge_index, index_to_name_mapping)
        session['process_success'] = True
    except Exception as e:
        session['process_success'] = False
        flash(f'Error: {str(e)}')
    finally:
        # Clear the session after processing is complete
        session.pop('node_filepath', None)
        session.pop('edge_filepath', None)
    return render_template('dataProcess.html', process_success=session.get('process_success', False))

# adding later
@app.route('/process_node2vec')
def data_process_transductive():
    return render_template('Home.html')

@app.route('/training_progress')
def training_progress():
    progress = session.get('training_progress', 'Not started')
    return jsonify({'progress': progress})

@app.route('/download_processed_file')
def download_processed_file():
    # Construct an absolute path to the file
    file_path = os.path.join(current_app.root_path, 'uploads', 'weighted_graph.csv')
    try:
        # Attempt to send the file
        return send_file(file_path, as_attachment=True)
    except FileNotFoundError:
        # Handle the error if the file does not exist
        return "File not found.", 404

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

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
        return redirect(url_for('upload_file'))  # Redirect the user to upload page

    # remove_low_degree_nodes(G)

    centrality = calculate_centrality(G, centrality_algo)
    communities = detect_communities(G, community_algo)
    community_map = {node: i for i, community in enumerate(communities) for node in community}

    graph_html_path = draw_graph_with_pyvis(G, centrality, community_map)

    return render_template('index.html', graph_html_path=graph_html_path, filename=filename, community_algo=community_algo, centrality_algo=centrality_algo)


@app.route('/show_top_communities/<filename>', methods=['GET', 'POST'])
def show_top_communities(filename):
    """Show top communities based on the selected algorithms."""
    if request.method == 'POST':
        top_n = int(request.form.get('topN', 10))
        centrality_algo = request.form.get('centrality', 'pagerank')  # Use request.form for POST data
        community_algo = request.form.get('community', 'louvain')  # Use request.form for POST data
    else:
        top_n = 10
        centrality_algo = 'pagerank'
        community_algo = 'louvain'

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    _, file_extension = os.path.splitext(filename)
    _, G = load_graph_data(filepath, file_extension)

    if G is None:  # Check if G is None, indicating an error occurred
        # flash('File format error. Please upload a valid file.', 'danger')
        return redirect(url_for('upload_file'))  # Redirect the user to upload page

    # remove_low_degree_nodes(G)

    centrality = calculate_centrality(G, centrality_algo)
    communities = detect_communities(G, community_algo)
    community_map = {node: i for i, community in enumerate(communities) for node in community}

    community_scores = {i: sum(centrality[node] for node in com) for i, com in enumerate(communities)}
    top_communities = sorted(community_scores, key=community_scores.get, reverse=True)[:top_n]

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

    path = nx.shortest_path(H, source=nodeStart, target=nodeEnd, weight='weight')
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

    # Compute the distance matrix directly from graph G
    distance_matrix = nx.floyd_warshall_numpy(G, weight='weight')
    # Convert the numpy array returned by floyd_warshall_numpy to a format suitable for the linkage function
    Z = linkage(squareform(distance_matrix, checks=False), method='complete')

    # Convert the linkage matrix Z into a JSON tree structure
    dendrogram_json = convert_to_dendrogram_json(Z, list(G.nodes()))

    # Save the dendrogram_json to a file
    dendrogram_json_path = os.path.join('static', 'dendrogram.json')
    with open(dendrogram_json_path, 'w') as f:
        json.dump(dendrogram_json, f)

    # Use Pyecharts to generate a dendrogram
    tree_chart = (
        Tree(init_opts=opts.InitOpts(width="1200px", height="900px", theme=ThemeType.LIGHT))
        .add("", [dendrogram_json],
             collapse_interval=10,
             initial_tree_depth=10,
             is_roam=True,
             symbol="circle",
             symbol_size=8,  # Adjust the size of the nodes
             label_opts=opts.LabelOpts(
                 font_size=10,
                 color="#fa8072",  # Darker color for labels for better readability
                 font_style="normal",
                 font_weight="bold",
                 position="right"  # Adjust label position if needed
             ),
             # leaves_label_opts=opts.LabelOpts(
             #     color="#fff",  # Light color for leaf labels if needed
             #     position="right",
             #     horizontal_align="right",
             #     vertical_align="middle",
             #     rotate=-90
             # ),
             )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Dendrogram",
                subtitle="Hierarchical Clustering",
                title_textstyle_opts=opts.TextStyleOpts(color="black"),
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}")  # Customizing tooltip
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

    # Build and return the JSON structure
    return build_json(tree)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(RAW_DATA_FOLDER):
        os.makedirs(RAW_DATA_FOLDER)
        
    app.run(debug=True) 
