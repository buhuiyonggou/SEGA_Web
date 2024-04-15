# Social Network Explorer
Welcome to **Social Network Explorer**. Our platform aims at exploring the intricate web of relationships between members of any organization, leveraging advanced machine learning and graph theory techniques to provide deep insights into network dynamics.

## About Social Network Explorer
**Social Network Explorer** serves as a unique tool for delving into organizational networks, making it an indispensable resource for researchers and practitioners alike. By combining transductive machine learning algorithms, such as **node2vec**, with Graph Neural Networks (GNN), such as **GraphSAGE**, our application excels in embedding graphs to accurately weigh and analyze network relationships.

### Key Features:
- **Data Upload:** Users can easily upload their dataset, enabling the application to process and analyze complex networks.
- **Graph Embedding:** Utilizing state-of-the-art algorithms, SEGA_Web embeds graphs to identify and weigh relationships within the network.
- **Downloadable Results:** Obtain results with plausible analysis, ready to be used with various libraries for generating detailed network graphs.
- **Advanced Analysis:** The app suggests implicit relationships within the network, such as centrality and clustering, based on specific attributes.
- **Flexible Input and Visualization:** SEGA_Web supports flexible input limitations and offers a variety of visualization options to suit different analytical needs.

## Contributions of Social Network Explorer

SEGA_Web's contributions are threefold, offering significant advancements in the analysis and visualization of organizational networks:

1. **Embedding Original Network Graphs:** Our app provides deep insights into the structure and dynamics of networks by embedding and analyzing original network graphs.
2. **Analysis of Network Relationships:** It suggests implicit relationships, such as centrality and clustering, based on the attributes within the network, offering a new perspective on network analysis.
3. **Flexible and Diverse Visualization:** With flexible input options and a wide range of visualization tools, SEGA_Web caters to diverse analytical needs, enabling users to explore and interpret their networks in various ways.

## Demo
https://drive.google.com/file/d/1M1v4DAhNFBSsHt2Huavpymvo63mEydKN/view

## How to Run

Follow these steps to get started with SEGA_Web:

1. We accept both CSV and XLSX files for parsing and embedding data.
### update Mar.26
2. The application now provide the maximum flexiblity, allowing client choose any column as label and graph infer indicator(if graph is not provided).
3. Please select meaningful combo of label and indicator, for example, using department to infer clusters in workplace makes sense, this column is highly relevant to managers. However, gender and positions are normally irrelavent(hope so).
4. We design explicit accuracy test and plot for validation.
5. To optimize model performance, you can provide your own network graph with columns `source`, `target`, and `weight`.

### Installation

Install the required libraries by running the following command:

```bash
pip install pandas openpyxl pyvis pyecharts torch torch_geometric node2vec






