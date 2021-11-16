import os

def retrieve_graph_filenames(data_directory):
    # Load graphs
    files = os.listdir(data_directory)
    graphs = [g for g in files if g.endswith('gml')]
    graphs.sort()
    return [os.path.join(data_directory, g) for g in graphs]