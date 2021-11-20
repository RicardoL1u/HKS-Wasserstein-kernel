import os

#################
# File loaders 
#################
def read_labels(filename):
    '''
    Reads labels from a file. Labels are supposed to be stored in each
    line of the file. No further pre-processing will be performed.
    '''
    labels = []
    with open(filename) as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
    return labels

def retrieve_graph_filenames(data_directory):
    # Load graphs
    files = os.listdir(data_directory)
    graphs = [g for g in files if g.endswith('gml')]
    graphs.sort()
    return [os.path.join(data_directory, g) for g in graphs]