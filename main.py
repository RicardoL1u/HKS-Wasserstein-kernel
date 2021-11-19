import argparse
import os
import wass_dis
import utilities
import numpy as np
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Provide the dataset name (MUTAG or Enzymes)',
                            choices=['MUTAG', 'ENZYMES'])
    parser.add_argument('--crossvalidation', default=False, action='store_true', help='Enable a 10-fold crossvalidation')
    parser.add_argument('--gridsearch', default=False, action='store_true', help='Enable grid search')
    parser.add_argument('--sinkhorn', default=False, action='store_true', help='Use sinkhorn approximation')
    parser.add_argument('--h', type = int, required=False, default=2, help = "(Max) number of WL iterations")

    args = parser.parse_args()
    dataset = args.dataset
    
    data_path = os.path.join('../data',dataset)
    output_path = os.path.join('output', dataset)
    results_path = os.path.join('results', dataset)
    #---------------------------------
    # Embeddings
    #---------------------------------
    # Load the data and generate the embeddings 
    embedding_type = 'continuous' if dataset == 'ENZYMES' else 'discrete'
    print(f'Generating {embedding_type} embeddings for {dataset}.')
    # if dataset == 'ENZYMES':
    #     label_sequences = compute_wl_embeddings_continuous(data_path, h)
    # else:
    #     label_sequences = compute_wl_embeddings_discrete(data_path, h)
    graph_filenames = utilities.retrieve_graph_filenames(data_path)
    graphs = [ig.read(filename) for filename in graph_filenames]
    wasserstein_distances = wass_dis.pairwise_wasserstein_distance(graphs)


    sinkhorn = False
    # Save Wasserstein distance matrices
    for i, D_w in enumerate(wasserstein_distances):
        filext = 'wasserstein_distance_matrix'
        if sinkhorn:
            filext += '_sinkhorn'
        filext += f'_it{i}.npy'
        np.save(os.path.join(output_path,filext), D_w)
    print('Wasserstein distances computation done. Saved to file.')
    print()



if __name__ == "__main__":
    main()