import argparse
import os

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

if __name__ == "__main__":
    main()