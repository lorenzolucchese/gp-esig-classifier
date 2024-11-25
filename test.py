import numpy as np
import os
import yaml
from lib.data.utils import get_order, add_timestamps
from lib.model import GPES
from lib.utils import evaluate_accuracy, to_tensor
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset


def test(args):
    device = 'cuda:{}'.format(args.device) if args.use_cuda and torch.cuda.is_available() else 'cpu'

    configs = yaml.safe_load(open(os.path.join(args.base_dir, args.configs), 'rb'))

    # dataset
    data_configs = configs.get('data', {})
    dataset = configs['data']['dataset']
    _, _, test_X, test_y = np.load(
        os.path.join(
            args.data_dir, 
            dataset + '.npy'
        ), 
        allow_pickle=True
    )

    # data hyperparameters
    L1 = data_configs.get('L1', 100)
    dim = data_configs.get('dim', 1)
    number_classes = data_configs.get('number_classes', 2)
    begin = data_configs.get('begin', 0)
    end = data_configs.get('end', 1)
    assert test_X.shape[1] == L1
    if dim == 1:
        assert test_X.ndim == 2
    else:
        assert test_X.ndim == 3 and test_X.shape[-1] == dim

    # model params
    model_configs = configs.get('model', {})
    division = model_configs.get('division', 1) 
    alpha = model_configs.get('alpha', 'full') 
    level = model_configs.get('level', 4)
    C = model_configs.get('C', 1e3)
    a = model_configs.get('a', 1)
    K = model_configs.get('K', 10)
    martingale_indices = model_configs.get('martingale_indices', None)

    known_times, new_times, order, extended_order = get_order(L1=L1, dim=dim, division=division, begin=begin, end=end)
    L2 = len(new_times)

    # fit params
    fit_configs = configs.get('fit', {})
    training_seed = fit_configs.get('seed', 0)
    
    # training dir
    parent_training_dir = os.path.join(args.base_dir, dataset, 'm-esig' if martingale_indices else 'esig')
    os.makedirs(parent_training_dir, exist_ok=True)
    training_dir = os.path.join(parent_training_dir, f'run_{training_seed}')
    os.makedirs(training_dir, exist_ok=True)

    # Initialize model, loss function, and optimizer
    model = GPES(
        L1=L1, 
        L2=L2, 
        dim=dim, 
        order=to_tensor(order), 
        extended_order=to_tensor(extended_order), 
        alpha=alpha, 
        level=level, 
        number_classes=number_classes, 
        C=C, 
        a=a, 
        K=K,
        martingale_indices=martingale_indices
    ).to(device)

    # get test data
    test_X = add_timestamps(test_X, known_times=known_times, new_times=new_times)

    test_data = TensorDataset(to_tensor(test_X).to(device), to_tensor(test_y).long().to(device))
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    # Load from best_weiths.pth
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(os.path.join(training_dir, 'best_model.pth'), map_location=torch.device(device)))
    model.eval()

    test_metrics = {}

    # Testing loop
    for test_seed in range(args.test_seeds):
        # set seed
        torch.manual_seed(test_seed)
        # Testing step
        test_running_loss = 0.0
        with torch.no_grad():  # No gradient calculation for testing
            for inputs, labels in test_loader:
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute test loss
                test_running_loss += loss.item()  # Accumulate test loss

        # Average loss and accuracy for the validation epoch
        test_loss = test_running_loss / len(test_loader)
        test_accuracy = evaluate_accuracy(model, test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        test_metrics[test_seed] = {'Test Loss': test_loss, 'Test Accuracy': test_accuracy}

    with open(os.path.join(training_dir, 'test_metrics.pkl'), 'wb') as f:
        pickle.dump(test_metrics, f)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Meta parameters
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-data_dir', default='./datasets', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-device', default=0, type=int)
    parser.add_argument('-configs', type=str)
    parser.add_argument('-test_seeds', default=10, type=int)

    args = parser.parse_args()
    test(args)