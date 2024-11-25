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


def train(args):
    device = 'cuda:{}'.format(args.device) if args.use_cuda and torch.cuda.is_available() else 'cpu'

    # dataset
    dataset = args.dataset
    train_X, train_y, test_X, test_y = np.load(
        os.path.join(
            args.data_dir, 
            dataset + '.npy'
        ), 
        allow_pickle=True
    )
    train_samples = len(train_X)

    with open(args.hyperparameters_config, 'r') as f:
        hyperparameters = yaml.safe_load(f)[dataset]

    # data hyperparameters
    data_hyperparameters = hyperparameters.get('data', {})
    L1 = data_hyperparameters.get('L1', 100)
    dim = data_hyperparameters.get('dim', 1)
    number_classes = data_hyperparameters.get('number_classes', 2)
    begin = data_hyperparameters.get('begin', 0)
    end = data_hyperparameters.get('end', 1)
    assert train_X.shape[1] == L1 and test_X.shape[1] == L1
    if dim == 1:
        assert (train_X.ndim == 2 and test_X.ndim == 2)
    else:
        assert (train_X.ndim == 3 and test_X.ndim == 3) and (train_X.shape[-1] == dim and test_X.shape[-1] == dim)

    # model params
    model_hyperparameters = hyperparameters.get('model', {})
    division = model_hyperparameters.get('division', 1) 
    alpha = model_hyperparameters.get('alpha', 'full') 
    level = model_hyperparameters.get('level', 4)
    C = model_hyperparameters.get('C', 1e3)
    a = model_hyperparameters.get('a', 1)
    K = model_hyperparameters.get('K', 10)
    martingale_indices = args.martingale_indices

    known_times, new_times, order, extended_order = get_order(L1=L1, dim=dim, division=division, begin=begin, end=end)
    L2 = len(new_times)

    # fit params
    fit_hyperparameters = hyperparameters.get('fit', {})
    batch_size = int(fit_hyperparameters.get('batch_frac', 0.25) * train_samples)
    lr = fit_hyperparameters.get('lr', 1e-2)
    epochs = fit_hyperparameters.get('epochs', 100)
    train_val_split = fit_hyperparameters.get('train_val_split', 0.75)
    seed = args.seed
    patience = fit_hyperparameters.get('patience', epochs)
    
    # training dir
    parent_training_dir = os.path.join(args.base_dir, dataset, 'm-esig' if martingale_indices else 'esig')
    os.makedirs(parent_training_dir, exist_ok=True)
    training_dir = os.path.join(parent_training_dir, f'run_{seed}')
    os.makedirs(training_dir, exist_ok=True)

    configs = {
        'data': {
            'dataset': dataset,
            'L1': L1,
            'dim': dim,
            'begin': begin,
            'end': end
        },
        'model': {
            'L1': L1,
            'L2': L2,
            'alpha': alpha, 
            'level': level, 
            'C': C, 
            'a': a, 
            'K': K,
            'martingale_indices': martingale_indices
        }, 
        'fit': {
            'batch_size': batch_size,
            'lr': lr,
            'epochs': epochs,
            'train_val_split': train_val_split,
            'seed': seed,
            'patience': patience
        }
    }

    with open(os.path.join(training_dir, 'configs.yaml'), 'w') as f:
        yaml.dump(configs, f)

    # set seed
    torch.manual_seed(args.seed)

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

    # split train set into train val
    np.random.seed(0) # fix train val split across different training runs
    indices = np.random.permutation(train_samples)
    split = int(train_val_split * train_samples)
    train_X, val_X = train_X[indices[:split]], train_X[indices[split:]]
    train_y, val_y = train_y[indices[:split]], train_y[indices[split:]]

    # add timestamps to time series as required by model architecture
    train_X = add_timestamps(train_X, known_times=known_times, new_times=new_times)
    val_X = add_timestamps(val_X, known_times=known_times, new_times=new_times)
    test_X = add_timestamps(test_X, known_times=known_times, new_times=new_times)

    training_data = TensorDataset(to_tensor(train_X).to(device), to_tensor(train_y).long().to(device))
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    val_data = TensorDataset(to_tensor(val_X).to(device), to_tensor(val_y).long().to(device))
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=True)

    test_data = TensorDataset(to_tensor(test_X).to(device), to_tensor(test_y).long().to(device))
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    # Initialize loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Define the path to save the checkpoint
    checkpoint_path = os.path.join(training_dir, 'checkpoint.pth')
    history_path = os.path.join(training_dir, 'history.pkl')

    # Load from checkpoint if it exists
    if os.path.isfile(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        assert len(history) == start_epoch
    else:
        start_epoch = 0  # Start from the beginning if no checkpoint exists
        best_loss = float('inf')
        early_stopping_counter = 0
        history = []

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            
            running_loss += loss.item()  # Accumulate loss
        
        # Average loss and accuracy for the training epoch
        train_loss = running_loss / len(train_loader)        
        train_accuracy = evaluate_accuracy(model, train_loader)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%.')

        # Validation step
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        with torch.no_grad():  # No gradient calculation for validation
            for inputs, labels in val_loader:
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute validation loss
                val_running_loss += loss.item()  # Accumulate validation loss

        # Average loss and accuracy for the validation epoch
        val_loss = val_running_loss / len(val_loader)
        val_accuracy = evaluate_accuracy(model, val_loader)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        history.append({'Training Loss': train_loss, 'Training Accuracy': train_accuracy, 'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy})

        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            early_stopping_counter = 0  # Reset counter if loss improves
            print("Improved! Saving model...")
            torch.save(model.state_dict(), os.path.join(training_dir, 'best_model.pth'))  # Save the best model
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break  # Stop training if patience is exceeded

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,  # Save next epoch
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'early_stopping_counter': early_stopping_counter,
        }, checkpoint_path)

    print("Training completed.")

    # Load the best model weights
    model.load_state_dict(torch.load(os.path.join(training_dir, 'best_model.pth')))

    model.eval()  # Set the model to evaluation mode

    metrics = {}
    for dataset_type, loader in zip(["Train", "Test"], [train_loader, test_loader]):
        running_loss = 0.0
        with torch.no_grad():  # No gradient calculation for testing
            for inputs, labels in loader:
                torch.manual_seed(0)
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute testing loss
                running_loss += loss.item()  # Accumulate testing loss

        # Average loss for the test set
        loss = running_loss / len(loader)
        print(f'{dataset_type} Loss: {loss:.4f}')

        # Calculate and print accuracy on validation set
        accuracy = evaluate_accuracy(model, loader)
        print(f'{dataset_type} Accuracy: {accuracy:.2f}%')

        metrics[f'{dataset_type} Loss'] = loss
        metrics[f'{dataset_type} Accuracy'] = accuracy

    with open(os.path.join(training_dir, 'eval_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Meta parameters
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-data_dir', default='./datasets', type=str)
    parser.add_argument('-hyperparameters_config', default='./hyperparameters.yaml', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-device', default=0, type=int)
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-run', default=None)
    parser.add_argument('-dataset', default='MBvsFMB', type=str)
    parser.add_argument('-martingale_indices', default=None, nargs='+', type=int)

    args = parser.parse_args()
    train(args)