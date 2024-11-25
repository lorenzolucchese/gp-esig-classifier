import numpy as np

def get_order(L1, dim = 1, division = 1, begin = 0, end = 1):
    '''
    L1: time series length
    dim = 1 - one dimensional
    division = 1 means we are taking the middle points as new time instants -> L2 = L1 - 1
    begin = first time steps
    end = last known time steps
    '''

    # generating the time steps
    known_times = np.linspace(begin, end, L1)
    new_times = np.zeros(division*(L1-1))
    for i in range(0, (L1-1)):
        new_times[(division*i):(division*(i+1))] = np.linspace(known_times[i], known_times[i+1], (division+2))[1:(1 + division)]

    timesteps = np.concatenate((known_times, new_times), axis=0)
    order = np.argsort(timesteps)

    extended_order = np.zeros(dim * len(order))
    for i in range(len(order)):
        extended_order[(i*dim):((i+1)*dim)] = np.arange(order[i]*dim, (order[i] + 1)*dim)

    return known_times, new_times, order, extended_order

def add_timestamps(X, known_times, new_times):
    '''
    X: time series of shape n_samples x L1 x dim
    dataset: n_samples x (L1 + L2 + d*L1)
    '''
    # adding known and unknown time stamps
    timestamps = np.concatenate((known_times, new_times))
    time_data = np.stack([timestamps]*len(X), axis=0)

    # full dataset train
    if X.ndim == 2:
        pass
    elif X.ndim == 3:
        X = X.reshape(len(X), -1)
    dataset = np.concatenate((time_data, X), axis=-1) # full dataset
    dataset = dataset.astype('float32')

    return dataset