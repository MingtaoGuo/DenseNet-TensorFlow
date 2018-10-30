import numpy as np
import scipy.io as sio


def read_cifar_data(path):
    # f = open(path, "rb")
    # data_and_labels = pickle.load(f, encoding="latin1")
    data_and_labels = sio.loadmat(path+".mat")
    data = np.reshape(data_and_labels["data"], [-1, 3, 32, 32])
    data = np.transpose(data, [0, 2, 3, 1])
    return data, np.squeeze(np.array(data_and_labels["labels"], dtype=np.int32), axis=1)

def shuffle(data, labels):
    length = data.shape[0]
    idx = np.arange(0, length)
    np.random.shuffle(idx)
    labels = labels[idx]
    data = data[idx]
    return data, labels


