import argparse
import torch

from torch.utils.data import Dataset


# collection of useful custom functions

def shuffle(x, y):
    '''Shuffles sample order of data(x) and label(y) while retaining correspondence'''
    shuffle_order = torch.randperm(x.size(0)) # random order of batch size
    x = torch.index_select(x, dim=0, index=shuffle_order)
    y = torch.index_select(y, dim=0, index=shuffle_order)

    return x, y


class CustomDataset(Dataset):
    '''custom dataset format'''
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def test(model, x, y):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)

        correct_count = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_count = float(y.size(0))

        accuracy = correct_count / total_count

        print("Test accuracy %.4f" % accuracy)


def config_parser():
    '''parses configuration from system input'''
    p = argparse.ArgumentParser()

     # *.pth file name of model to save
    # p.add_argument('--model_filename', type=str, default='model_fc.pth')

    # model parameters
    p.add_argument('--train_valid_ratio', type=float, default=0.8)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)

    # should we flatten input during preprocessing? 
    p.add_argument('--flatten', default=True) 

    # choose between FULLY_CONNECTED, CNN, VANILLA_RNN, LSTM models
    p.add_argument('--model', type=str, default='fc')
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--dropout_p', type=float, default=.2)
    p.add_argument('--hidden_size', type=int, default=64)

    p.add_argument('--verbose', type=int, default=1)


    config = p.parse_args()

    return config

def set_model(model_type):
    '''set which model to use for MNIST classification task'''
    config.model = model_type
    model_type = model_type.split('_')

    if model_type[0]=='fc':
        model = FullyConnectedClassifier(num_layers=int(model_type[-1]))
        config.flatten = True
    elif model_type[0]=='auto':
        model = AutoencoderClassifier(num_layers=int(model_type[-1]))
        config.flatten = True
    elif model_type[0]=='cnn':
        model = ConvolutionalClassifier(kernel_size=int(model_type[-1]))
        config.flatten = False
    elif model_type[0]=='lstm':
        model = LSTMClassifier(n_layers=int(model_type[-1]))
        config.flatten = False        
    else:
        raise NotImplementedError('specify model')
    return model
