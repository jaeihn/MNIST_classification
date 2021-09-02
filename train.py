import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from models import FullyConnectedClassifier, AutoencoderClassifier, ConvolutionalClassifier, LSTMClassifier
from common_files.CustomEngine import EngineRun
from dataloader import get_loaders



def config_parser():
    '''parses configuration from system input'''
    p = argparse.ArgumentParser()

    # model parameters
    p.add_argument('--train_valid_ratio', type=float, default=0.8)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    # only flatten for fully connected and autoencoder models 
    p.add_argument('--flatten', default=True) 

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


config = config_parser()

# models with different parameters 
fc_models = ['fc_'+str(x) for x in [2, 4, 6]]
auto_models = ['auto_'+str(x) for x in [2, 4, 6]]
cnn_models = ['cnn_'+str(x) for x in [3, 5, 7]]
lstm_models = ['lstm_'+str(x) for x in [1, 2, 4]]

# switches for models to train 
models = [fc_models, auto_models, cnn_models, lstm_models]
model_switches = [True, True, True, True]

models_to_train = []
for i in range(len(model_switches)):
    if model_switches[i]:
        models_to_train.extend(models[i])

with open('models/models.txt', 'w') as f:
    for M in models_to_train:
        f.write(str(M)+',')
    
# train models 
for M in models_to_train:
    
    # set model configuration
    model = set_model(M)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # get MNIST-specific loaders 
    train_loader, valid_loader, _ = get_loaders(config)

    # let the training begin
    trainer = EngineRun(model, config, optimizer, criterion)
    trainer.train(train_loader, valid_loader)

