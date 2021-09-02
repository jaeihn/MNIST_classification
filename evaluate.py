import torch
import torch.nn

from dataloader import get_loaders
from common_files.CustomEngine import EngineRun
from models import FullyConnectedClassifier, AutoencoderClassifier, ConvolutionalClassifier, LSTMClassifier


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


models_to_test = open('./models/models.txt', 'r').read().split(',')[:-1]

for M in models_to_test:
    # model configuration
    filename = './models/model_'+M+'.pth'

    config = torch.load(filename)['config']
    config.verbose = 1
    _, _, test_loader = get_loaders(config)

    model = set_model(M)
    model.load_state_dict(torch.load(filename)['model'])
    
    tester = EngineRun(model, config)
    accuracy = tester.test(test_loader)

    print('{:<25}{:>25.4f}'.format('ACCURACY OF MODEL %s' % M.upper(), accuracy))