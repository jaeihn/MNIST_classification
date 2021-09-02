import torch
import torch.nn as nn


class FullyConnectedClassifier(nn.Module):
    
    def __init__(self, num_layers):
        
        super().__init__()

        input_size = 28**2
        output_size = 10

        '''determine hidden dimensions'''
        h_dim = [input_size]
        for _ in range(1, num_layers):
            h_dim.append(int(h_dim[-1]/2))

        modules = []
        
        for i in range(len(h_dim)-1):
            modules.append(nn.Linear(h_dim[i], h_dim[i+1]))
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm1d(h_dim[i+1]))
        modules.append(nn.Linear(h_dim[-1], output_size))
        modules.append(nn.Softmax(dim=-1))

        self.layers = torch.nn.Sequential(*modules)
        
    def forward(self, x):
        y_hat = self.layers(x)
        return y_hat


class AutoencoderClassifier(nn.Module):
    
    def __init__(self, num_layers):
        
        super().__init__()

        input_size = 28**2
        output_size = 10

        '''determine hidden dimensions'''
        h_dim = [input_size]
        for _ in range(1, num_layers):
            h_dim.append(int(h_dim[-1]/2))

        encoder_modules = []
        decoder_modules = []
        
        for i in range(len(h_dim)-1):
            encoder_modules.append(nn.Linear(h_dim[i], h_dim[i+1]))
            encoder_modules.append(nn.ReLU())
            encoder_modules.append(nn.BatchNorm1d(h_dim[i+1]))
            decoder_modules = [nn.Linear(h_dim[i+1], h_dim[i])] + decoder_modules
            decoder_modules = [nn.BatchNorm1d(h_dim[i+1])] + decoder_modules
            decoder_modules = [nn.ReLU()]
        encoder_modules.append(nn.Linear(h_dim[-1], output_size))
        decoder_modules.append(nn.Linear(output_size, h_dim[-1]))

        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)
        
    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)

        return y_hat



class ConvolutionBlock(nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=(kernel_size, kernel_size), padding=int(kernel_size/2)),
            nn.ReLU(),
            nn.BatchNorm2d(output_ch),
            nn.Conv2d(output_ch, output_ch, kernel_size=(kernel_size, kernel_size), stride=2, padding=int(kernel_size/2)),
            nn.ReLU(),
            nn.BatchNorm2d(output_ch),
        )
    def forward(self, x):
        y = self.layers(x)
        return y


class ConvolutionalClassifier(nn.Module):

    def __init__(self, kernel_size):

        super().__init__()

        output_size = 10

        self.blocks = nn.Sequential( 
            # each block reduces h and w because of stride=2
            ConvolutionBlock(1, 16, kernel_size), # 14 14
            ConvolutionBlock(16, 32, kernel_size), # 7 7
            ConvolutionBlock(32, 64, kernel_size), # 4 4
            ConvolutionBlock(64, 128, kernel_size), # 2 2
            ConvolutionBlock(128, 256, kernel_size), # 1 1
        )
        self.layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        if x.dim() == 3:
            # |x| = (batch, height, weight)
            # resize to add channel dimension
            x = x.view(-1, 1, x.size(-2), x.size(-1))
            # |x| = (batch_size, 1, h, w)
        # conv blocks reduce h and w to 1 and increase channels to 256
        z = self.blocks(x)
        # only the channels are needed 
        y = self.layers(z.squeeze()) 

        return y


class LSTMClassifier(nn.Module):

    def __init__(self, n_layers):

        super().__init__()

        input_size = 28 # MNIST dimension
        output_size = 10 # 10 digits to classify into 
        hidden_size = 32 # number of lstm units 
        dropout = 0.2

        self.lstm = nn.LSTM(
            input_size=input_size,
            num_layers=n_layers,
            bidirectional=True, 
            batch_first=True,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size*2),
            nn.Linear(hidden_size*2, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        z, (hn, cn) = self.lstm(x)
        # z contains = (batch, sequence length, 2 * hidden output)
        # we are only interested in the final hidden states 
        z = z[:, -1]
        # linear layers for classification 
        y = self.layers(z)
        return y
