from torch.utils.data import DataLoader

from common_files.utils import shuffle, CustomDataset


def download_mnist(is_train=True):
    '''downloads mnist datasets and flattens x based on configuration'''
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.ToTensor(),
    )

    x = dataset.data.float() / 255.0
    y = dataset.targets

    return x, y


def get_loaders(config):
    '''returns DataLoaders for MNIST classification'''
    # download mnist datasets
    x, y = download_mnist(is_train=True)
    test_x, test_y = download_mnist(is_train=False)

    # flatten x and test_x based on configuration
    if config.flatten:
        x = x.view(x.size(0), -1)
        test_x = test_x.view(test_x.size(0), -1)

    # split into train and validation sets
    x, y = shuffle(x,y)
    train_count = int(x.size(0) * config.train_valid_ratio)
    valid_count = x.size(0) - train_count

    train_x, valid_x = x.split([train_count, valid_count])
    train_y, valid_y = y.split([train_count, valid_count])

    # define and return DataLoaders
    train_loader = DataLoader(
        dataset=CustomDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset=CustomDataset(valid_x, valid_y),
        batch_size=config.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=CustomDataset(test_x, test_y),
        batch_size=len(test_x),
        shuffle=False
    )

    return train_loader, valid_loader, test_loader 