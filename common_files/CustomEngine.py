import numpy as np
import torch

from copy import deepcopy
from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar


# define engines

class CustomEngine(Engine):
    '''Customizing Ignite Engine for my convenience'''
    def __init__(self, process_func, model, config, optimizer=None, criterion=None):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion

        super().__init__(process_func)

        self.min_loss = np.inf
        self.best_model = None


def train_process(engine, batch):
    engine.model.train()
    engine.optimizer.zero_grad()

    x, y = batch
    
    y_hat = engine.model(x)
    loss = engine.criterion(y_hat, y)
    loss.backward()

    # discern between regression and classification task
    if isinstance(y, torch.LongTensor): # classification task
        accuracy = (y == torch.argmax(y_hat, dim=-1)).sum() / float(y.size(0))
    else: # regression task 
        accuracy = 0 

    engine.optimizer.step()

    return {
        'loss': float(loss),
        'accuracy': float(accuracy)
    }


def valid_process(engine, batch):

    engine.model.eval()

    with torch.no_grad():
        x, y = batch
        y_hat = engine.model(x)
        loss = engine.criterion(y_hat, y)

        # discern between regression and classification task
        if isinstance(y, torch.LongTensor): # classification task
            accuracy = (y == torch.argmax(y_hat, dim=-1)).sum() / float(y.size(0))
        else: # regression task 
            accuracy = 0 

    return {
        'loss': float(loss),
        'accuracy': float(accuracy)
    }


def test_process(engine, batch):
    engine.model.eval()

    with torch.no_grad():
        x, y = batch
        y_hat = engine.model(x)

        # discern between regression and classification task
        if isinstance(y, torch.LongTensor): # classification task
            accuracy = (y == torch.argmax(y_hat, dim=-1)).sum() / float(y.size(0))
        else: # regression task 
            accuracy = 0 

    return {
        'accuracy': float(accuracy)
    }


def running_average(engine, metric_name):
    RunningAverage(output_transform=lambda x: x[metric_name]).attach(engine, metric_name,)

def log_progress(engine, dataset, metric_names):
    
    for metric_name in metric_names:
        running_average(engine, metric_name)
    
    progress = ProgressBar(bar_format=None, ncols=100)
    progress.attach(engine, metric_names)

    @engine.on(Events.EPOCH_COMPLETED)
    def print_logs(engine):
        if dataset=='train':
            print('[EPOCH {}] train_loss={:.4e} train_acc={:.4f}'.format(
                engine.state.epoch,
                engine.state.metrics['loss'],
                engine.state.metrics['accuracy'],
                ))
        if dataset=='valid':
            print('valid_loss={:.4e} valid_acc={:.4f} min_loss={:.4e}'.format(
            engine.state.metrics['loss'],
            engine.state.metrics['accuracy'],
            engine.min_loss,
            ))

def print_sample_size(samples):
    print('='*50)
    for name, length in samples: 
        print('{:<25}{:>25}'.format('# OF %s SAMPLES' % name, length))


def check_best(engine):
    '''checks for minimum loss'''
    loss = float(engine.state.metrics['loss'])
    if loss <= engine.min_loss:
        engine.min_loss = loss
        engine.best_model = deepcopy(engine.model.state_dict())

def save_model(engine, config):
    '''save model to file system'''
    filename = 'models/model_' + config.model + '.pth'
    torch.save({
            'model': engine.best_model,
            'config': config,
        }, filename
    )


class EngineRun():

    def __init__(self, model, config, optimizer=None, criterion=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config

        super().__init__()

    def train(self, train_loader, valid_loader):
        # define engines (custom engine format)
        train_engine = CustomEngine(train_process, self.model, self.config, self.optimizer, self.criterion)
        valid_engine = CustomEngine(valid_process, self.model, self.config, self.optimizer, self.criterion)

        # attach metrics to print 
        # attach_train(train_engine, valid_engine)
        log_progress(train_engine, 'train', ['loss', 'accuracy'])
        log_progress(valid_engine, 'valid', ['loss', 'accuracy'])

        train_engine.add_event_handler(
            Events.STARTED, 
            print_sample_size,
            [('TRAIN', len(train_loader.dataset)), ('VALID', len(valid_loader.dataset))])

        train_engine.add_event_handler(
            Events.STARTED, 
            lambda _: print(
                '='*50 + '\n{:^50}\n'.format('START OF TRAINING MODEL %s' % self.config.model.upper())
            ))
    
        # every time a TRAINING epoch is completed, perform a VALIDATION epoch  
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            valid_engine.run,
            valid_loader, max_epochs=1,
        )

        # at the end of each VALIDATION epoch, compare for MIN_LOSS
        valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            check_best,
            valid_engine,
        )

        # at the end of each VALIDATION epoch, save best model
        valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            save_model,
            valid_engine, self.config,
        )

        train_engine.add_event_handler(Events.COMPLETED, lambda _: print(
            '\n{:^50}\n'.format('END OF TRAINING') + '='*50
        ))

        # let the training begin
        train_engine.run(train_loader, max_epochs=self.config.n_epochs,)
        # reload best model 
        self.model.load_state_dict(valid_engine.best_model)

        return self.model

    def test(self, test_loader):
        test_engine = CustomEngine(test_process, self.model, self.config)
        
        log_progress(test_engine, 'test', ['accuracy'])

        if self.config.verbose==2:
            test_engine.add_event_handler(
                Events.STARTED, 
                print_sample_size,
                [('TEST', len(test_loader.dataset))])

            test_engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _: print(
                '\n{:<25}{:>25.4f}'.format('ACCURACY OF MODEL %s' % self.config.model.upper(), test_engine.state.metrics['accuracy'])
            ))

            test_engine.add_event_handler(Events.COMPLETED, lambda _: print(
                '\n{:^50}\n'.format('END OF TEST') + '='*50
            ))

        test_engine.run(test_loader,max_epochs=1,)

        return test_engine.state.metrics['accuracy']


