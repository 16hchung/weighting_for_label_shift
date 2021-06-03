import torch
from tqdm import tqdm
from enum import Enum, auto

from algorithms.single_model_algorithm import SingleModelAlgorithm
from configs.supported import process_outputs_functions
from models.initializer import initialize_model
from optimizer import initialize_optimizer
from scheduler import initialize_scheduler
from utils import confusion_matrix

class BBSE(SingleModelAlgorithm):

    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, delta=0):
        model = initialize_model(config, d_out).to(config.device)
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.d_out = d_out
        self.n_train_steps = n_train_steps
        self.loss_weights = None
        self.delta = delta

    def reset(self, config):
        self.model = initialize_model(config, self.d_out).to(config.device)
        self.optimizer = initialize_optimizer(config, self.model)
        self.schedulers = [initialize_scheduler(config, self.optimizer, self.n_train_steps)]

    def objective(self, results):
        loss = self.loss.compute_element_wise(results['y_pred'], 
                                              results['y_true'], 
                                              return_dict=False)
        if self.loss_weights is not None:
            loss = loss * self.loss_weights[results['y_true']]
        return loss.mean()

    def setup_for_weighted_training(self, config, datasets, adapt_split='val'):
        ''' returns True if should continue training '''
        # compute and save confusion matrix
        trainset = datasets['train_cm']
        evalset = datasets[f'{adapt_split}_wts']
        assert trainset['dataset'].n_classes == 2
        # compute confusion matrix
        y_true, y_pred = self.eval_on_loader(config, trainset['loader'])
        confusion = confusion_matrix(y_true, y_pred, 2)
        print('===================================================')
        print('CONFUSION MATRIX', confusion)
        eig_min = torch.eig(confusion)[0][:,0].min()
        if eig_min < self.delta:
            return False
        # estimate test density
        _, y_pred_te = self.eval_on_loader(config, evalset['loader'])
        pos_freq = y_pred_te.float().mean()
        neg_freq = 1 - pos_freq
        mu_hat = torch.Tensor([neg_freq, pos_freq])
        # compute weights
        self.loss_weights = torch.clamp(
            torch.pinverse(confusion) @ mu_hat, min=0
        ).to(config.device)
        print('WEIGHTS', self.loss_weights)
        # reset model to start training
        self.reset(config)
        return True

    def eval_on_loader(self, config, loader):
        epoch_y_true = []
        epoch_y_pred = []
        for batch in tqdm(loader):
            results = self.evaluate(batch)
            epoch_y_true.append(results['y_true'].clone().detach())
            epoch_y_pred.append(results['y_pred'].clone().detach())
        y_true = torch.cat(epoch_y_true)
        y_pred = torch.cat(epoch_y_pred)
        y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
        return y_true, y_pred
