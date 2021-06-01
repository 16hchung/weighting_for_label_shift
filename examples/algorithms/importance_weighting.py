import torch
from tqdm import tqdm
from enum import Enum, auto

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from optimizer import initialize_optimizer
from scheduler import initialize_scheduler
from utils import confusion_matrix

class BBSE(SingleModelAlgorithm):

    def __init__(self, configs, d_out, grouper, loss, metric, n_train_steps):
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

    def reset(self, config):
        self.model = initialize_model(config, self.d_out).to(config.device)
        self.optimizer = initialize_optimizer(config, self.model)
        self.schedulers = [initialize_scheduler(config, self.optimizer, self.n_train_steps)]

    def objective(self, results):
        loss = self.loss.compute(results['y_pred'], 
                                 results['y_true'], 
                                 return_dict=False)
        if self.loss_weights is not None:
            import pdb;pdb.set_trace()
            loss = loss * self.loss_weights[results['y_true']]
        return loss

    def setup_for_weighted_training(self, config, datasets, adapt_split='val'):
        # TODO compute and save confusion matrix
        import pdb;pdb.set_trace()
        trainset = datasets['train_adapt']
        evalset = datasets[f'{adapt_split}_adapt']
        assert trainset['dataset'].n_classes == 2
        # compute confusion matrix
        y_true, y_pred = self.eval_on_loader(trainset['loader'])
        confusion = confusion_matrix(y_true, y_pred, 2)
        # estimate test density
        _, y_pred_te = self.eval_on_loader(evalset['loader'])
        pos_freq = y_pred_te.mean()
        neg_freq = 1 - pos_freq
        mu_hat = torch.from_numpy([neg_freq, pos_freq])
        # compute weights
        self.loss_weights = torch.maximum(torch.inverse(confusion) @ mu_hat, 0)
        # reset model to start training
        self.reset(config)

    def eval_on_loader(self, loader)
        epoch_y_true = []
        epoch_y_pred = []
        for batch in tqdm(loader):
            results = self.evaluate(batch)
            epoch_y_true.append(batch_results['y_true'].clone().detach())
            epoch_y_pred.append(batch_results['y_pred'].clone().detach())
        y_true = torch.cat(epoch_y_true)
        y_pred = torch.cat(epoch_y_pred)
        return y_true, y_pred
