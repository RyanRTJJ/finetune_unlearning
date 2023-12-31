import torch
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_generator import plot_data
import matplotlib.pyplot as plt
import numpy as np

class AnthropicToyModel(torch.nn.Module):
    """
    This model follows this equation. For any input vector x:
    model(x) = ReLU (W.T @ W @ x + b)

    In reality, since we are dealing with matrix equations, it's more like
    model(X) = ReLU (X @ W @ W.T + b)

    This model is not entirely intuitive to me yet, so I created my own below.
    """
    def __init__(self, input_dim, hidden_dim):
        super(AnthropicToyModel, self).__init__()

        # Manual Xavier initialization
        W_bound = np.sqrt(6 / (input_dim + hidden_dim))
        b_bound = np.sqrt(6 / (input_dim + 1))
        self.W = torch.nn.Parameter(torch.Tensor(input_dim, hidden_dim).uniform_(-W_bound, W_bound))
        self.b = torch.nn.Parameter(torch.Tensor(1, input_dim).uniform_(-b_bound, b_bound))
        self.final_activation = torch.nn.ReLU()

    def init_weights(self):
        pass
    
    def forward(self, x):
        return self.final_activation(x @ self.W @ self.W.T + self.b)


class RyanToyModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ToyModel, self).__init__()

        # "down-sample"
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()

        # "up-sample"
        self.linear2 = torch.nn.Linear(hidden_dim, input_dim)
        self.final_activation = torch.nn.Identity()
    
    def init_weights(self):
        for layer in [self.linear1, self.linear2]:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        z = self.linear1(x)
        z_act = self.activation(z)
        y = self.linear2(z_act)
        y_act = self.final_activation(y)
        return y_act

def create_lr_scheduler(lr_warmup_ratio, lr_constant_ratio, num_training_steps, optimizer):
    """
    Desc:           Set up the scheduler. The optimizer of the trainer must have been set up before this method is called.
                    This method was built based on https://arxiv.org/pdf/2006.13979 :
                    "The learning rate schedule has three phases: warm up for the first 10% of updates, 
                    keep constant for 40% and then linearly decay for the remainder"
    
    @param lr_warmup_ratio:         (float) proportion of training steps to apply warmup LR
    @param lr_constant_ratio:       (float) proportion of training steps to apply constant LR
    @param num_training_steps:      (int)
    @param optimizer:               (torch.optim.Optimizer) something like Adam
    """
    def lr_lambda(current_step):
        """
        @param current_step:        (int)
        """
        warmup_steps = int(num_training_steps * lr_warmup_ratio)
        constant_steps = int(num_training_steps * lr_constant_ratio)
        
        if current_step < warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, warmup_steps))
        elif (lr_warmup_ratio + lr_constant_ratio) == 1.0 or current_step < (warmup_steps + constant_steps):
            # Constant phase
            return 1
        else:
            # Linear decay phase
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (warmup_steps + constant_steps)))
            )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(model, X, num_epochs, lr, tensorboard_logdir=None, clean_logdir_first=True):
    """
    Desc :          Trains model self-supervised style to reconstruct X.
                    Implicit assumption: model significantly overdetermined

    @param model:                       (Torch.nn.Module)
    @param X:                           (np.ndarray)
    @param num_epochs:                  (int) num epochs to train
    @param tensorboard_logdir:          (str)
    """
    if tensorboard_logdir != None:
        if os.path.isdir(tensorboard_logdir) and clean_logdir_first:
            shutil.rmtree(tensorboard_logdir)
            print(f"Done cleaning contents of {tensorboard_logdir}")
        os.makedirs(tensorboard_logdir, exist_ok=True)

    if tensorboard_logdir != None:
        writer = SummaryWriter(log_dir=tensorboard_logdir)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0001)
    lr_scheduler = create_lr_scheduler(
        lr_warmup_ratio=0.05,
        lr_constant_ratio=0.45,
        num_training_steps=num_epochs, 
        optimizer=optimizer
    )

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        
        X_tensor = torch.from_numpy(X).float()
        y_hat = model(X_tensor)
        loss = loss_fn(y_hat, X_tensor)

        # Tensorboard stuff
        if tensorboard_logdir != None:
            writer.add_scalar("train/MSELoss", loss, epoch)
            fig, PCA_M, PCA_mu = plot_data(
                y_hat.detach().numpy(),
                pathological_idxs = [],
                colors=('coral',
                'teal'),
                fig=None,
                PCA_M=None,
                PCA_mu=None)
            fig, _, _ = plot_data(
                X,
                pathological_idxs = [],
                fig=fig,
                PCA_M=PCA_M,
                PCA_mu=PCA_mu)
            writer.add_figure('train/plot', fig, epoch)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    if tensorboard_logdir != None:
        writer.flush()

    return model, loss.detach()
