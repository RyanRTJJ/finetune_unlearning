from toy_model import *
from data_generator import *
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def sparsity_vs_loss(tensorboard_logdir, clean_logdir_first):
    # args
    N = 1000
    k = 3
    bottleneck_dim = 2
    num_epochs = 300
    sparsities = [i * 0.10 for i in range(0, 10)]
    lrs = [0.05] * 10
    final_losses = []

    writer = SummaryWriter(log_dir=tensorboard_logdir)
    # Clear contents of file first
    if tensorboard_logdir != None:
        if os.path.isdir(tensorboard_logdir) and clean_logdir_first:
            shutil.rmtree(tensorboard_logdir)
        print(f"Done cleaning contents of {tensorboard_logdir}")
        os.makedirs(tensorboard_logdir, exist_ok=True)

    for i, (S, lr) in enumerate(zip(sparsities, lrs)):
        data = generate_data(N, k, S)
        model = AnthropicToyModel(k, bottleneck_dim)
        model.init_weights()
        model, final_loss = train(model, data, num_epochs, lr)
        final_losses.append(final_loss)

        last_y_hat = model(torch.from_numpy(data).float())

        fig, PCA_M, PCA_mu = plot_data(
            last_y_hat.detach().numpy(), 
            pathological_idxs = [], 
            colors=('coral', 'teal'), 
            alpha=0.7,
            fig=None, 
            PCA_M=None, 
            PCA_mu=None)
        fig, _, _ = plot_data(
            data, 
            pathological_idxs = [], 
            colors=('blue', 'pink'), 
            alpha=0.1, 
            fig=fig, 
            PCA_M=PCA_M, 
            PCA_mu=PCA_mu)
        
        writer.add_figure('experiment_sparsity/final_fits', fig, i)

    fig, ax = plt.subplots()
    ax.plot(sparsities, final_losses)
    ax.set_xlabel("Sparsity %")
    ax.set_ylabel("MSELoss")
    writer.add_figure('experiment_sparsity/sparsity_vs_loss', fig)

def embeddings_over_train(tensorboard_logdir):
    N = 1000
    k = 2
    bottleneck_dim = 2
    num_epochs = 300
    S = 0.5
    lr = 0.05
    writer = SummaryWriter(log_dir=tensorboard_logdir)

    data = generate_data(N, k, S)
    model = AnthropicToyModel(k, bottleneck_dim)
    model.init_weights()
    model, final_loss = train(model, data, num_epochs, lr, tensorboard_logdir, True)


if __name__ == '__main__':
    np.random.seed(420)
    tensorboard_logdir = "train_outs/debug"

    # pathological_vector = np.array([1] * k)
    # pathological_idxs = get_indices_within_angle(data, pathological_vector, 45)

    # plot_data(data, pathological_idxs)

    # Experiment 1
    sparsity_vs_loss(tensorboard_logdir, True)

    # Experiment 2
    # embeddings_over_train(tensorboard_logdir)