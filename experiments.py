from toy_model import *
from data_generator import *
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def sparsity_vs_loss(tensorboard_logdir, clean_logdir_first):
    # args
    N = 1000
    k = 2
    bottleneck_dim = 2
    num_epochs = 300
    sparsities = [i * 0.10 for i in range(0, 10)]
    lrs = [0.05] * 10
    final_losses = []

    writer = SummaryWriter(log_dir=tensorboard_logdir)
    # Clear contents of file first
    if tensorboard_logdir != None and clean_logdir_first:
        for filename in os.listdir(tensorboard_logdir):
            file_path = os.path.join(tensorboard_logdir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print(f"Done cleaning contents of {tensorboard_logdir}")

    for i, (S, lr) in enumerate(zip(sparsities, lrs)):
        data = generate_data(N, k, S)
        model = ToyModel(k, bottleneck_dim)
        model.init_weights()
        model, final_loss = train(model, data, num_epochs, lr)
        final_losses.append(final_loss)

        last_y_hat = model(torch.from_numpy(data).float())
        fig, PCA_M = plot_data(last_y_hat.detach().numpy(), pathological_idxs = [], colors=('coral', 'teal'), alpha=0.7, fig=None, PCA_M=None)
        fig, _ = plot_data(data, pathological_idxs = [], alpha=0.2, fig=fig, PCA_M=PCA_M)
        
        writer.add_figure('train/final_fits', fig, i)

    fig, ax = plt.subplots()
    ax.plot(sparsities, final_losses)
    ax.set_xlabel("Sparsity %")
    ax.set_ylabel("MSELoss")
    writer.add_figure('train/sparsity_vs_loss', fig)


if __name__ == '__main__':
    np.random.seed(420)
    tensorboard_logdir = "train_outs/debug"

    # pathological_vector = np.array([1] * k)
    # pathological_idxs = get_indices_within_angle(data, pathological_vector, 45)

    # plot_data(data, pathological_idxs)

    sparsity_vs_loss(tensorboard_logdir, True)