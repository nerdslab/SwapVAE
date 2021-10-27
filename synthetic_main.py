import os
import numpy as np

from absl import app
from absl import flags

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from neural_kits.utils import set_random_seeds, onlywithin_indices
from neural_kits.dataset import LocalGlobalGenerator
from synthetic_data import sample_sequence, get_angular_data_synthetic
from vae_kits.model import swapVAE_neural
from vae_kits.trainers import swap_VAE_neural_Learner


FLAGS = flags.FLAGS

# Random seed
flags.DEFINE_integer('random_seed', 999, 'Random seed.')

# model
flags.DEFINE_integer('l_size', 32, 'Representation size.')
flags.DEFINE_integer('len', 4, 'Length of the sampled sequence.')
flags.DEFINE_float('alpha', 10, 'kl loss weight')
flags.DEFINE_float('beta', 1, 'l2 loss weight.')

flags.DEFINE_float('lr', 5e-4, 'Base learning rate.')
flags.DEFINE_integer('num_epochs', 7, 'Number of training epochs.')

# log
flags.DEFINE_string('TB_logs', 'synthetic', 'checkpoint and log file name.')
flags.DEFINE_string('TB_logs_folder', 'SwapVAE', 'Tensorboard log folder name.')


class spatial_only_synthetic__(Dataset):
    def __init__(self, data_loader, transform=None, target_transform=None):

        self.transform, self.target_transform = transform, target_transform

        self.firing_rates = torch.flatten(torch.Tensor(data_loader.firing_rates), start_dim=0, end_dim=1)

        labels = torch.flatten(torch.Tensor(data_loader.direction_label), start_dim=0, end_dim=1)
        angles = (2 * np.pi / 8 * labels)[:, np.newaxis]
        self.labels = torch.squeeze(torch.Tensor(np.concatenate([np.cos(angles), np.sin(angles)], axis=1)))

        # print(self.firing_rates.shape, self.labels.shape)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        x, target = self.firing_rates[index, :], self.labels[index, :]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x, target

    def __len__(self):
        return self.firing_rates.shape[0]


def main(argv):
    set_random_seeds(FLAGS.random_seed)

    # progress recording
    TB_LOG_NAME = FLAGS.TB_logs
    if not os.path.exists("ckpt/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger(FLAGS.TB_logs_folder, name=TB_LOG_NAME)

    # load data
    train_path = "./data/sim/sim_100d_poisson_ran_train.npz"
    test_path = "./data/sim/sim_100d_poisson_ran_test.npz"
    train_dat = np.load(train_path)
    test_dat = np.load(test_path)

    number_neurons = 100

    ### a data generator that produces paired x_true (needs pseduo u_true info to pair)
    train_loader = sample_sequence(train_dat, len=FLAGS.len)
    test_loader = sample_sequence(test_dat, len=FLAGS.len)

    train_angular, test_angular = get_angular_data_synthetic(train_loader, test_loader)

    # models
    sequence_lengths = train_loader.sequence_length
    firing_rates = torch.flatten(torch.Tensor(train_loader.firing_rates), start_dim=0, end_dim=1).numpy()
    print(firing_rates.shape)

    pair_sets = onlywithin_indices(sequence_lengths, k_min=-3, k_max=3)
    generator = LocalGlobalGenerator(firing_rates, pair_sets, sequence_lengths,
                                     num_examples=firing_rates.shape[0],
                                     batch_size=256,
                                     pool_batch_size=0,
                                     transform=None, num_workers=1,
                                     structured_transform=True)
    train_data = DataLoader(generator, num_workers=1, drop_last=True)

    model = swapVAE_neural(s_dim=int(FLAGS.l_size/2), l_dim=FLAGS.l_size, input_size=number_neurons,
                           hidden_dim=[number_neurons, FLAGS.l_size], batchnorm=True)

    learner = swap_VAE_neural_Learner(
        alpha=FLAGS.alpha,
        beta=FLAGS.beta,
        net=model,
        train_angular=train_angular,
        test_angular=test_angular,
        transform=None,
        TB_LOG_NAME=TB_LOG_NAME,
        SAVE=1,
        LR=FLAGS.lr,
        l_size=FLAGS.l_size,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1, max_epochs=FLAGS.num_epochs,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, train_data)


if __name__ == "__main__":
    print(f'PyTorch version: {torch.__version__}')

    app.run(main)