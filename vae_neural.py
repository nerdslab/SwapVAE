import os
import sys
import numpy
import warnings

from absl import app
from absl import flags
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from neural_kits.dataset import ReachNeuralDataset, get_angular_data, LocalGlobalGenerator
import neural_kits.utils as utils
import neural_kits.transforms as transforms

from vae_kits.dataloaders import spatial_only_neural, spatial_only_neural_angle
from vae_kits.model import VAE_neural, swapVAE_neural,
from vae_kits.trainers import VAE_neural_Learner, swap_VAE_neural_Learner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not sys.warnoptions:
    warnings.simplefilter("ignore")

numpy.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(edgeitems=10000)


FLAGS = flags.FLAGS

# Dataset
flags.DEFINE_string('data_path', './data/mihi-chewie', 'Path to monkey data.')
flags.DEFINE_enum('primate', 'chewie', ['chewie', 'mihi'], 'Primate name.')
flags.DEFINE_integer('day', 1, 'Day of recording.', lower_bound=1, upper_bound=2)
flags.DEFINE_float('binning', 0.1, 'binning_period', lower_bound=0.001, upper_bound=1.)

# Transforms
flags.DEFINE_integer('max_lookahead', 5, 'Max lookahead.')
flags.DEFINE_float('noise_sigma', 0.2, 'Noise sigma.', lower_bound=0.)

flags.DEFINE_float('dropout_p', 0.6, 'Dropout probability.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('dropout_apply_p', 1.0, 'Probability of applying dropout.', lower_bound=0., upper_bound=1.)

flags.DEFINE_float('pepper_p', 0.5, 'Pepper probability.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('pepper_sigma', 10, 'Pepper sigma.', lower_bound=0.)
flags.DEFINE_float('pepper_apply_p', 1.0, 'Probability of applying pepper.', lower_bound=0., upper_bound=1.)
flags.DEFINE_boolean('structured_transform', True, 'Whether the transformations are consistent across temporal shift.')

# Dataloader
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('num_workers', 1, 'Number of workers.')

# architecture
flags.DEFINE_integer('l_size', 128, 'Representation size.')
flags.DEFINE_float('alpha', 10, 'kl loss weight')
flags.DEFINE_float('beta', 1, 'l2 loss weight.')
### temp
flags.DEFINE_integer('ablation_type', 0, 'ablation experiment type')
flags.DEFINE_integer('aug_ablation', 0, 'data augmentation ablation experiment type')

# Training parameters
flags.DEFINE_float('lr', 5e-4, 'Base learning rate.')
flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs.')
flags.DEFINE_integer('check_clf', 1, 'How many epochs to check clf performance.')

# Random seed
flags.DEFINE_integer('random_seed', 100, 'Random seed.')

# Logfile
flags.DEFINE_string('TB_logs', 'test0513-dropout', 'checkpoint and log file name.')
flags.DEFINE_string('TB_logs_folder', 'test0427', 'Tensorboard log folder name.')


def main_swap(argv):
    utils.set_random_seeds(FLAGS.random_seed)

    # load dataset
    dataset = ReachNeuralDataset(FLAGS.data_path, primate=FLAGS.primate, day=FLAGS.day, binning_period=FLAGS.binning,
                                 scale_firing_rates=False, train_split=0.8)

    transform_temp = transforms.Pair_Compose(transforms.RandomizedDropout(FLAGS.dropout_p, apply_p=FLAGS.dropout_apply_p)
                                             )

    train_data = spatial_only_neural(dataset, transform=None, target_transform=None, train='train')
    test_data = spatial_only_neural(dataset, transform=None, target_transform=None, train='test')

    train_data = DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)
    test_data = DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers, drop_last=True)

    train_angular, test_angular = get_angular_data(dataset, device='cuda', velocity_threshold=5)

    # progress recording
    TB_LOG_NAME = FLAGS.TB_logs
    if not os.path.exists("ckpt/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger(FLAGS.TB_logs_folder, name=TB_LOG_NAME)

    number_neurons = next(iter(train_data))[0].shape[1]

    # model and trainer
    model = swapVAE_neural(s_dim=64, l_dim=FLAGS.l_size, input_size=number_neurons,
                           hidden_dim = [number_neurons, 128], batchnorm=True)

    ablation_type = FLAGS.ablation_type

    class swap_VAE_neural_Learner_ablation(swap_VAE_neural_Learner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def _step(self, x, force_val=False):
            l2_loss, recon_loss, kl_loss, recon_ori_loss, images = self.forward(x, force_val)

            # original loss

            if ablation_type == 0:
                loss = recon_loss + recon_ori_loss + self.alpha * kl_loss + self.beta * l2_loss
            elif ablation_type == 1:
                loss = 2 * recon_ori_loss + self.alpha * kl_loss
            elif ablation_type == 2:
                loss = 2 * recon_ori_loss + self.alpha * kl_loss / 5
            elif ablation_type == 3:
                loss = recon_loss + recon_ori_loss + self.alpha * kl_loss
            elif ablation_type == 4:
                loss = 2 * recon_ori_loss + self.alpha * kl_loss + self.beta * l2_loss
            elif ablation_type == 5:
                loss = 2 * recon_loss + self.alpha * kl_loss
            elif ablation_type == 6:
                loss = self.alpha * kl_loss + self.beta * l2_loss
            elif ablation_type == 7:
                loss = self.beta * l2_loss
            else:
                raise NotImplementedError

            return {'loss': loss,
                    'recon_loss': recon_loss,
                    'recon_ori_loss': recon_ori_loss,
                    'l2_loss': l2_loss,
                    'kl_loss': kl_loss,
                    'images': images,
                    }

    learner = swap_VAE_neural_Learner_ablation(
        alpha = FLAGS.alpha,
        beta = FLAGS.beta,
        check_clf = FLAGS.check_clf,
        net = model,
        train_angular = train_angular,
        test_angular = test_angular,
        transform = transform_temp,
        TB_LOG_NAME=TB_LOG_NAME,
        SAVE=20,
        LR=FLAGS.lr,
        l_size=FLAGS.l_size,
        train_mm=train_data,
        test_mm=test_data,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1, max_epochs=FLAGS.num_epochs,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    ##### model fitting
    trainer.fit(learner, train_data)


if __name__ == "__main__":
    print(f'PyTorch version: {torch.__version__}')
    
    app.run(main_swap)




