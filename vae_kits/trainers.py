import os

import torch
import pytorch_lightning as pl

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from neural_kits import neural_tasks

class VAE_neural_Learner(pl.LightningModule):
    def __init__(self, net, train_angular, test_angular,
                 TB_LOG_NAME, SAVE, LR, l_size=20, transform=None,
                 train_mm=None, test_mm=None):
        super().__init__()
        self.net = net
        self.train_angular, self.test_angular = train_angular, test_angular
        self.transform = transform
        self.train_mm, self.test_mm = train_mm, test_mm
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.LR = LR
        self.l_size = l_size

    def forward(self, img):
        if self.transform is not None:
            img = self.transform(img)
        return self.net(img)

    def _step(self, batch, batch_idx):
        x, _ = batch
        recon_loss, kl_loss = self.forward(x)

        alpha = 1
        loss = recon_loss + alpha* kl_loss
        
        return {'loss': loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss}

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch, batch_idx)

        self.logger.log_metrics({'Loss/total': outputs['loss'],
                                 'Loss/recon': outputs['recon_loss'],
                                 'Loss/kl': outputs['kl_loss']}, step=self.global_step)
        return {'loss': outputs['loss']}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()

        # classification acc
        classifier = torch.nn.Sequential(torch.nn.Linear(self.l_size, 2)).to('cuda')
        class_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

        acc, delta_acc = neural_tasks.train_angle_classifier(
            self.net._representation, classifier, self.train_angular, self.test_angular, class_optimizer,
            transform=None, transform_val=None, device='cuda',
            num_epochs=100, batch_size=256)

        self.logger.log_metrics({'trial_angles/acc_train': acc.train_smooth,
                                 'trial_angles/delta_acc_train': delta_acc.train_smooth,
                                 'trial_angles/acc_test': acc.val_smooth,
                                 'trial_angles/delta_acc_test': delta_acc.val_smooth}, step=self.current_epoch)

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            torch.save(self.net.state_dict(), os.path.join("ckpt", self.TB_LOG_NAME, "epoch{}.pth".format(self.current_epoch)))

        self.net.train()


class swap_VAE_neural_Learner(VAE_neural_Learner):
    def __init__(self, alpha=1, beta=1, check_clf=1, **kwargs):
        super().__init__(**kwargs)
        # assert self.transform is not None
        self.alpha = alpha
        self.beta = beta
        self.acc_best = 0
        self.delta_acc_best = 0
        self.normal_data = False
        self.check_clf = check_clf

    def forward(self, img, force_val=False):
        if self.normal_data and self.transform is not None:
            img1, img2 = self.transform(img)
        elif self.normal_data and self.transform is None:
            img1, img2 = img, img
        elif force_val:
            img1, img2 = self.transform(img)
        else:
            assert self.normal_data is False
            img1, img2 = img[0], img[1]

        return self.net(img1, img2)

    def _step(self, x, force_val=False):
        l2_loss, recon_loss, kl_loss, recon_ori_loss, images = self.forward(x, force_val)

        loss = recon_loss + recon_ori_loss + self.alpha* kl_loss + self.beta* l2_loss
        # time.sleep(5)
        return {'loss': loss,
                'recon_loss': recon_loss,
                'recon_ori_loss': recon_ori_loss,
                'l2_loss': l2_loss,
                'kl_loss': kl_loss,
                'images': images,
                }

    def training_step(self, batch, batch_idx):

        if self.normal_data:
            x, _ = batch
            outputs = self._step(x)
        else:
            x1 = torch.squeeze(batch[0])
            x2 = torch.squeeze(batch[1])
            x = [x1, x2]
            outputs = self._step(x)

        self.logger.log_metrics({'Loss/total': outputs['loss'],
                                 'Loss/recon': outputs['recon_loss'],
                                 'Loss/recon_ori': outputs['recon_ori_loss'],
                                 'Loss/l2': outputs['l2_loss'],
                                 'Loss/kl': outputs['kl_loss']}, step=self.global_step)
        return {'loss': outputs['loss']}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()

        if (self.current_epoch + 1) % self.check_clf == 0:
            # classification acc
            classifier = torch.nn.Sequential(torch.nn.Linear(self.l_size, 2)).to('cuda')
            class_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

            acc, delta_acc, add = neural_tasks.train_angle_classifier(
                self.net._representation, classifier, self.train_angular, self.test_angular, class_optimizer,
                transform=None, transform_val=None, device='cuda', num_epochs=300, batch_size=256)

            if acc.val_smooth > self.acc_best:
                self.acc_best = acc.val_smooth
            if delta_acc.val_smooth > self.delta_acc_best:
                self.delta_acc_best = delta_acc.val_smooth

            self.logger.log_metrics({'trial_angles/acc_train': acc.train_smooth,
                                     'trial_angles/delta_acc_train': delta_acc.train_smooth,
                                     'trial_angles/acc_test': acc.val_smooth,
                                     'trial_angles/delta_acc_test': delta_acc.val_smooth,
                                     'trial_angles/best_acc': add["best_acc"],
                                     'trial_angles/best_delta_acc': add["best_delta_acc"],}, step=self.current_epoch)

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            torch.save(self.net.state_dict(), os.path.join("ckpt", self.TB_LOG_NAME, "epoch{}.pth".format(self.current_epoch)))

        self.net.train()

