import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class Simple_Trans(Dataset):
    def __init__(self, data, transform=None):
        # [reps, labels]
        self.reps = data[0]
        self.labels = data[1]
        # print(self.reps.shape, self.labels.shape) # torch.Size([60000, 64]) torch.Size([60000])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.reps[idx, :], self.labels[idx]


class linear_clf(object):
    def __init__(self, net, classifier, optimizer, train_dataloader, test_dataloader, device = "cpu", batch_size=1024,
                 num_epochs = 10, disable_tqdm = False, writer=None, writer_tag = "", pair=False):
        self.net = net
        #self.net.eval()

        self.classifier = classifier
        self.optimizer = optimizer
        self.writer = writer
        self.tag = writer_tag

        self.disable_tqdm = disable_tqdm
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.data_train = Simple_Trans(self.compute_representations(train_dataloader))
        self.data_test = Simple_Trans(self.compute_representations(test_dataloader))

        self.best_number = 0
        self.train_linear_layer()

        self.train_acc = self.compute_accuracy(DataLoader(self.data_train, batch_size=batch_size))
        self.test_acc = self.compute_accuracy(DataLoader(self.data_test, batch_size=batch_size))
        #self.net.train()

    def compute_representations(self, dataloader):
        """ store the representations
        :param net: ResNet or smth
        :param dataloader: train_loader and test_loader
        """
        #self.net.eval()
        reps, labels = [], []

        for i, (x, label) in enumerate(dataloader):
            # load data
            x = x.to(self.device)
            labels.append(label)

            # forward
            with torch.no_grad():
                representation = self.net(x)
                reps.append(representation.detach().cpu())

            if i % 100 == 0:
                reps = [torch.cat(reps, dim=0)]
                labels = [torch.cat(labels, dim=0)]

        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        #self.net.train()
        return [reps, labels]

    def compute_accuracy(self, dataloader):
        #self.net.eval()
        self.classifier.eval()
        right = []
        total = []
        for x, label in dataloader:
            x, label = x.to(self.device), label.to(self.device)
            # feed to network and classifier
            with torch.no_grad():
                pred_logits = self.classifier(x)
            # compute accuracy
            _, pred_class = torch.max(pred_logits, 1)
            right.append((pred_class == label).sum().item())
            total.append(label.size(0))
        self.classifier.train()
        #self.net.train()
        return sum(right) / sum(total)

    def train_linear_layer(self):
        #self.net.eval()
        class_criterion = torch.nn.CrossEntropyLoss()
        progress_bar = tqdm(range(self.num_epochs), disable=self.disable_tqdm, position=0, leave=True)
        for epoch in progress_bar:
            for x, label in DataLoader(self.data_train, batch_size=self.batch_size):
                self.classifier.train()
                x, label = x.to(self.device), label.to(self.device)
                pred_class = self.classifier(x)
                loss = class_criterion(pred_class, label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            curr_number = self.compute_accuracy(DataLoader(self.data_test, batch_size=self.batch_size))
            if curr_number >= self.best_number:
                self.best_number = curr_number

            if self.writer is not None:
                self.writer.log_metrics({'CLFtraining/val-tag{}'.format(self.tag): curr_number}, step = epoch)

            progress_bar.set_description('Linear_CLF Epoch: [{}/{}] Acc@1:{:.3f}% BestAcc@1:{:.3f}%'
                                         .format(epoch, self.num_epochs, curr_number, self.best_number))
        #self.net.train()


class linear_clf_wlabels(object):
    def __init__(self, net, classifier, optimizer, train_dataloader, test_dataloader, device = "cpu", batch_size=1024,
                 num_epochs = 10, disable_tqdm = False, writer=None, writer_tag = "", pair=False):
        self.net = net
        #self.net.eval()

        self.classifier = classifier
        self.optimizer = optimizer
        self.writer = writer
        self.tag = writer_tag

        self.disable_tqdm = disable_tqdm
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.data_train = Simple_Trans(self.compute_representations(train_dataloader))
        self.data_test = Simple_Trans(self.compute_representations(test_dataloader))

        self.best_number = 0
        self.train_linear_layer()

        self.train_acc = self.compute_accuracy(DataLoader(self.data_train, batch_size=batch_size))
        self.test_acc = self.compute_accuracy(DataLoader(self.data_test, batch_size=batch_size))
        #self.net.train()

    def compute_representations(self, dataloader):
        """ store the representations
        :param net: ResNet or smth
        :param dataloader: train_loader and test_loader
        """
        #self.net.eval()
        reps, labels = [], []

        for i, (x, label) in enumerate(dataloader):
            # load data
            x = x.to(self.device)
            labels.append(label)

            one_hot = torch.zeros(x.shape[0], 8).to('cuda')  # hard code here, label class 8 in total
            one_hot[torch.arange(x.shape[0]), label] = 1

            # forward
            with torch.no_grad():
                representation = self.net(x, one_hot)
                reps.append(representation.detach().cpu())

            if i % 100 == 0:
                reps = [torch.cat(reps, dim=0)]
                labels = [torch.cat(labels, dim=0)]

        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        #self.net.train()
        return [reps, labels]

    def compute_accuracy(self, dataloader):
        #self.net.eval()
        self.classifier.eval()
        right = []
        total = []
        for x, label in dataloader:
            x, label = x.to(self.device), label.to(self.device)
            # feed to network and classifier
            with torch.no_grad():
                pred_logits = self.classifier(x)
            # compute accuracy
            _, pred_class = torch.max(pred_logits, 1)
            right.append((pred_class == label).sum().item())
            total.append(label.size(0))
        self.classifier.train()
        #self.net.train()
        return sum(right) / sum(total)

    def train_linear_layer(self):
        #self.net.eval()
        class_criterion = torch.nn.CrossEntropyLoss()
        progress_bar = tqdm(range(self.num_epochs), disable=self.disable_tqdm, position=0, leave=True)
        for epoch in progress_bar:
            for x, label in DataLoader(self.data_train, batch_size=self.batch_size):
                self.classifier.train()
                x, label = x.to(self.device), label.to(self.device)
                pred_class = self.classifier(x)
                loss = class_criterion(pred_class, label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            curr_number = self.compute_accuracy(DataLoader(self.data_test, batch_size=self.batch_size))
            if curr_number >= self.best_number:
                self.best_number = curr_number

            if self.writer is not None:
                self.writer.log_metrics({'CLFtraining/val-tag{}'.format(self.tag): curr_number}, step = epoch)

            progress_bar.set_description('Linear_CLF Epoch: [{}/{}] Acc@1:{:.3f}% BestAcc@1:{:.3f}%'
                                         .format(epoch, self.num_epochs, curr_number, self.best_number))
        #self.net.train()
