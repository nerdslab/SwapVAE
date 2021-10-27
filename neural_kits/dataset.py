import os
import time
import pickle
import numpy as np
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

import scipy
from scipy import io as spio
import torch
from torch.utils.data import IterableDataset

FILENAMES = {
    ('mihi', 1): 'full-mihi-03032014',
    ('mihi', 2): 'full-mihi-03062014',
    ('chewie', 1): 'full-chewie-10032013',
    ('chewie', 2): 'full-chewie-12192013',
}


def loadmat(filename):
    r"""This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    """

    def _check_keys(d):
        r"""Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries.
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        r"""A recursive function which constructs from matobjects nested dictionaries."""
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        r"""A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

class ReachNeuralDataset:
    '''edited by Ran to suit the computation of psth firing rate'''
    def __init__(self, path, primate='mihi', day=1,
                 binning_period=0.1, binning_overlap=0.0, train_split=0.8,
                 scale_firing_rates=False, scale_velocity=False, sort_by_reach=False):
        '''sort_by_reach=True is the previous default'''

        # get path to data
        self.path = path

        assert primate in ['mihi', 'chewie']
        assert day in [1, 2]
        self.filename = FILENAMES[(primate, day)]

        self.raw_path = os.path.join(self.path, 'raw/%s.mat') % self.filename
        self.processed_path = os.path.join(self.path, 'processed_ran/%s.pkl') % (self.filename + '-%.2f' % binning_period)

        # get binning parameters
        self.binning_period = binning_period
        self.binning_overlap = binning_overlap
        if self.binning_overlap != 0:
            raise NotImplemented

        # train/val split
        self.train_split = train_split

        # initialize some parameters
        self.dataset_ = {}
        self.subset = 'train'  # default selected subset

        ### Process data
        # load data --> data is not shuffled
        if not os.path.exists(self.processed_path):
            data_train_test = self._process_data()
        else:
            data_train_test = self._load_processed_data()

        # split data
        data_train, data_test = self._split_data(data_train_test)
        self._num_trials = {'train': len(data_train['firing_rates']),
                            'test': len(data_test['firing_rates'])}

        # compute mean and std of firing rates
        self.mean, self.std = self._compute_mean_std(data_train, feature='firing_rates')

        # remove neurons with no variance
        data_train, data_test = self._remove_static_neurons(data_train, data_test)

        # scale data
        if scale_firing_rates:
            data_train, data_test = self._scale_data(data_train, data_test, feature='firing_rates')
        if scale_velocity:
            data_train, data_test = self._scale_data(data_train, data_test, feature='velocity')

        # sort by reach direction
        if sort_by_reach:
            data_train = self._sort_by_reach_direction(data_train)
            data_test = self._sort_by_reach_direction(data_test)

        #################
        #################
        # below things are replaced. Because if the data is merged, it can not be used to plot the psth
        #################
        #################

        # original, not concatenated data and labels
        data_train['original_data'] = data_train['firing_rates'].copy()
        data_test['original_data'] = data_test['firing_rates'].copy()
        data_train['original_label'] = data_train['labels'].copy()
        data_test['original_label'] = data_test['labels'].copy()

        # build sequences
        trial_lengths_train = [seq.shape[0] for seq in data_train['firing_rates']]

        # merge everything
        for feature in data_train.keys():
            if feature != "original_data" and feature != "original_label"\
                    and feature != 'position_raw' and feature != 'velocity_raw':
                data_train[feature] = np.concatenate(data_train[feature]).squeeze()
                data_test[feature] = np.concatenate(data_test[feature]).squeeze()

        data_train['trial_lengths'] = trial_lengths_train
        data_train['reach_directions'] = np.unique(data_train['labels']).tolist()
        data_train['reach_lengths'] = [np.sum(data_train['labels'] == reach_id)
                                       for reach_id in data_train['reach_directions']]

        # map labels to 0 .. N-1 for training
        data_train['raw_labels'] = data_train['labels'].copy()
        data_test['raw_labels'] = data_test['labels'].copy()

        data_train['labels'] = self._map_labels(data_train)
        data_test['labels'] = self._map_labels(data_test)

        self.dataset_['train'] = data_train
        self.dataset_['test'] = data_test

    @property
    def dataset(self):
        return self.dataset_[self.subset]

    def __getattr__(self, item):
        return self.dataset[item]

    def train(self):
        self.subset = 'train'

    def test(self):
        self.subset = 'test'

    @property
    def num_trials(self):
        return self._num_trials[self.subset]

    @property
    def num_neurons(self):
        return self[0]['firing_rates'].shape[1]

    def _process_data(self):
        print('Preparing dataset: Binning data.')
        # load data
        mat_dict = loadmat(self.raw_path)

        # bin data
        data = self._bin_data(mat_dict)

        self._save_processed_data(data)
        return data

    def _save_processed_data(self, data):
        with open(self.processed_path, 'wb') as output:
            pickle.dump({'data': data}, output)

    def _load_processed_data(self):
        with open(self.processed_path, "rb") as fp:
            data = pickle.load(fp)['data']
        return data

    def _bin_data(self, mat_dict):
        # load matrix
        trialtable = mat_dict['trial_table'] # (159, 13)
        neurons = mat_dict['out_struct']['units']
        pos = np.array(mat_dict['out_struct']['pos'])
        vel = np.array(mat_dict['out_struct']['vel'])
        acc = np.array(mat_dict['out_struct']['acc'])
        force = np.array(mat_dict['out_struct']['force'])
        time = vel[:, 0]

        num_neurons = len(neurons)
        num_trials = trialtable.shape[0]

        data = {'firing_rates': [], 'position': [], 'velocity': [], 'acceleration': [],
                'force': [], 'labels': [], 'sequence': [], 'position_raw': [], 'velocity_raw': []}
        for trial_id in tqdm(range(num_trials)):
            # assume that trials are all started on min_T.
            min_T = trialtable[trial_id, 9]
            max_T = trialtable[trial_id, 12]

            # grids= minT:(delT-TO):(maxT-delT);
            grid = np.arange(min_T, max_T + self.binning_period, self.binning_period)
            grids = grid[:-1]
            gride = grid[1:]
            num_bins = len(grids) # 9, 10, 11, etc

            neurons_binned = np.zeros((num_bins, num_neurons))
            pos_binned = np.zeros((num_bins, 2))
            vel_binned = np.zeros((num_bins, 2))
            acc_binned = np.zeros((num_bins, 2))
            force_binned = np.zeros((num_bins, 2))
            targets_binned = np.zeros((num_bins, 1))
            id_binned = trial_id * np.ones((num_bins, 1))

            all_mask = (time >= grids[0]) & (time <= gride[-1])
            if len(pos) > 0:
                pos_original = np.array(pos[all_mask, 1:]) # per trial
            else:
                pos_original = 'nan'
                print("len = 0 for position")

            vel_original = np.array(vel[all_mask, 1:])

            for k in range(num_bins):
                bin_mask = (time >= grids[k]) & (time <= gride[k])
                if len(pos) > 0:
                    pos_binned[k, :] = np.mean(pos[bin_mask, 1:], axis=0)
                vel_binned[k, :] = np.mean(vel[bin_mask, 1:], axis=0)
                if len(acc):
                    acc_binned[k, :] = np.mean(acc[bin_mask, 1:], axis=0)
                if len(force) > 0:
                    force_binned[k, :] = np.mean(force[bin_mask, 1:], axis=0)
                targets_binned[k, 0] = trialtable[trial_id, 1]

            for i in range(num_neurons):
                for k in range(num_bins):
                    spike_times = neurons[i]['ts'] # this is the recording time, where neuron fired
                    #####
                    bin_mask = (spike_times >= grids[k]) & (spike_times <= gride[k])
                    neurons_binned[k, i] = np.sum(bin_mask) / self.binning_period

            data['firing_rates'].append(neurons_binned)
            data['position'].append(pos_binned)
            data['velocity'].append(vel_binned)
            data['acceleration'].append(acc_binned)
            data['force'].append(force_binned)
            data['labels'].append(targets_binned)
            data['sequence'].append(id_binned)

            data['position_raw'].append(pos_original)
            data['velocity_raw'].append(vel_original)
        return data

    def _split_data(self, data):
        num_trials = len(data['firing_rates'])
        split_id = int(num_trials * self.train_split)

        data_train = {}
        data_test = {}
        for key, feature in data.items():
            # print(key, len(feature), (feature)[0])
            # firing_rates 159 (11, 174)
            # position (11, 2)
            # velocity (11, 2)
            # acceleration (11, 2)
            # force (11, 2)
            # labels (11, 1)
            # sequence (11, 1)
            data_train[key] = feature[:split_id]
            data_test[key] = feature[split_id:]
        return data_train, data_test

    def _remove_static_neurons(self, data_train, data_test):
        for i in range(len(data_train['firing_rates'])):
            data_train['firing_rates'][i] = data_train['firing_rates'][i][:, self.std > 1e-3]
        for i in range(len(data_test['firing_rates'])):
            data_test['firing_rates'][i] = data_test['firing_rates'][i][:, self.std > 1e-3]
        self.mean = self.mean[self.std > 1e-3]
        self.std = self.std[self.std > 1e-3]
        return data_train, data_test

    def _compute_mean_std(self, data, feature='firing_rates'):
        concatenated_data = np.concatenate(data[feature])
        mean = concatenated_data.mean(axis=0)
        std = concatenated_data.std(axis=0)
        return mean, std

    def _scale_data(self, data_train, data_test, feature):
        concatenated_data = np.concatenate(data_train[feature])
        mean = concatenated_data.mean(axis=0)
        std = concatenated_data.std(axis=0)

        for i in range(len(data_train[feature])):
            data_train[feature][i] = (data_train[feature][i] - mean) / std
        for i in range(len(data_test[feature])):
            data_test[feature][i] = (data_test[feature][i] - mean) / std
        return data_train, data_test

    def _sort_by_reach_direction(self, data):
        sorted_by_label = np.argsort(np.array([reach_dir[0, 0] for reach_dir in data['labels']]))
        for feature in data.keys():
            data[feature] = np.array(data[feature])[sorted_by_label]
        return data

    def _map_labels(self, data):
        labels = data['labels']
        for i, l in enumerate(np.unique(labels)):
            labels[data['labels']==l] = i
        return labels

    @staticmethod
    def _same_length_for_psth(original_data, original_label):
        new_data = []
        new_label = []

        # get minimum length
        length_list = []
        for data_i in original_data:
            length_list.append(data_i.shape[0])
        min_len = int(min(length_list))

        # loop through data and label to cut them
        for i, data in enumerate(original_data):
            data = data[:min_len, :]
            label = original_label[i][:min_len, :]

            new_data.append(data)
            new_label.append(label)

        # satck them
        new_data = np.stack(new_data, axis=0)
        new_label = np.stack(new_label, axis=0)
        return new_data, new_label

    @staticmethod
    def _draw_conditional_psth(processed_data, processed_label, neuron=0, direction=0):

        # select the condition (the label == direction)
        processed_label = processed_label[:, 0, 0]
        processed_data = processed_data[processed_label == direction, :, :]

        # select the neuron
        processed_data = processed_data[:, :, neuron]

        # sum the spikes
        psth = np.sum(processed_data, axis=0) / processed_data.shape[0]

        return psth, np.arange(psth.shape[0])

    @staticmethod
    def _draw_conditional_psth_all(processed_data, processed_label):

        psth_all = processed_data.copy() # trials, 9, neurons
        label_ids = processed_label[:, 0, 0]

        # loop through all labels
        for label in range(8):
            processed_data_i = processed_data[label_ids == label, :, :]
            len_i = processed_data_i.shape[0]
            # loop through all neurons
            for neuron in range(processed_data.shape[-1]):
                processed_data_i_j = processed_data_i[:, :, neuron]
                psth_i_j = np.sum(processed_data_i_j, axis=0) / processed_data_i_j.shape[0]

                check = False
                if check:
                    plt.plot(np.arange(psth_i_j.shape[0]), psth_i_j)
                    plt.plot(np.arange(psth_i_j.shape[0]), np.squeeze(processed_data_i_j[0, :]))
                    plt.show()
                    time.sleep(10)

                psth_i = np.stack([psth_i_j for numb in range(len_i)], axis=0)

                psth_all[label_ids == label, :, neuron] = psth_i

            # print('psth_all', psth_all)

        return psth_all

    @staticmethod
    def _plot_trajectory(position_raw, original_label):
        '''trajectory colored via label value'''

        cmap = matplotlib.cm.get_cmap('tab10')

        fig, ax = plt.subplots(nrows=1, ncols=1)

        for i, position in enumerate(position_raw):
            ax.plot(position[:, 0], position[:, 1], c=cmap(original_label[i][0, 0]))

        plt.show()



def get_class_data(dataset, device='cpu'):
    def get_data():
        firing_rates = dataset.firing_rates
        labels = dataset.labels
        data = [torch.tensor(firing_rates, dtype=torch.float32, device=device),
                      torch.tensor(labels, dtype=torch.long, device=device)]
        return data
    dataset.train()
    data_train = get_data()

    dataset.test()
    data_test = get_data()

    dataset.train()
    return data_train, data_test


def get_angular_data(dataset, velocity_threshold=-1., device='cpu'):
    def get_data():
        velocity_mask = np.linalg.norm(dataset.velocity, 2, axis=1) > velocity_threshold
        firing_rates = dataset.firing_rates[velocity_mask]
        labels = dataset.labels[velocity_mask]

        angles = (2 * np.pi / 8 * labels)[:, np.newaxis]
        cos_sin = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
        data = [torch.tensor(firing_rates, dtype=torch.float32, device=device),
                torch.tensor(angles, dtype=torch.float32, device=device),
                torch.tensor(cos_sin, dtype=torch.float32, device=device),
                torch.tensor(labels, dtype=torch.float32, device=device)]
        return data
    dataset.train()
    data_train = get_data()

    dataset.test()
    data_test = get_data()

    dataset.train()
    return data_train, data_test


class LocalGlobalGenerator(IterableDataset):
    r"""Pair generator for trial-based dataset.
    Args:
        features (numpy.ndarray): Feature matrix of size (num_samples, num_features).
        pair_set (numpy.ndarray): Matrix containing information about possible pairs of features. (N, 3) dimensions
            where the two first columns are the ids for the pair of neighbor features and third column is the id for
            the sequence they belong to.
        sequence_lengths (numpy.ndarray): Number of data points in each sequence. (num_of_sequences)
        batch_size (int): Batch size.
        pool_batch_size (int): Batch size of candidate samples.
        num_examples (int): Total number of examples.
        transform (Callable, Optional): Transformation.
        structure_transform (bool, Optional): If :obj:`True`, the same transformation to generate augmented views.
            (default: :obj:`False`)
        num_workers (int, Optional): Number of workers used in dataloader. (default: :obj:`1`)
    """
    def __init__(self, features, pair_set, sequence_lengths, batch_size, pool_batch_size, num_examples,
                 transform=None, structured_transform=False, num_workers=1):
        self.features = features
        self.pair_set = pair_set[:, 1:]
        self.sequence_ids = pair_set[:, 0]

        sequence = []
        for i, n in enumerate(sequence_lengths):
            sequence.append(np.ones(n) * i)
        self.sequence = np.concatenate(sequence).astype(int)

        self.num_trials = int(self.sequence_ids[-1]) + 1
        self.batch_size = batch_size
        self.pool_batch_size = pool_batch_size
        self.transform = transform
        self.structured_transform = structured_transform
        self.sequence_mask, self.random_mask = self._build_masks()
        self.num_workers = num_workers
        self.num_iterations = int(num_examples // batch_size)

    def _build_masks(self):
        pair_masks = {}
        random_masks = {}
        for i in range(self.num_trials):
            pair_masks[i] = self.sequence_ids == i
            random_masks[i] = self.sequence == i
        return pair_masks, random_masks

    def _sample_trials(self):
        permuted_trials = np.random.permutation(self.num_trials)
        # print("permuted_trials", permuted_trials)
        split_trials = int(self.num_trials / 2 + np.random.normal(self.num_trials / 10))
        # print("split_trials", split_trials)
        trials_1 = permuted_trials[:split_trials]
        trials_2 = permuted_trials[split_trials:]
        return trials_1, trials_2

    def _sample_pairs(self):
        sample_indices = self.pair_set[np.random.choice(self.pair_set.shape[0], self.batch_size, replace=False)].T
        x1, x2 = self.features[sample_indices, :]
        dt = sample_indices[1] - sample_indices[0]
        return x1, x2, dt

    def _sample_random(self, from_trials, batch_size):
        mask = np.sum([self.random_mask[i] for i in from_trials], axis=0).astype(bool)
        # print('mask', mask.shape)
        if sum(mask) >= batch_size:
            sample_indices = np.random.choice(sum(mask), batch_size, replace=False)
        else:
            sample_indices = np.concatenate([np.arange(sum(mask)),
                                             np.random.choice(sum(mask), batch_size-sum(mask), replace=False)])
        # print('sample_indices', sample_indices.shape)
        x = self.features[mask][sample_indices]
        # print(x.shape)
        return x

    def __iter__(self):
        for _ in range(self.num_iterations):
            trials_1, trials_2 = self._sample_trials()
            # print(trials_1.shape, trials_2.shape)

            x1, x2, dt = self._sample_pairs()
            x3 = self._sample_random(trials_1, self.batch_size)
            x4 = self._sample_random(trials_2, self.pool_batch_size)

            x1 = torch.tensor(x1)
            x2 = torch.tensor(x2)
            x3 = torch.tensor(x3)
            x4 = torch.tensor(x4)

            if self.transform is not None:
                if self.structured_transform:
                    x1, x2 = self.transform(x1, x2)
                else:
                    [x1] = self.transform(x1)
                    [x2] = self.transform(x2)
                [x3] = self.transform(x3)
                [x4] = self.transform(x4)
            yield x1.type(torch.float32), x2.type(torch.float32), \
                  x3.type(torch.float32), x4.type(torch.float32)

    def __len__(self):
        return self.num_iterations * self.batch_size * self.num_workers

    @staticmethod
    def prepare_views(inputs):
        x1, x2, x3, x4 = inputs
        outputs = {'view1': x1.squeeze(), 'view2': x2.squeeze(),
                   'view3': x3.squeeze(), 'view_pool': x4.squeeze()}
        return outputs
