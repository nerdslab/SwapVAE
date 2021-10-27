"""this synthetic experiment file is adapted from Poisson Identifiable VAE (pi-VAE)
at https://github.com/zhd96/pi-vae. Pls refer to the original code if needed."""

import numpy as np
import torch

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from keras import backend as K


# util functions
def slice_func(x, start, size):
    return tf.slice(x, [0,start],[-1,size])

def perm_func(x, ind):
    # print(ind)
    ind = tf.dtypes.cast(ind, tf.int32)
    return tf.gather(x, indices=ind, axis=-1)

def realnvp_layer(x_input):
    DD = x_input.shape.as_list()[-1]  ## DD needs to be an even number
    dd = (DD // 2)

    ## define some lambda functions
    clamp_func = Lambda(lambda x: 0.1 * tf.tanh(x))
    trans_func = Lambda(lambda x: x[0] * tf.exp(x[1]) + x[2])
    sum_func = Lambda(lambda x: K.sum(-x, axis=-1, keepdims=True))

    ## compute output for s and t functions
    x_input1 = Lambda(slice_func, arguments={'start': 0, 'size': dd})(x_input)
    x_input2 = Lambda(slice_func, arguments={'start': dd, 'size': dd})(x_input)
    st_output = x_input1

    n_nodes = [dd // 2, dd // 2, DD]
    act_func = ['relu', 'relu', 'linear']
    for ii in range(len(act_func)):
        st_output = layers.Dense(n_nodes[ii], activation=act_func[ii])(st_output)
    s_output = Lambda(slice_func, arguments={'start': 0, 'size': dd})(st_output)
    t_output = Lambda(slice_func, arguments={'start': dd, 'size': dd})(st_output)
    s_output = clamp_func(s_output)  ## keep small values of s

    ## perform transformation
    trans_x = trans_func([x_input2, s_output, t_output])
    output = layers.concatenate([trans_x, x_input1], axis=-1)
    return output

def realnvp_block(x_output):
    for _ in range(2):
        x_output = realnvp_layer(x_output)
    return x_output


# simulate data
def simulate_cont_data_diff_var(length, n_dim, type=0):
    ## simulate 2d z
    np.random.seed(777+type)

    u_true = np.random.uniform(2 * np.pi/8 * type, 2 * np.pi/8 * (type+1), size=[length, 1])
    mu_true = np.hstack((5 * np.sin(u_true), 5 * np.cos(u_true)))
    # var_true = 0.03 * np.abs(mu_true)
    var_true = 0.10 * np.abs(mu_true)
    var_true[:, 0] = 0.6 - var_true[:, 1]
    z_true = np.random.normal(0, 1, size=[length, 2]) * np.sqrt(var_true) + mu_true
    z_true = np.hstack((z_true, np.zeros((z_true.shape[0], n_dim - 2))))

    ## simulate mean
    dim_x = z_true.shape[-1]
    permute_ind = []
    n_blk = 4
    for ii in range(n_blk):
        np.random.seed(ii)
        permute_ind.append(tf.convert_to_tensor(np.random.permutation(dim_x)))

    x_input = layers.Input(shape=(dim_x,))
    x_output = realnvp_block(x_input)
    for ii in range(n_blk - 1):
        x_output = Lambda(perm_func, arguments={'ind': permute_ind[ii]})(x_output)
        x_output = realnvp_block(x_output)

    realnvp_model = Model(inputs=[x_input], outputs=x_output)
    mean_true = realnvp_model.predict(z_true)
    lam_true = np.exp(2.2 * np.tanh(mean_true))
    return z_true, u_true, mean_true, lam_true


# store data to use later
def synthetic_data():
    length = 15000
    n_dim = 100
    z_true, u_true, mean_true, lam_true = simulate_cont_data_diff_var(length, n_dim)

    np.random.seed(777)
    x_true = np.random.poisson(lam_true)

    # import os
    # os.makedirs("./data/sim")
    np.savez('./data/sim/sim_100d_poisson_cont_label.npz', u=u_true, z=z_true, x=x_true, lam=lam_true, mean=mean_true)

# util functions for training SwapVAE
class sample_sequence():
    '''sample sequence based on existing data
    the goal is:
        return a `firing rate'
        return a `reaching direction label'
        return a `time label'
        return a `sequence length thing'
    '''
    def __init__(self, data, len=4):
        self.u = data['u'] # (20000*0.8, 1) (u: the label)
        self.z = data['z'] # (20000*0.8, 100)
        self.x = data['x'] # (20000*0.8, 100)

        self.len = len
        self.trial_id = 0

        self.firing_rates = []
        self.time_label = []
        self.direction_label = []
        self.sequence_length = []

        self.loop_through()

    def get_specific_direction(self, direction=0):
        '''direction = 0, 1, 2, 3'''
        return self.u[direction*4000:(direction+1)*4000], self.z[direction*4000:(direction+1)*4000], self.x[direction*4000:(direction+1)*4000]

    def sample_seq(self, u, z, x, len=4, direction=0):
        '''for example direction0 --> 2*np.pi/8 * 0, 2 * np.pi/8 * (0+1), actually direction is 0,2,4,6
        to sample the real sequence we would like 2*np.pi/8 / len as the division'''

        div = (2*np.pi/8)/len
        real_direction = direction*2
        base_direction = (2*np.pi/8 * real_direction) # beginning direction

        u_set = []
        z_set = []
        x_set = []
        len_set = []

        time_label = []

        for time in range(len):
            mask = (u >= base_direction + div * time) & (u < base_direction + div * (time+1))
            len_set.append(u[mask].shape[0])
        len_min = min(len_set)
        #print("len_min ", len_min)

        # prune additional ones
        for time in range(len):
            mask = (u >= base_direction + div * time) & (u < base_direction + div * (time + 1))
            mask = np.squeeze(mask)
            #print(u[mask].shape)
            u_set.append(u[mask, :][:len_min, :]) # for example (1000, 1)
            z_set.append(z[mask, :][:len_min, :])
            x_set.append(x[mask, :][:len_min, :]) # for example (1000, 100)
            time_label.append(np.ones((len_min, 1))*time)

        u_set = np.stack(u_set, axis=1) # (1000, 4, 1)
        z_set = np.stack(z_set, axis=1)
        x_set = np.stack(x_set, axis=1) # (1000, 4, 100) --> trials*time*firing_rate

        self.firing_rates.append(x_set)
        self.direction_label.append(np.ones((x_set.shape[0], x_set.shape[1], 1))*direction)
        self.time_label.append(np.stack(time_label, axis=1))
        for trial in range(len_min):
            self.sequence_length.append(len)
            self.trial_id += 1

    def loop_through(self):
        for direction in range(4):
            u,z,x = self.get_specific_direction(direction=direction)
            self.sample_seq(u,z,x, len=self.len, direction=direction)

        self.firing_rates = np.concatenate(self.firing_rates)
        self.time_label = np.concatenate(self.time_label)
        self.direction_label = np.concatenate(self.direction_label)

        print(self.firing_rates.shape, self.time_label.shape, self.direction_label.shape)

    def sample_portion(self, numb=1000):
        perm = torch.randperm(self.firing_rates.shape[0])
        idx = perm[:numb]
        return self.firing_rates[idx], self.direction_label[idx], self.time_label[idx]
        # samples = tensor[idx]

def get_angular_data_synthetic(data_train, data_test, device='cpu'):
    def get_data(loader):

        firing_rates = torch.flatten(torch.Tensor(loader.firing_rates), start_dim=0, end_dim=1).numpy()
        labels = torch.flatten(torch.Tensor(loader.direction_label), start_dim=0, end_dim=1).numpy()

        angles = (2 * np.pi / 4 * labels)[:, np.newaxis]
        cos_sin = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
        data = [torch.tensor(firing_rates, dtype=torch.float32, device=device),
                torch.tensor(angles, dtype=torch.float32, device=device),
                torch.tensor(cos_sin, dtype=torch.float32, device=device),
                torch.tensor(labels, dtype=torch.float32, device=device)]
        return data
    data_train__ = get_data(data_train)

    data_test__ = get_data(data_test)

    return data_train__, data_test__
