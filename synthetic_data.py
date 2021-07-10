import numpy as np
import torch

def simulate_data(length, n_cls, n_dim):
    ## simulate 2d z
    np.random.seed(888)
    mu_true = np.random.uniform(-5, 5, [2, n_cls])
    var_true = np.random.uniform(0.5, 3, [2, n_cls])

    u_true = np.array(np.tile(np.arange(n_cls), int(length / n_cls)), dtype='int')
    z_true = np.vstack((np.random.normal(mu_true[0][u_true], np.sqrt(var_true[0][u_true])),
                        np.random.normal(mu_true[1][u_true], np.sqrt(var_true[1][u_true])))).T

    z_true = np.hstack((z_true, np.zeros((z_true.shape[0], n_dim - 2))))

    ## simulate mean
    dim_x = z_true.shape[-1]
    permute_ind = []
    n_blk = 4
    for ii in range(n_blk):
        np.random.seed(ii)
        permute_ind.append(torch.Tensor(np.random.permutation(dim_x)))

    x_input = layers.Input(shape=(dim_x,))
    x_output = realnvp_block(x_input)
    for ii in range(n_blk - 1):
        x_output = Lambda(perm_func, arguments={'ind': permute_ind[ii]})(x_output)
        x_output = realnvp_block(x_output)

    realnvp_model = Model(inputs=[x_input], outputs=x_output)
    mean_true = realnvp_model.predict(z_true)
    lam_true = np.exp(2 * np.tanh(mean_true))
    return z_true, u_true, mean_true, lam_true

def synthtic():
    length = 10000
    n_cls = 5
    n_dim = 100

    z_true, u_true, mean_true, lam_true = simulate_data(length, n_cls, n_dim)
    np.random.seed(777)
    x_true = np.random.poisson(lam_true)