# coding=utf-8
import tqdm
import numpy as np
import numpy.random as nr
import warnings
import time
import os


class GaussianBernoulliRBM(object):
    """
    Gaussian Bernoulli Restricted Boltzmann Machine
    Visible units and hidden units are modeled with Gaussian and Bernoulli distribution, respectively.
    Therefore, this machine can be applied into real-value data without transforming the data into binary data.
    """

    def __init__(self, num_v_unit, num_h_unit):
        """
        :param num_v_unit: number of visible units
        :param num_h_unit: number of hidden units
        """
        self.num_v_unit = num_v_unit
        self.num_h_unit = num_h_unit

        # Initialization
        # W: weight (dim num hidden * num visible)
        # B: bias of visible units(dim 1 * num visible)
        # C: bias of hidden units(dim 1 * num hidden)
        # Sigma: numpy array (dim 1 * visible units)
        self.W = nr.rand(self.num_h_unit, self.num_v_unit)
        self.B = nr.rand(1, self.num_v_unit)
        self.C = nr.rand(1, self.num_h_unit)
        self.Sigma = np.ones((1, self.num_v_unit))

    def learning(self, train_data, max_epoch, mini_batch_size=10, sampling_type="CD", sampling_times=1, Sigma_fix=True,
                 learning_rate=0.01, momentum_rate=0, weight_decay_rate=0, sparse_regularization_target=0,
                 sparse_regularization_rate=0):
        """

        :param train_data:
        :param max_epoch:
        :param mini_batch_size:
        :param sampling_type:
        :param sampling_times:
        :param Sigma_fix:
        :param learning_rate:
        :param momentum_rate:
        :param weight_decay_rate:
        :param sparse_regularization_target:
        :param sparse_regularization_rate:
        :return:
        """

        # Model Parameters
        W = self.W
        B = self.B
        C = self.C
        Sigma = self.Sigma

        # For Momentum
        delta_W = np.zeros((self.num_h_unit, self.num_v_unit))
        delta_B = np.zeros((1, self.num_v_unit))
        delta_C = np.zeros((1, self.num_h_unit))
        delta_Sigma = np.zeros((1, self.num_v_unit))

        # For Sparse Regularization
        rho_new = 0

        # train_data is supposed to be 2-d numpy array
        # this method below will transform the array into 3-d numpy array
        mini_batch = make_mini_batch(train_data, mini_batch_size)

        len_mini_batch = mini_batch.shape[0]

        save_mini_batch_for_low_memory(mini_batch, "MB")

        del mini_batch

        learning_params = {"sampling_type": sampling_type,
                           "sampling_times": sampling_times,
                           "learning_rate": learning_rate,
                           "momentum_rate": momentum_rate,
                           "weight_decay_rate": weight_decay_rate,
                           "sparse_regularization_target": sparse_regularization_target,
                           "sparse_regularization_rate": sparse_regularization_rate,
                           "Sigma_fix": Sigma_fix}

        # no need for X_k to be equal to the fist mini_batch
        # this is just for an initialization
        X_k = np.array(mini_batch[0])

        # ENTRY of Contrastive Divergence Learning
        for e in tqdm.tqdm(range(0, max_epoch)):
            # Gibbs Sampling

            # X = np.array(mini_batch[int(e % len(mini_batch))])

            X = numpy_array_load(os.path.join("MB", str(e % len_mini_batch)))

            X_k = gibbs_sampling(X, X_k, W, B, C, Sigma,
                                 learning_params["sampling_type"],
                                 learning_params["sampling_times"])

            # Gradient Update
            X, X_k, W, B, C, Sigma, delta_W, delta_B, delta_C, delta_Sigma, rho_new = gradient_update(X, X_k, W, B, C,
                                                                                                      Sigma,
                                                                                                      delta_W, delta_B,
                                                                                                      delta_C,
                                                                                                      delta_Sigma,
                                                                                                      rho_new,
                                                                                                      learning_params)
            # Non Negative Limitation
            W[W < 0] = 0
        # END of Contrastive Divergence Learning

        # Learned Model Parameters
        self.W = W
        self.B = B
        self.C = C
        self.Sigma = Sigma

    def set_model_params(self, W, B, C, Sigma):
        """

        :param W:
        :param B:
        :param C:
        :param Sigma:
        :return:
        """

        if (W.shape == (self.num_h_unit, self.num_v_unit)
                and B.shape == (1, self.num_v_unit)
                and C.shape == (1, self.num_h_unit)
                and Sigma.shape == (1, self.num_v_unit)):
            self.W = W
            self.B = B
            self.C = C
            self.Sigma = Sigma

        else:
            raise ArrNotMatchException("raise")

    def get_model_params(self):
        """

        :return:
        """
        return self.W, self.B, self.C, self.Sigma

    def get_W(self):
        """

        :return:
        """
        return self.W


def make_mini_batch(data, mini_batch_size):
    """
    makes mini batch from 2-d numpy array
    :param data: 2-d numpy array
    :param mini_batch_size:
    :return: 3-d numpy array (total num of mini_batch * mini_batch_size, original_data_column)
    """
    r, c = data.shape

    rest = r % mini_batch_size

    round_data, rest_data = np.split(data, [r - rest])

    round_data.reshape((int(round_data.shape[0] / mini_batch_size), mini_batch_size, c))

    rest_data = np.vstack([rest_data, data[0:int(mini_batch_size - rest)]])

    data = np.vstack([round_data, rest_data])

    return data.reshape((int(data.shape[0] / mini_batch_size), mini_batch_size, c))


def save_mini_batch_for_low_memory(mini_batch, path):
    """

    :param mini_batch:
    :param path:
    :return:
    """

    for i, m_b in enumerate(mini_batch):
        numpy_array_save(os.path.join(path, str(i)), m_b)


def numpy_array_save(filename, array):
    """
    Saves numpy array into a directory.
    :param filename: filename
    :param array: array to be saved
    :return: None
    """

    np.save('%s.npy' % filename, array)


def numpy_array_load(filename):
    """

    :param filename:
    :return:
    """
    return np.load(filename)


def gradient_update(X, X_k, W, B, C, Sigma, delta_W, delta_B, delta_C, delta_Sigma, rho, learning_params):
    """

    :param X:
    :param X_k:
    :param W:
    :param B:
    :param C:
    :param Sigma:
    :param delta_W:
    :param delta_B:
    :param delta_C:
    :param delta_Sigma:
    :param rho_new:
    :param learning_params:
    :return:
    """
    learning_rate = learning_params["learning_rate"]
    sparse_regularization_target = learning_params["sparse_regularization_target"]
    sparse_regularization_rate = learning_params["sparse_regularization_rate"]
    momentum_rate = learning_params["momentum_rate"]
    weight_decay_rate = learning_params["weight_decay_rate"]
    Sigma_fix = learning_params["Sigma_fix"]

    P_H_1_X = prob_H_1_X(X, W, C, Sigma)
    P_H_1_X_k = prob_H_1_X(X_k, W, C, Sigma)

    rho_old = rho
    W_old = W
    B_old = B
    C_old = C
    Sigma_old = Sigma

    rho, grad_E_sparse_W, grad_E_sparse_C = sparse_regularization(X,
                                                                  W_old, C_old,
                                                                  Sigma_old, rho_old, sparse_regularization_target)
    start = time.time()
    C = C_old + learning_rate * (
            CD_C(P_H_1_X, P_H_1_X_k) - sparse_regularization_rate * grad_E_sparse_C) + momentum_rate * delta_C
    print("C:  %f s \n" % (time.time() - start))

    start = time.time()
    B = B_old + learning_rate * CD_B(X, X_k, Sigma_old) + momentum_rate * delta_B

    print("B:  %f s \n" % (time.time() - start))

    start = time.time()
    W = W_old + learning_rate * (CD_W(X, X_k, P_H_1_X, P_H_1_X_k,
                                      Sigma_old) - weight_decay_rate * W_old - sparse_regularization_rate * grad_E_sparse_W) + momentum_rate * delta_W
    print("W:  %f s \n" % (time.time() - start))

    start = time.time()
    if Sigma_fix:
        Sigma = Sigma_old
    else:
        Sigma = Sigma_old + learning_rate * (CD_Sigma(X, X_k, P_H_1_X, P_H_1_X_k, W_old, B_old,
                                                      Sigma_old) - weight_decay_rate * Sigma_old) + momentum_rate * delta_Sigma

    print("Sigma:  %f s \n" % (time.time() - start))
    delta_W = W - W_old
    delta_B = B - B_old
    delta_C = C - C_old
    delta_Sigma = Sigma - Sigma_old

    return X, X_k, W, B, C, Sigma, delta_W, delta_B, delta_C, delta_Sigma, rho


def gibbs_sampling(X, X_k, W, B, C, Sigma, sampling_type, sampling_times):
    """

    :param X:
    :param X_k:
    :param W:
    :param B:
    :param C:
    :param Sigma:
    :param sampling_type:
    :param sampling_times:
    :return:
    """

    if sampling_type == 'PCD':
        X_k = block_gibbs_sampling(X, W, B, C, Sigma, sampling_times)

    elif sampling_type == 'CD':
        X_k = block_gibbs_sampling(X_k, W, B, C, Sigma, sampling_times)

    return X_k


def sparse_regularization(X, W, C, sigma, rho_old, sparse_regularization_target):
    """

    :param X:
    :param W:
    :param C:
    :param sigma:
    :param rho_old:
    :param sparse_regularization_target:
    :return:
    """

    N = X.shape[0]

    # dim: 1 * num_hidden_units
    rho_new = 0.9 * rho_old + 0.1 * np.sum(prob_H_1_X(X, W, C, sigma), axis=0) / N

    delta_E_sparse_C = (-sparse_regularization_target / rho_new + (1 - sparse_regularization_target) / (
            1 - rho_new)) / N
    delta_E_sparse_C = delta_E_sparse_C[np.newaxis, :]

    S = np.empty((X.shape[0], delta_E_sparse_C.shape[1], X.shape[1]))
    for index, x in enumerate(X):
        S[index, :, :] = np.dot(delta_E_sparse_C.T, x[np.newaxis, :])

    delta_E_sparse_W = np.sum(S, axis=0) / N

    return rho_new, delta_E_sparse_W, delta_E_sparse_C


def block_gibbs_sampling(X, W, B, C, Sigma, sampling_times):
    """
    Block Gibbs Sampling
    :param X: values of visible (dim: num data * num visible units)
    :param C: biases of hidden units(dim 1 * num hidden)
    :param B: biases of visible units(dim 1 * num visible)
    :param W: weight (dim num hidden * num visible)
    :param sigma: scalar or numpy array (dim 1 * visible units)
    :return: sampled and averaged visible values X
    """

    temp = np.zeros((X.shape[0], X.shape[1]))
    X_k = X
    for _ in range(0, sampling_times):
        H_k_1_X = prob_H_1_X(X_k, W, C, Sigma)
        H_k = sampling_H_X(H_k_1_X)
        X_k = sampling_X_H(H_k, W, B, Sigma)
        temp += X_k

    return temp / sampling_times


def prob_H_1_X(X, W, C, Sigma):
    """
    A row is a vector where i-th is the probability of h_i becoming 1 when given X
    :param X: values of visible (dim: num data * num visible units)
    :param W: weight (dim num hidden * num visible)
    :param C: biases of hidden units(dim 1 * num hidden)
    :param Sigma: scalar or numpy array (dim 1 * visible units)
    :return: numpy array (dim: num data * num hidden)
    """

    warnings.filterwarnings('error')
    try:

        return 1 / (1 + np.exp(-C - (np.dot(X / (Sigma * Sigma), np.transpose(W)))))

    except RuntimeWarning as warn:

        # Over float is interpreted as RuntimeWarning.
        # An array filled with 0 will be returned instead of the array with over floated number.
        return np.zeros((X.shape[0], W.shape[0]))


def sampling_H_X(P_H_1):
    """
    Gets samples of H following Bernoulli distribution when given X
    :param P_H_1: probability of H becoming 1 when given X
    :return: array (dim: num_data*num_hidden_units)
    """

    return np.fmax(np.sign(P_H_1 - np.random.rand(P_H_1.shape[0], P_H_1.shape[1])),
                   np.zeros((P_H_1.shape[0], P_H_1.shape[1])))


def sampling_X_H(H, W, B, Sigma):
    """
    Gets samples of X following Gaussian distribution when given H
    :param H: values of hidden (dim: num data * num hidden)
    :param W: weight (dim num hidden * num visible)
    :param B: biases of visible (dim: num data * num visible)
    :param Sigma: scalar or numpy array (dim 1 * visible units)
    :return: numpy array (dim: num data * num visible)
    """

    return Sigma * np.random.randn(H.shape[0], W.shape[1]) + B + np.dot(H, W)


def CD_C(P_H_1_X, P_H_1_X_k):
    """
    Gradient approximation of C
    :param P_H_1_X: probability of H becoming 1 when given X
    :param P_H_1_X_k: probability of H becoming 1 when given X_k
    :return: numpy vector (dim: 1 * num_hidden_units)
    """

    return np.sum(P_H_1_X - P_H_1_X_k, axis=0) / P_H_1_X.shape[0]


def CD_B(X, X_k, Sigma):
    """
    Gradient approximation of B
    :param B: biases of visible (dim: num data * num visible)
    :param X_k: values of sampled visible (dim: num data * num visible units)
    :return: numpy vector (dim: 1 * num_visible_units)
    """

    return (np.sum(X - X_k, axis=0)) / (X.shape[0] * Sigma * Sigma)


def CD_W(X, X_k, P_H_1_X, P_H_1_X_k, Sigma):
    """
    Gradient approximation of W
    :param X: values of  visible (dim: num data * num visible units)
    :param X_k: values of sampled visible (dim: num data * num visible units)
    :param P_H_1_X: probability of H becoming 1 when given X
    :param P_H_1_X_k: probability of H becoming 1 when given X_k
    :return: numpy array(dim: num_hidden_units * num_visible_units)
    """

    # Numpy array was faster in some experiments.
    E = np.empty((X.shape[0], P_H_1_X.shape[1], X.shape[1]))

    for index, (P_x, x, P_x_k, x_k) in enumerate(zip((P_H_1_X),
                                                     (X),
                                                     (P_H_1_X_k),
                                                     (X_k))):
        E[index, :, :] = np.dot(P_x[:, np.newaxis], x[np.newaxis, :]) - np.dot(P_x_k[:, np.newaxis],
                                                                               x_k[np.newaxis, :])

    return np.sum(E, axis=0) / (X.shape[0] * Sigma * Sigma)


def CD_Sigma(X, X_k, P_H_1_X, P_H_1_X_k, W, B, Sigma):
    """
    Gradient approximation of sigma
    :param X: values of  visible (dim: num data * num visible units)
    :param X_k: values of sampled visible (dim: num data * num visible units)
    :param P_H_1_X: probability of H becoming 1 when given X
    :param P_H_1_X_k: probability of H becoming 1 when given X_k
    :param W: weight (dim num hidden * num visible)
    :param B: array (dim: num_data, num_visible_units)
    :param Sigma: scalar or numpy array (dim 1 * visible units)
    :return: numpy array (dim: 1)
    """

    E_1 = np.sum(np.diag(np.dot((X - B), np.transpose((X - B)))), axis=0) - 2 * np.sum(
        np.diag(np.dot(X, np.transpose(W)) * P_H_1_X))

    E_2 = (np.sum(np.diag(np.dot((X_k - B), np.transpose((X_k - B)))), axis=0) - 2 * np.sum(
        np.diag(np.dot(X_k, np.transpose(W)) * P_H_1_X_k)))

    return (E_1 - E_2) / (X.shape[0] * Sigma * Sigma * Sigma)


class ArrNotMatchException(Exception):
    def my_func(self):
        print("Array size does not match the vector size of h and v units")
