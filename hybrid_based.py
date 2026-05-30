X = np.load('.....npy') #Load data saved from other scripts
Y = np.laod('.....npy') #Downlaod from ABIDE and HCP dataset

# Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb

# HYDRA: Competing convolutional kernels for fast and accurate time series classification
# https://arxiv.org/abs/2203.13652

# ** EXPERIMENTAL **
# This is an *untested*, *experimental* extension of Hydra to multivariate input.

# todo: cleanup, documentation
import numpy as np
import random
import torch
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class HydraMultivariate(nn.Module):

    def __init__(self, input_length, num_channels, k = 16, g = 128, max_num_channels = 8):

        super().__init__()

        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        #self.dilations = 2 ** torch.arange(int(max_exponent) + 1)

        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        # if g > 1, assign: half the groups to X, half the groups to diff(X)
        divisor = 2 if self.g > 1 else 1
        _g = g // divisor
        self._g = _g

        self.W = [self.normalize(torch.randn(divisor, k * _g, 1, 9)) for _ in range(self.num_dilations)]

        # combine num_channels // 2 channels (2 < n < max_num_channels)
        num_channels_per = np.clip(num_channels // 2, 2, max_num_channels)
        self.I = [torch.randint(0, num_channels, (divisor, _g, num_channels_per)) for _ in range(self.num_dilations)]

    @staticmethod
    def normalize(W):
        W -= W.mean(-1, keepdims = True)
        W /= W.abs().sum(-1, keepdims = True)
        return W

    # transform in batches of *batch_size*
    def batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for i, batch in enumerate(batches):
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):

        num_examples = X.shape[0]

        if self.g > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            # diff_index == 0 -> X
            # diff_index == 1 -> diff(X)
            for diff_index in range(min(2, self.g)):

                _Z = F.conv1d(X[:, self.I[dilation_index][diff_index]].sum(2) if diff_index == 0 else diff_X[:, self.I[dilation_index][diff_index]].sum(2),
                              self.W[dilation_index][diff_index], dilation = d, padding = p,
                              groups = self._g) \
                      .view(num_examples, self._g, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self._g, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self._g, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z
    

    # Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://arxiv.org/abs/2203.13652

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class Hydra(nn.Module):

    def __init__(self, input_length, k = 8, g = 64, seed = None):

        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        self.W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 9)
        self.W = self.W - self.W.mean(-1, keepdims = True)
        self.W = self.W / self.W.abs().sum(-1, keepdims = True)

    # transform in batches of *batch_size*
    def batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):

        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):

                _Z = F.conv1d(X if diff_index == 0 else diff_X, self.W[dilation_index, diff_index], dilation = d, padding = p) \
                      .view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z

class SparseScaler():

    def __init__(self, mask = True, exponent = 4):

        self.mask = mask
        self.exponent = exponent

        self.fitted = False

    def fit(self, X):

        assert not self.fitted, "Already fitted."

        X = X.clamp(0).sqrt()

        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8

        self.mu = X.mean(0)
        self.sigma = X.std(0) + self.epsilon

        self.fitted = True

    def transform(self, X):

        assert self.fitted, "Not fitted."

        X = X.clamp(0).sqrt()

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from scipy.stats import pearsonr


# =====================================================
# FIXED SEED
# =====================================================
FIXED_SEED = 69472


def set_seed(seed=FIXED_SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(FIXED_SEED)


# =====================================================
# INPUTS
# =====================================================
# Temporal data
# X shape -> (N,200,T)

# Target
# Y shape -> (N,)

# CNN features
CNN = np.load(
    "cnn_similarity_features" #obtained from Similarity_based_prediction.py
)

print("Temporal shape:", X.shape)
print("CNN shape:", CNN.shape)
print("Y shape:", Y.shape)


# =====================================================
# 5-FOLD CV
# =====================================================
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=FIXED_SEED
)

all_pred = []
all_true = []

print("\nStarting 5-Fold CV...\n")

for fold, (train_idx, test_idx) in enumerate(
        kf.split(X)):

    print("=" * 60)
    print(f"Fold {fold+1}/5")
    print("=" * 60)

    # -------------------------------------------------
    # Split
    # -------------------------------------------------
    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = Y[train_idx]
    y_test = Y[test_idx]

    cnn_train = CNN[train_idx]
    cnn_test = CNN[test_idx]

    # -------------------------------------------------
    # Torch conversion
    # -------------------------------------------------
    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # -------------------------------------------------
    # HYDRA TRANSFORM
    # -------------------------------------------------
    transform = HydraMultivariate(
        X_train.shape[-1],
        116
    )

    X_train_h = transform(
        X_train.float()
    )

    X_test_h = transform(
        X_test.float()
    )

    print(
        "HYDRA features:",
        X_train_h.shape
    )

    # -------------------------------------------------
    # SCALING
    # -------------------------------------------------
    scaler = SparseScaler()

    X_train_h = scaler.fit_transform(
        X_train_h.float()
    )

    X_test_h = scaler.transform(
        X_test_h.float()
    )

    # -------------------------------------------------
    # CONCATENATE TEMPORAL + CNN
    # -------------------------------------------------
    TR_CON = np.concatenate(
        (X_train_h, cnn_train),
        axis=1
    )

    TE_CON = np.concatenate(
        (X_test_h, cnn_test),
        axis=1
    )

    print(
        "Combined feature shape:",
        TR_CON.shape
    )

    # -------------------------------------------------
    # RIDGE REGRESSION
    # -------------------------------------------------
    regressor = RidgeCV(
        alphas=np.logspace(-3, 3, 10)
    )

    regressor.fit(
        TR_CON,
        y_train
    )

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------
    pred = regressor.predict(
        TE_CON
    )

    fold_r, _ = pearsonr(
        pred,
        y_test
    )

    print(
        f"Fold Pearson r: "
        f"{fold_r:.4f}"
    )

    all_pred.extend(pred)
    all_true.extend(y_test)


# =====================================================
# FINAL RESULTS
# =====================================================
all_pred = np.array(all_pred)
all_true = np.array(all_true)

# -------------------------------------------------
# Pearson r
# -------------------------------------------------
r, _ = pearsonr(
    all_pred,
    all_true
)

# -------------------------------------------------
# MAE / RMSE
# -------------------------------------------------
mae = mean_absolute_error(
    all_true,
    all_pred
)

rmse = np.sqrt(
    mean_squared_error(
        all_true,
        all_pred
    )
)

# =====================================================
# CI FOR r (FISHER Z)
# =====================================================
n = len(all_true)

z = np.arctanh(r)

se = 1 / np.sqrt(n - 3)

z_low = z - 1.96 * se
z_high = z + 1.96 * se

r_low = np.tanh(z_low)
r_high = np.tanh(z_high)

# =====================================================
# BOOTSTRAP CI
# =====================================================
rng = np.random.default_rng(
    FIXED_SEED
)

n_boot = 1000

mae_scores = []
rmse_scores = []

for _ in range(n_boot):

    idx = rng.choice(
        len(all_true),
        len(all_true),
        replace=True
    )

    yt = all_true[idx]
    yp = all_pred[idx]

    mae_scores.append(
        mean_absolute_error(
            yt,
            yp
        )
    )

    rmse_scores.append(
        np.sqrt(
            mean_squared_error(
                yt,
                yp
            )
        )
    )

mae_ci = np.percentile(
    mae_scores,
    [2.5, 97.5]
)

rmse_ci = np.percentile(
    rmse_scores,
    [2.5, 97.5]
)

# =====================================================
# PERMUTATION TEST
# =====================================================
n_perm = 1000

perm_r = []

for _ in range(n_perm):

    shuffled = np.random.permutation(
        all_true
    )

    rr, _ = pearsonr(
        all_pred,
        shuffled
    )

    perm_r.append(rr)

perm_r = np.array(perm_r)

p_perm = np.mean(
    np.abs(perm_r)
    >= np.abs(r)
)

# =====================================================
# SAVE RESULTS
# =====================================================
np.save(
    "hybrid_predictions.npy",
    all_pred
)

np.save(
    "hybrid_groundtruth.npy",
    all_true
)

# =====================================================
# PRINT RESULTS
# =====================================================
print("\nResults")
print("=" * 60)

print(f"Pearson r : {r:.4f}")
print(
    f"95% CI    : "
    f"[{r_low:.4f}, {r_high:.4f}]"
)

print(f"MAE       : {mae:.4f}")
print(
    f"95% CI    : "
    f"[{mae_ci[0]:.4f}, "
    f"{mae_ci[1]:.4f}]"
)

print(f"RMSE      : {rmse:.4f}")
print(
    f"95% CI    : "
    f"[{rmse_ci[0]:.4f}, "
    f"{rmse_ci[1]:.4f}]"
)

print(
    f"Permutation p-value: "
    f"{p_perm:.6f}"
)

# =====================================================
# SCATTER PLOT
# =====================================================
sns.set_style("white")

plt.figure(figsize=(6, 6))

sns.regplot(
    x=all_pred,
    y=all_true,
    scatter_kws={
        'color': 'green',
        'alpha': 0.8,
        's': 40
    },
    line_kws={
        'color': 'black',
        'linewidth': 2
    },
    ci=95
)

plt.xlabel(
    "Predicted IQ score",
    fontsize=12,
    weight='bold'
)

plt.ylabel(
    "Actual IQ score",
    fontsize=12,
    weight='bold'
)

plt.title(
    "HYBRID MODEL",
    fontsize=14,
    weight='bold'
)

plt.text(
    0.95,
    0.05,
    f"r = {r:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    weight='bold',
    horizontalalignment='right'
)

plt.grid(False)

plt.tight_layout()

plt.savefig(
    "Hybrid_model_plot.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()