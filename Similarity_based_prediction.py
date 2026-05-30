import numpy as np
from scipy.signal import coherence, hilbert
from tslearn.metrics import dtw, lcss
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Input
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from scipy.stats import pearsonr
# ==========================================================
# Similarity Measure Function
# ==========================================================
def generate_similarity_matrices(x):   #Use this function to generate similarity measures
    """
    Parameters
    ----------
    x : ndarray
        Shape = (N_subjects, N_ROIs, Timepoints)

    Returns
    -------
    correlation : (N, R, R)
    coherence_m : (N, R, R)
    plv_m       : (N, R, R)
    pli_m       : (N, R, R)
    lcss_m      : (N, R, R)
    dtw_m       : (N, R, R)
    """

    n_subjects, n_rois, _ = x.shape

    # Initialize matrices
    correlation = np.zeros((n_subjects, n_rois, n_rois))
    coherence_m = np.zeros((n_subjects, n_rois, n_rois))
    plv_m = np.zeros((n_subjects, n_rois, n_rois))
    pli_m = np.zeros((n_subjects, n_rois, n_rois))
    lcss_m = np.zeros((n_subjects, n_rois, n_rois))
    dtw_m = np.zeros((n_subjects, n_rois, n_rois))

    # ------------------------------------------------------
    # Loop through subjects
    # ------------------------------------------------------
    for s in range(n_subjects):

        print(f"Processing Subject {s+1}/{n_subjects}")

        subj = x[s]  # (R, T)

        # Correlation (fast)
        correlation[s] = np.corrcoef(subj)

        # Pairwise measures
        for i in range(n_rois):

            signal_i = subj[i]

            for j in range(i, n_rois):

                signal_j = subj[j]

                # ==========================================
                # Coherence
                # ==========================================
                _, coh = coherence(signal_i, signal_j)
                coh_val = np.mean(coh)

                coherence_m[s, i, j] = coh_val
                coherence_m[s, j, i] = coh_val

                # ==========================================
                # PLV + PLI
                # ==========================================
                phase_i = np.angle(hilbert(signal_i))
                phase_j = np.angle(hilbert(signal_j))

                phase_diff = phase_i - phase_j

                plv = np.abs(
                    np.mean(np.exp(1j * phase_diff))
                )

                pli = np.abs(
                    np.mean(np.sign(phase_diff))
                )

                plv_m[s, i, j] = plv
                plv_m[s, j, i] = plv

                pli_m[s, i, j] = pli
                pli_m[s, j, i] = pli

                # ==========================================
                # DTW
                # ==========================================
                dtw_val = dtw(signal_i, signal_j)

                dtw_m[s, i, j] = dtw_val
                dtw_m[s, j, i] = dtw_val

                # ==========================================
                # LCSS
                # ==========================================
                lcss_val = lcss(signal_i, signal_j)

                lcss_m[s, i, j] = lcss_val
                lcss_m[s, j, i] = lcss_val

    return (
        correlation,
        coherence_m,
        plv_m,
        pli_m,
        lcss_m,
        dtw_m
    )
#Similarity-based features
# x = np.load('your_data.npy') Use your downlaoded data either HCP, ABIDE or anyother
(
    correlation,
    coherence_m,
    plv_m,
    pli_m,
    lcss_m,
    dtw_m
) = generate_similarity_matrices(x)

np.save("correlation.npy", correlation)
np.save("coherence.npy", coherence_m)
np.save("PLV.npy", plv_m)
np.save("PLI.npy", pli_m)
np.save("LCSS.npy", lcss_m)
np.save("DTW.npy", dtw_m)




A = np.load(r'correlation.npy')
B = np.load(r'coherence.npy')
C = np.load(r'PLV.npy')
D = np.load(r'PLI.npy')
E = np.load(r'LCSS.npy')
F = np.load(r'DTW.npy')

print(A.shape,B.shape,C.shape,D.shape,E.shape,F.shape)

# Step 1: Concatenate A and B horizontally to form the top half
top_half = np.concatenate((A, B), axis=2)

# Step 2: Concatenate C and D horizontally to form the bottom half
bottom_half = np.concatenate((C, D), axis=2)

# Step 3: Concatenate the top and bottom halves vertically to form the final square matrix
final_matrix = np.concatenate((top_half, bottom_half), axis=1)

# Step 4: Split the final matrix into top and bottom rectangular matrices
top_rectangular = final_matrix[:, :200, :]
bottom_rectangular = final_matrix[:, 200:, :]

# Print shapes to verify
print("Final square matrix shape:", final_matrix.shape)
print("Top rectangular matrix shape:", top_rectangular.shape)
print("Bottom rectangular matrix shape:", bottom_rectangular.shape)

import numpy as np

# Step 1: Concatenate A and B horizontally to form the top half
top_half1 = np.concatenate((C, D), axis=2)

# Step 2: Concatenate C and D horizontally to form the bottom half
bottom_half1 = np.concatenate((E, F), axis=2)

# Step 3: Concatenate the top and bottom halves vertically to form the final square matrix
final_matrix1 = np.concatenate((top_half1, bottom_half1), axis=1)

# Step 4: Split the final matrix into top and bottom rectangular matrices
top_rectangular1 = final_matrix1[:, :200, :]
bottom_rectangular1 = final_matrix1[:, 200:, :]

# Print shapes to verify
print("Final square matrix shape:", final_matrix1.shape)
print("Top rectangular matrix shape:", top_rectangular1.shape)
print("Bottom rectangular matrix shape:", bottom_rectangular1.shape)



# ==========================================================
# ADD CHANNEL DIMENSION
# ==========================================================
bottom_rectangular = bottom_rectangular1[..., np.newaxis]

print("CNN Input Shape:", bottom_rectangular.shape)


# ==========================================================
# CNN MODEL FUNCTION
# ==========================================================
def build_cnn():

    model = Sequential([

        Conv2D(
            128,
            (3, 3),
            activation='relu',
            input_shape=(200, 400, 1)
        ),

        MaxPooling2D((2, 2)),

        Conv2D(
            64,
            (3, 3),
            activation='relu'
        ),

        MaxPooling2D((2, 2)),

        Conv2D(
            32,
            (3, 3),
            activation='relu'
        ),

        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(
            32,
            activation='relu'
        ),

        Dense(1)

    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    return model

all_features = np.zeros((len(y1), 32))
# ==========================================================
# 5-FOLD CROSS VALIDATION
# ==========================================================
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

all_pred = []
all_true = []


all_features = []
print("\nStarting 5-Fold CV...\n")

for fold, (train_idx, test_idx) in enumerate(
        kf.split(bottom_rectangular)):

    print("=" * 60)
    print(f"Fold {fold+1}/5")
    print("=" * 60)

    # Split
    X_train = bottom_rectangular[train_idx]
    X_test = bottom_rectangular[test_idx]

    y_train = y1[train_idx]
    y_test = y1[test_idx]

    print(X_train.shape, X_test.shape)

    # Build model fresh each fold
    model = build_cnn()

    # Train
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=16,
        verbose=1
    )

    # ==========================================================
    # FEATURE EXTRACTOR (Dense(32) layer)
    # ==========================================================
    feature_model = Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )

    # CNN Features
    cnn_features = feature_model.predict(
        X_test,
        verbose=0
    )

    # shape -> (n_test_fold, 32)
    all_features[test_idx] = cnn_features

    # ==========================================================
    # Prediction
    # ==========================================================
    pred = model.predict(
        X_test,
        verbose=0
    ).ravel()

    # Store predictions
    all_pred.extend(pred)
    all_true.extend(y_test)

    # Fold correlation
    fold_r, _ = pearsonr(pred, y_test)

    print(f"Fold r: {fold_r:.4f}")


# ==========================================================
# FINAL RESULTS
# ==========================================================
all_pred = np.array(all_pred)
all_true = np.array(all_true)

print(
    "CNN feature shape:",
    all_features.shape
)

# Save CNN features
np.save(
    "cnn_similarity_features.npy",
    all_features
)

print(
    "Saved CNN features -> "
    "cnn_similarity_features.npy"
)
r, _ = pearsonr(all_pred, all_true)

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


# ==========================================================
# 95% CI FOR r (Fisher z)
# ==========================================================
n = len(all_true)

z = np.arctanh(r)
se = 1 / np.sqrt(n - 3)

z_low = z - 1.96 * se
z_high = z + 1.96 * se

r_low = np.tanh(z_low)
r_high = np.tanh(z_high)


# ==========================================================
# BOOTSTRAP CI FOR MAE/RMSE
# ==========================================================
n_boot = 1000

mae_scores = []
rmse_scores = []

rng = np.random.default_rng(42)

for _ in range(n_boot):

    idx = rng.choice(
        len(all_true),
        len(all_true),
        replace=True
    )

    yt = all_true[idx]
    yp = all_pred[idx]

    mae_scores.append(
        mean_absolute_error(yt, yp)
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


# ==========================================================
# PERMUTATION TEST
# ==========================================================
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
    perm_r >= r
)


# ==========================================================
# PRINT RESULTS
# ==========================================================
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
    f"[{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]"
)

print(f"RMSE      : {rmse:.4f}")
print(
    f"95% CI    : "
    f"[{rmse_ci[0]:.4f}, {rmse_ci[1]:.4f}]"
)

print(
    f"Permutation p-value: "
    f"{p_perm:.6f}"
)


# ==========================================================
# SCATTER PLOT
# ==========================================================
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
    "SIMILARITY-BASED MODEL",
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
    "Similarity_model_plot.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()