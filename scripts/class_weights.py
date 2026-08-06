"""
Mrinal Kanti Dhar
13 Aug 2025

"""

import pandas as pd
import numpy as np

#%%
def inv_freq_class_weights(
    df: pd.DataFrame,
    columns: list,
    scale_down_factor: float = 10.0,
    verbose: bool = False
) -> dict:
    """
    It computes per-class weights for imbalanced classification, 
    using an inverse-frequency formula.

    weight_c = n_total / (scale_down_factor * pos_c)

    If pos_c is 0, a warning is shown and weight is set to np.inf.
    """

    n_total = len(df)
    if verbose: print('Total samples:', n_total)

    class_weights = {}
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df")

        pos = int(df[col].sum())
        if verbose: print(f"{col}: positives={pos}")

        if pos == 0:
            print(f"Warning: No positive samples found for class '{col}'. "
                  "Weight set to infinity.")
            pos_weight = np.inf
        else:
            pos_weight = n_total / (scale_down_factor * pos)

        class_weights[col] = pos_weight

    return class_weights

#%%
def pos_weights_bce(
    df: pd.DataFrame,
    columns: list,
    alpha: float = 1.0, # typically, 1 to 10
    clip_max: float | None = None,
    verbose: bool = False,
    ) -> dict:
    """
    Compute per-class pos_weight for BCEWithLogitsLoss with Laplace smoothing.

    Summary line.

    :param df: (pd.DataFrame) DataFrame containing binary label columns.
    :param columns: (list) List of label column names in the desired order.
    :param alpha: (float) Laplace smoothing; pos_weight_c = (N - P_c + alpha) / (P_c + alpha). Choose between 1 to 10.
    :param clip_max: (float|None) If set, clip each pos_weight to this maximum.   
    :param verbose: (bool) If True, print per-class positive counts and weights.
    :return class_weights: (dict) Mapping {class_name: pos_weight}.
    """
    n_total = len(df)
    if verbose: print('Total samples:', n_total)

    class_weights = {}
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df")

        pos = int(df[col].sum())
        if verbose: print(f"{col}: positives={pos}")

        if pos == 0:
            print(f"Warning: No positive samples for class '{col}'. "
                  "Weight set to infinity.")
            pos_weight = np.inf
        else:
            pos_weight = (n_total - pos + alpha) / (pos + alpha) # BCE formula with Laplacian smoothing (alpha)

        if clip_max is not None:
                    pos_weight = float(min(pos_weight, clip_max))

        class_weights[col] = pos_weight

    return class_weights


if __name__ == "__main__":
    import os

    dir = '/research/m324371/Project/Digital_Twin/CT-CLIP/Dataframes/'
    filename= 'Dataset_CT_train_label.xlsx'

    df = pd.read_excel(os.path.join(dir, filename))

    # Get column names
    columns = df.columns[1:].tolist()

    "Uncomment to use inverse-frequence class weights"
    pos_weights = inv_freq_class_weights(df, 
                                    columns=columns, 
                                    scale_down_factor=10,
                                    verbose=True)

    "Uncomment to use bce class weights"
    # pos_weights = pos_weights_bce(df, 
    #                           columns=columns,  
    #                           alpha=10.0, 
    #                           clip_max=50,
    #                           verbose=True)

    for k,v in pos_weights.items():
        print(f"{k}: {v}")

    # Print only weights as a list
    w = []
    for k,v in pos_weights.items():
         w.append(round(v,4))
    print("\n\nWeights in list:\n", w)







