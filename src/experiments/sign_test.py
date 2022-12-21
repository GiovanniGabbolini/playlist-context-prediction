import os
import json
import argparse
import numpy as np
from scipy.stats import ttest_rel
from src.metrics import _map, _mrr, _fh
from src.consts import preprocessed_dataset_path


parser = argparse.ArgumentParser(description="What do you want to compare? Put the most performing algo as --what_1.")
parser.add_argument("--what_1", type=str, default="mf_seq")
parser.add_argument("--what_2", type=str, default="mf_avg")
args = parser.parse_args()

assert args.what_1 in ["mf_avg", "mf_seq", "kg_avg", "kg_seq", "kg_int_only_avg",
                       "kg_int_only_seq", "audio_avg", "audio_seq", "hybrid_avg", "hybrid_seq"]

assert args.what_2 in ["mf_avg", "mf_seq", "kg_avg", "kg_seq", "kg_int_only_avg",
                       "kg_int_only_seq", "audio_avg", "audio_seq", "hybrid_avg", "hybrid_seq"]

predictions_1 = np.load(f"{preprocessed_dataset_path}/results/{args.what_1}/best_run/predictions.npy")
predictions_2 = np.load(f"{preprocessed_dataset_path}/results/{args.what_2}/best_run/predictions.npy")
labels = np.load(f"{preprocessed_dataset_path}/results/{args.what_1}/best_run/ys.npy")

# Sign test MAP
map_1 = _map(predictions_1, labels, 5, False)
map_2 = _map(predictions_2, labels, 5, False)
_, p = ttest_rel(map_1, map_2)
print(f"Sign test MAP@5. P-value: {p}")

# Sign test MRR
mrr_1 = _mrr(predictions_1, labels, False)
mrr_2 = _mrr(predictions_2, labels, False)
_, p = ttest_rel(mrr_1, mrr_2)
print(f"Sign test MRR. P-value: {p}")

# Sign test FH
for k in [1, 5]:

    metric_values_1, metric_values_2 = [], []
    for _ in range(1000):
        indices = np.random.choice(len(predictions_1), len(predictions_1))

        bootstrap_replica_1 = predictions_1[indices]
        bootstrap_replica_2 = predictions_2[indices]
        bootstrap_replica_true = labels[indices]

        metric_1 = _fh(bootstrap_replica_1, bootstrap_replica_true, k)
        metric_2 = _fh(bootstrap_replica_2, bootstrap_replica_true, k)

        metric_values_1.append(metric_1)
        metric_values_2.append(metric_2)

    metric_values_1 = np.array(metric_values_1)
    metric_values_2 = np.array(metric_values_2)

    p = 1 - (np.sum(metric_values_1 > metric_values_2) / 1000)
    print(f"Sign test FH@{k}. P-value: {p}")
