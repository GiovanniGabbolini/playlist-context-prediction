from src.consts import preprocessed_dataset_path
import pandas as pd
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

plt.style.use("science")


def visualisation(method):
    dataset_name = "mpd"
    folder_path = f"{preprocessed_dataset_path}/theme_prediction/{dataset_name}"
    split = "train"

    ground_truth = pd.read_csv(f"{folder_path}/{split}/ground_truth.csv")
    interactions = pd.read_csv(f"{folder_path}/{split}/interactions.csv")
    with open(f"{folder_path}/annotations.json") as f:
        annotations = json.load(f)
    with open(f"{folder_path}/embeddings/{method}/tracks.json") as f:
        tracks_embeddings = json.load(f)

    # retrieve the ten most frequent contexts
    contexts = ground_truth.context.value_counts()[:5].index.values

    embedding_matrix = []
    classes = []
    for c in tqdm(contexts):
        for pid in ground_truth[ground_truth.context == c].pid:

            l = []
            for tid in interactions[interactions.pid == pid].tid:
                l.append(tracks_embeddings[str(tid)])
            embedding_matrix.append(np.average(np.array(l), axis=0))
            classes.append(c)

    embedding_matrix = np.array(embedding_matrix)
    tsne = TSNE(n_jobs=-1)
    tsne_results = tsne.fit_transform(embedding_matrix)

    for c in contexts:
        l = [idx for idx, cl in enumerate(classes) if cl == c]
        plt.scatter(
            tsne_results[l, 0],
            tsne_results[l, 1],
            label=annotations[str(c)][0],
            s=0.1,
            marker=".",
        )

    plt.legend(bbox_to_anchor=(1, 1), markerscale=12)
    plt.savefig(f"embeddings_{method}", dpi=1800)


if __name__ == "__main__":
    visualisation("matrix_factorization")
