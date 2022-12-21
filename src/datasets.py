import json
import torch
import numpy as np
import pandas as pd
from src.consts import preprocessed_dataset_path, raw_dataset_path
import json
import pandas as pd
import numpy as np
from time import time


def load_embedding_dictionary(method):
    assert method in ["matrix_factorization", "knowledge_graph", "audio", "knowledge_graph_interactions_only"]

    print(f"Started reading embeddings {method}.")
    start = time()
    d = np.load(f"{raw_dataset_path}/embeddings/dict_{method}.npy", allow_pickle=True).item()
    print(f"Ended reading embeddings. It took: {time()-start}")

    return d


class EmbeddingsTrackLevel(torch.utils.data.Dataset):

    def __init__(self, embeddings_dictionary, split="train"):
        assert split in ["train", "validation", "test"]

        self.dict = embeddings_dictionary
        self.df = pd.read_csv(f"{preprocessed_dataset_path}/{split}/ground_truth.csv")
        self.pid2tids = json.load(open(f"{preprocessed_dataset_path}/pid2tids.json"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        slice = self.df.iloc[idx]
        pid = int(slice.pid)
        tids = self.pid2tids[str(pid)]

        x = []

        for tid in tids:
            try:
                embedding = self.dict[tid]
                embedding = torch.from_numpy(embedding)
                x.append(embedding)
            except KeyError:
                continue

        x = torch.stack(x)
        y = torch.tensor(int(slice.context))

        return x, y


class EmbeddingsPlaylistLevel(EmbeddingsTrackLevel):

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        x = torch.mean(x, 0)
        return x, y


class EmbeddingsTuple(torch.utils.data.Dataset):
    """
    Tuples made of two datasets objects of type EmbeddingsTrackLevel.
    """

    def __init__(self, embeddings_1, embeddings_2):
        assert all(type(o) is EmbeddingsTrackLevel for o in [embeddings_1, embeddings_2])
        self.embedding_1 = embeddings_1
        self.embedding_2 = embeddings_2

    def __len__(self):
        assert self.embedding_1.__len__() == self.embedding_2.__len__()
        return self.embedding_1.__len__()

    def __getitem__(self, idx):
        x1, y1 = self.embedding_1.__getitem__(idx)
        x2, y2 = self.embedding_2.__getitem__(idx)
        assert y1 == y2
        return x1, x2, y1
