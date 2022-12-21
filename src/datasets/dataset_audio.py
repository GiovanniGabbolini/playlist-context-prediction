import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.consts import preprocessed_dataset_path
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


class Audio(torch.utils.data.Dataset):

    """Pytorch dataset for models using audio.
    Every sample is a: (playlist, context).
    In this case:
    - the data represeting a playlist are the name of the mp3 files of the audio files;
    - the context is the context of the playlist.

    We use a caching mechanism. It relies on the ordered nature of dictionaries from Python3.7+. On lower versions, the caching mechanism is less efficient.
    """

    def __init__(self, split):
        assert split in ["train", "validation", "test"]

        self.base_path = f"{preprocessed_dataset_path}/theme_prediction/mpd"
        self.df = pd.read_csv(f"{self.base_path}/{split}/ground_truth.csv")
        self.pid2tids = json.load(open(f"{self.base_path}/pid2tids.json"))
        self.tracks = pd.read_csv(
            f"{self.base_path}/tracks.csv", usecols=["tid", "track_uri"]
        ).set_index("tid", drop=True)
        self.cache = self._load_dict()

    def _load_dict(self):
        d = {}
        for pid in tqdm(self.df.pid):
            for tid in self.pid2tids[str(pid)]:

                if tid in d:
                    continue

                turi = str(self.tracks.loc[tid].track_uri)
                turi = turi.split(":")[-1]

                embedding = np.load(f"{self.base_path}/spectrograms/{turi}.npy")
                embedding = torch.from_numpy(embedding)

                d[tid] = embedding

        return d

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        slice = self.df.iloc[idx]

        x = []
        for tid in self.pid2tids[str(slice.pid)]:
            x_to_append = self.cache[tid]
            x.append(x_to_append)
        x = torch.stack(x)

        y = torch.tensor(int(slice.context))

        return x, y
