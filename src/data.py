import io
import os
import json
import torch
import pydub
import urllib
import librosa
import warnings
import requests
import traceback
import soundfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import json
import os
import pandas as pd
import numpy as np
import implicit
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from src.consts import raw_dataset_path, preprocessed_dataset_path, audio_path


def prepare_mpd():
    """Theme annotations of playlists in the Spotify's Million Playlist Dataset (MPD).

    They find a number of different contexts, each of them identified by playlist titles.
    e.g. Context 1: summer, summer vibes -> playlists with titles summer and summer vibes will be assigned to context 1.

    The annotations were made by the authors of [1].

    For every titles associated to context, we find all the playlists in MPD with that title.
    We assign to such playlists the corresponding context, and form a ground truth.

    This files generates:
    - annotations.json: {context 1: [title 1, title 2], context 2: [title 3, title 4], ...};

    - three folders: train; validation; test. in every folder generates:
        - ground_truth.csv: playlist id (pid), context;
        - interactions.csv: slice from MDP, with just the playlists in ground_truth.csv;
      Notice:
      + the splitting in train, validation and test is 60, 20, 20%, statified by context;
      + we remove validation and test playlists all the tracks that do not occour also in the train.
        doing so is required by a baseline introduced in [1], that we benchmark;
      + we remove from train, validation and test playlists all the tracks we don't have audio.

    - tracks.csv: slice from MDP, with just the tracks of the playlists in the training set,
                  and of which we can have the audio.
                  Notice: in validation and test we find only the tracks in train, see above.

    - audio folder: 30s mp3 audio previews for songs in tracks.csv (folder raw);
                    mel spectrogram of most pop tracks (folder preprocessed).

    - pid2tids.json: utility file that indices all the tids belonging to a pid.

    [1]: Choi, J., & Epure, E. V. (2020). Prediction of User Listening Contexts for Music Playlists.
    """

    def read_csv_with_condition(what, chunksize, condition):
        """Read a raw MPD dataset file, and filter it out based on some condition while reading it,
        instead of reading everything and then filtering. This is more memory efficient.

        It reads in chunks, and it is possible to specify the chunk size.

        Args:
            what (str)
            chunksize (int)
            condition (callable): filter on the chunks.

        Returns the dataframe as read and filtered.
        """
        return_value = []
        chunks = pd.read_csv(
            f"{raw_dataset_path}/spotify_recsys2018/{what}.csv",
            chunksize=chunksize,
            delimiter="\t",
        )
        for chunk in tqdm(chunks):
            chunk = chunk[condition(chunk)]
            return_value.append(chunk)
        return_value = pd.concat(return_value)
        return return_value

    raw_audio_path = f"{audio_path}/raw"
    preprocessed_audio_path = f"{audio_path}/preprocessed"
    os.makedirs(preprocessed_audio_path, exist_ok=True)
    os.makedirs(raw_audio_path, exist_ok=True)

    annotations = json.load(open(f"{preprocessed_dataset_path}/annotations.json"))

    # create ground_truth
    df = pd.read_csv(
        f"{raw_dataset_path}/spotify_recsys2018/playlists.csv",
        delimiter="\t",
        usecols=["pid", "name"],
    )
    df.name = df.name.str.lower()
    ground_truth = []
    for context, titles in annotations.items():
        df_context = pd.concat([df[df.name == title] for title in titles])
        df_context.insert(0, "context", context)
        ground_truth.append(df_context)
    ground_truth = pd.concat(ground_truth)[["pid", "context"]]

    # create interactions
    valid_pid = list(ground_truth.pid.values)
    interactions = read_csv_with_condition(
        "interactions", 100000, lambda chunk: chunk.pid.isin(valid_pid)
    )

    # split playlists into train, validation and test, with 60, 20 and 20%, stratified by context annotations
    ground_truth_trai, ground_truth_hold = train_test_split(
        ground_truth, stratify=ground_truth.context, test_size=0.4, random_state=42
    )
    ground_truth_vali, ground_truth_test = train_test_split(
        ground_truth_hold,
        stratify=ground_truth_hold.context,
        test_size=0.5,
        random_state=42,
    )

    interactions_trai = interactions[
        interactions.pid.isin(ground_truth_trai.pid.values)
    ]
    interactions_vali = interactions[
        interactions.pid.isin(ground_truth_vali.pid.values)
    ]
    interactions_test = interactions[
        interactions.pid.isin(ground_truth_test.pid.values)
    ]

    # filter out tracks that occour only on hold-out playlists
    tid_exclusively_in_hold = list(
        (set(interactions_vali.tid.values) | set(interactions_test.tid.values))
        - set(interactions_trai.tid.values)
    )
    interactions_vali = interactions_vali[
        ~interactions_vali.tid.isin(tid_exclusively_in_hold)
    ]
    interactions_test = interactions_test[
        ~interactions_test.tid.isin(tid_exclusively_in_hold)
    ]

    # filter out playlists on vali and test left without tracks in interactions by the previous filter
    ground_truth_vali = ground_truth_vali[
        ground_truth_vali.pid.isin(set(interactions_vali.pid.values))
    ]
    ground_truth_test = ground_truth_test[
        ground_truth_test.pid.isin(set(interactions_test.pid.values))
    ]

    # create tracks
    valid_tid = list(set(interactions_trai.tid.values))
    tracks = read_csv_with_condition(
        "tracks", 10000, lambda chunk: chunk.tid.isin(valid_tid)
    )
    tracks = tracks[["tid", "arid", "alid", "track_uri", "track_name"]]

    valid_alid = list(set(tracks.alid.values))
    valid_arid = list(set(tracks.arid.values))

    albums = read_csv_with_condition(
        "albums", 10000, lambda chunk: chunk.alid.isin(valid_alid)
    )
    artists = read_csv_with_condition(
        "artists", 10000, lambda chunk: chunk.arid.isin(valid_arid)
    )

    tracks = tracks.merge(albums, on="alid", how="left").merge(
        artists, on="arid", how="left"
    )

    # scrape the audio
    _scrape_audio(tracks[["track_uri"]], raw_audio_path)

    # gather tracks we don't have audio
    tid_we_dont_have_audio = []
    for track_uri, tid in tqdm(zip(tracks.track_uri, tracks.tid)):
        if not os.path.exists(f"{audio_path}/raw/{track_uri.split(r':')[-1]}.mp3"):
            tid_we_dont_have_audio.append(tid)

    # filter out tracks we don't have audio
    tracks = tracks[~tracks.tid.isin(tid_we_dont_have_audio)]
    interactions_trai = interactions_trai[
        ~interactions_trai.tid.isin(tid_we_dont_have_audio)
    ]
    interactions_vali = interactions_vali[
        ~interactions_vali.tid.isin(tid_we_dont_have_audio)
    ]
    interactions_test = interactions_test[
        ~interactions_test.tid.isin(tid_we_dont_have_audio)
    ]

    # filter out playlists on trai, vali and test left without tracks in interactions by the previous filter
    ground_truth_trai = ground_truth_trai[
        ground_truth_trai.pid.isin(set(interactions_trai.pid.values))
    ]
    ground_truth_vali = ground_truth_vali[
        ground_truth_vali.pid.isin(set(interactions_vali.pid.values))
    ]
    ground_truth_test = ground_truth_test[
        ground_truth_test.pid.isin(set(interactions_test.pid.values))
    ]

    # save tracks, ground_truth and interactions
    tracks.to_csv(f"{preprocessed_dataset_path}/tracks.csv", index=False)

    # preprocess audio
    tracks = pd.read_csv(f"{preprocessed_dataset_path}/tracks.csv")
    _preprocess_audio(tracks, raw_audio_path, preprocessed_audio_path)

    for split, t in [
        ("train", (ground_truth_trai, interactions_trai)),
        ("validation", (ground_truth_vali, interactions_vali)),
        ("test", (ground_truth_test, interactions_test)),
    ]:
        os.makedirs(f"{preprocessed_dataset_path}/{split}", exist_ok=True)
        t[0].to_csv(f"{preprocessed_dataset_path}/{split}/ground_truth.csv", index=False)
        t[1].to_csv(f"{preprocessed_dataset_path}/{split}/interactions.csv", index=False)
        print(f"Number of playlists in {split}: {len(t[0])}")
        print(
            f"Average and std number of tracks for playlists in {split}: \
            {t[1].groupby('pid').count().tid.mean()}\
            {t[1].groupby('pid').count().tid.std()}"
        )

    _save_pid2tids("mpd")


def _scrape_audio(df, base_path):
    # Scrape audio from spotify
    # This method is called by `prepare_mpd`.
    for track_uri in tqdm(df.track_uri):

        file_path = f"{base_path}/{track_uri.split(r':')[-1]}"
        if os.path.exists(f"{file_path}.mp3"):
            continue

        # Fetch preview url.
        embed_url = f"https://open.spotify.com/embed/track/{track_uri.split(':')[-1]}"
        try:
            soup = BeautifulSoup(requests.get(embed_url).text, "html.parser")
            url = json.loads(
                urllib.parse.unquote(soup.find("script", {"id": "resource"}).string)
            )["preview_url"]
        except Exception as ex:
            print(
                f"An exception has happened for {track_uri} while resolving the url of the preview, skipping."
            )
            traceback.print_exception(type(ex), ex, ex.__traceback__)
            continue

        if url is None:
            print(f"Preview url not available for {track_uri}, skipping.")
            continue

        try:
            wav = io.BytesIO()
            with urllib.request.urlopen(url) as r:
                r.seek = lambda *args: None  # allow pydub to call seek(0)
                pydub.AudioSegment.from_file(r).export(wav, "wav")
            wav.seek(0)
        except Exception as ex:
            print(
                f"An exception has happened for {track_uri} while downloading the audio preview, skipping."
            )
            traceback.print_exception(type(ex), ex, ex.__traceback__)
            continue

        y, sr = librosa.load(wav)
        assert sr == 22050
        soundfile.write(f"{file_path}.wav", y, sr)

        song = pydub.AudioSegment.from_file(f"{file_path}.wav", format="wav")
        _ = song.export(f"{file_path}.mp3", format="mp3", bitrate="36k")
        os.remove(f"{file_path}.wav")


def _preprocess_audio(df_tracks, raw_audio_path, preprocessed_audio_path):
    # Save mel spectrograms of audio, by calling the _spectrogram_chunks method.
    # This method is called by `prepare_mpd`.

    for uri in tqdm(df_tracks.track_uri):

        uri = uri.split(":")[-1]

        song_path = f"{raw_audio_path}/{uri}.mp3"
        spectrogram_path = f"{preprocessed_audio_path}/{uri}"

        if not os.path.exists(f"{spectrogram_path}.npy"):
            try:
                np.save(spectrogram_path, _spectrogram_chunks(song_path))
            except FileNotFoundError:
                print(f"The audio for track {uri} is not available, skipping.")


def _spectrogram_chunks(path):
    """Return mel spectrograms of the audio in path.

    Args:
        path (str): Path to audio. It is assumed it was saved by the _scrape_audio method.

    Returns:
        array: Spectrogram, as avg of 3s spectrograms.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # the audio has 661500 samples, that is 30 seconds at 22050 sample per second.
        audio = np.zeros(661500, dtype=np.float32)
        try:
            f = librosa.load(path, sr=None)[0][:661500]
            audio[: f.shape[0]] = f
        except EOFError:
            pass

    spectrograms = []
    for chunk in np.split(audio, 10):
        spectrograms.append(
            librosa.feature.melspectrogram(
                y=chunk, sr=22050, n_fft=1024, hop_length=512, n_mels=128
            )
        )
    return_value = np.mean(spectrograms, axis=0)

    return return_value


def _save_pid2tids(dataset):
    pid2tids = {}
    for split in ["train", "validation", "test"]:

        df = pd.read_csv(
            f"{preprocessed_dataset_path}/theme_prediction/{dataset}/{split}/interactions.csv"
        )
        for pid, tid in tqdm(zip(df["pid"], df["tid"])):
            try:
                pid2tids[pid].append(tid)
            except KeyError:
                pid2tids[pid] = [tid]

    with open(
        f"{preprocessed_dataset_path}/theme_prediction/{dataset}/pid2tids.json",
        "w",
    ) as f:
        json.dump(pid2tids, f, indent=4)


def embeddings_matrix_factorization():
    """Embeds tracks based on Matrix Factorization (MF), as described in [1].

    As a MF algorithm, it uses Alternating Least Square. Since we are not given further indication, we resort to the
    library the implementation of [2] to be found in the implicit library [3].

    We set the embedding dimension to 50, as done in [1].
    We leave leave the other params as default.

    [1]: Choi, J., & Epure, E. V. (2020). Prediction of User Listening Contexts for Music Playlists.
    [2]: Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008. Collaborative Filtering for Implicit Feedback Datasets. In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM '08).
    [3]: https://github.com/benfred/implicit
    """
    embeddings_path = f"{raw_dataset_path}/embeddings/matrix_factorization"
    os.makedirs(embeddings_path, exist_ok=True)

    # init matrix
    annotations = json.load(open(f"{preprocessed_dataset_path}/annotations.json"))
    contexts = [int(k) for k in annotations.keys()]

    # all tids can be found in training set
    tids = list(
        set(
            pd.read_csv(
                f"{preprocessed_dataset_path}/train/interactions.csv", usecols=["tid"]
            ).tid.values
        )
    )
    tids_to_matrix_cols = {k: v for k, v in zip(tids, range(len(tids)))}

    matrix = np.zeros((len(contexts), len(tids)), dtype=np.int32)

    df = pd.read_csv(f"{preprocessed_dataset_path}/train/ground_truth.csv")
    pid2tids = json.load(open(f"{preprocessed_dataset_path}/pid2tids.json"))

    for pid, context in zip(df["pid"], df["context"]):
        tids = pid2tids[str(pid)]
        matrix_cols = [tids_to_matrix_cols[tid] for tid in tids]
        matrix[context, matrix_cols] += 1

    # td-idf transformation to be operated track-wise, [1].
    matrix = TfidfTransformer(norm=False, smooth_idf=False).fit_transform(matrix.T).T

    model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=50, regularization=0.01, random_state=42
    )
    model.fit(matrix)

    for _, tids in tqdm(pid2tids.items()):
        for tid in tids:
            col = tids_to_matrix_cols[tid]

            embedding = np.array([float(e) for e in model.item_factors[col]])

            path = f"{embeddings_path}/{tid}.npy"
            if not os.path.exists(path):
                np.save(path, embedding)


def embeddings_knowledge_graph(pendix=""):
    """
    Use pendix=="" for standard embedding and pendix=="_interactions_only" for the ablation study.
    """
    pid2tids = json.load(open(f"{preprocessed_dataset_path}/pid2tids.json"))
    tid_embedding = json.load(open(f"{preprocessed_dataset_path}/embeddings_knowledge_graph{pendix}.json"))

    embeddings_path = f"{raw_dataset_path}/embeddings/knowledge_graph{pendix}"
    os.makedirs(embeddings_path, exist_ok=True)

    for _, tids in tqdm(pid2tids.items()):
        for tid in tids:

            try:
                embedding = tid_embedding[str(tid)]

                path = f"{embeddings_path}/{tid}.npy"
                if not os.path.exists(path):
                    np.save(path, embedding)

            except KeyError:
                continue


def embedding_dictionary(embedding_type):
    """
    To be ran after `embeddings_matrix_factorization()` and `embeddings_knowledge_graph()`,
    as `embedding_dictionary(matrix_factorization)` and `embedding_dictionary(knowledge_graph)`.

    Saves a track-level embedding dictionary to be read before training.
    """
    pid2tids = json.load(open(f"{preprocessed_dataset_path}/pid2tids.json"))

    if embedding_type == "audio":
        tracks = pd.read_csv(f"{preprocessed_dataset_path}/tracks.csv", usecols=["tid", "track_uri"]).set_index("tid", drop=True)

    d = {}

    for split in ["train", "validation", "test"]:

        df = pd.read_csv(f"{preprocessed_dataset_path}/{split}/ground_truth.csv")

        for pid in tqdm(df.pid):
            for tid in pid2tids[str(pid)]:

                if tid in d:
                    continue

                try:

                    if embedding_type in ["matrix_factorization", "knowledge_graph", "knowledge_graph_interactions_only"]:
                        source = f"{raw_dataset_path}/embeddings/{embedding_type}/{tid}.npy"
                    else:
                        turi = str(tracks.loc[tid].track_uri)
                        turi = turi.split(":")[-1]
                        source = f"{audio_path}/preprocessed/{turi}.npy"

                    embedding = np.load(source)
                    embedding = embedding.astype(np.float32)

                    d[tid] = embedding
                except FileNotFoundError:
                    continue

    np.save(f"{raw_dataset_path}/embeddings/dict_{embedding_type}.npy", d)


if __name__ == "__main__":
    prepare_mpd()
    embeddings_matrix_factorization()
    embeddings_knowledge_graph()
    embeddings_knowledge_graph("_interactions_only")
    embedding_dictionary("knowledge_graph_interactions_only")
    embedding_dictionary("matrix_factorization")
    embedding_dictionary("audio")
