"""
Created on Thu Apr 22 2021

@author Name Redacted Surname Redacted
"""


import os
import json
import pickle
import traceback
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from pykg2vec.pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.pykg2vec.common import Importer, KGEArgParser
from pykg2vec.pykg2vec.utils.trainer import Trainer
from src.consts import preprocessed_dataset_path
from src.knowledge_graph.construct_graph import *


"""This file serves to create the embeddings of tracks with knowledge graph embedding tecnique,
with both interactions and metadata (the method we propose).

Things to run to get the embeddings:
`knowledge_graph` -> `knowledge_graph_embedding` -> `tracks_embeddings`.
"""


def knowledge_graph(start_from=0):
    """Save a knowledge graph that represent tracks and information related to the tracks.

    It uses tracks.csv of a given dataset.
    """

    def custom_initilizer(seed):

        t = nx.DiGraph()
        mergiable_id = seed.pop("track_uri")
        t.add_node(
            "source",
            type="source",
            value=seed,
            id="source",
            mergiable_id=mergiable_id,
            tree=t,
        )
        return t

    dataset_name = "mpd"
    base_path = f"{preprocessed_dataset_path}/theme_prediction/{dataset_name}"
    folder_path = f"{base_path}/embeddings/knowledge_graph"
    os.makedirs(folder_path, exist_ok=True)

    df = pd.read_csv(f"{base_path}/tracks.csv")
    seeds = []

    for t in zip(df.track_name, df.track_uri, df.artist_name, df.album_name):

        d = {
            "track_name": t[0],
            "track_uri": t[1],
            "artist_name": t[2],
        }

        if not pd.isna(t[3]):
            d["album_name"] = t[3]

        seeds.append(d)

    if start_from > 0:
        print(f"Starting to build upon saved graph, with seeds up to {start_from} ...")
        graph = nx.read_gpickle(f"{folder_path}/graph_up_to_{start_from}")
        with open(f"{folder_path}/history_up_to_{start_from}", "rb") as f:
            history = pickle.load(f)
    else:
        print(f"Starting to build graph from scratch ...")
        graph = nx.MultiDiGraph()
        history = set()

    for i, seed in enumerate(tqdm(seeds[start_from:])):
        try:
            tree, history = constrained_construct_tree(seed, history, custom_initilizer)
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)
            break
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        graph = compose(graph, tree)

    i = i + start_from
    if i < len(seeds) - 1:
        nx.write_gpickle(graph, f"{folder_path}/graph_up_to_{i}")
        with open(f"{folder_path}/history_up_to_{i}", "wb") as f:
            pickle.dump(history, f)
        print(i)
    else:
        nx.write_gpickle(graph, f"{folder_path}/graph")


def knowledge_graph_embedding():
    """Embeds the knowledge graph created by the method `knowledge_graph`.

    This method first prepares the data, dumping the knowledge graph in triplets, and then uses an algorithm for knowledge graph embedding.

    As for the embedding algorithm, it resorts to the state of the art TransD embedding algorithm [1].
    TransD is implemented by the library pykg2vec [2].

    [1]: Dai, Y., Wang, S., & Xiong, N. N. (2020). A Survey on Knowledge Graph Embedding : Approaches , Applications and Benchmarks. Electronics, 1–29. https://doi.org/10.3390/electronics9050750
    [2]: https://pykg2vec.readthedocs.io/en/latest/
    """
    dataset_name = "mpd"
    print("Saving triplets ...")

    folder_path = f"{preprocessed_dataset_path}/theme_prediction/{dataset_name}"
    os.makedirs(
        f"{folder_path}/embeddings/knowledge_graph/graph_embedding", exist_ok=True
    )

    def context_node_id_strategy(x):
        return f"context:{x}"

    def pid_node_id_strategy(x):
        return f"pid:{x}"

    # read graph created by the method: `knowledge_graph`.
    graph = nx.read_gpickle(f"{folder_path}/embeddings/knowledge_graph/graph")

    # enrich graph with theme annotations
    df = (
        pd.read_csv(f"{folder_path}/train/interactions.csv", usecols=["pid", "tid"])
        .merge(
            pd.read_csv(f"{folder_path}/train/ground_truth.csv"), on="pid", how="left"
        )
        .merge(
            pd.read_csv(f"{folder_path}/tracks.csv", usecols=["tid", "track_uri"]),
            on="tid",
            how="left",
        )
    )
    for pid, group in tqdm(list(df.groupby("pid"))):
        context_node_id = context_node_id_strategy(group.context.values[0])
        pid_node_id = pid_node_id_strategy(pid)
        assert pid_node_id not in graph
        graph.add_node(pid_node_id)
        graph.add_node(context_node_id)
        graph.add_edge(pid_node_id, context_node_id, type="context")
        for track_node_id in group.track_uri:
            graph.add_node(track_node_id)
            graph.add_edge(track_node_id, pid_node_id, type="interaction")

    # save triplets
    with open(
        f"{folder_path}/embeddings/knowledge_graph/graph_embedding/{dataset_name}-train.txt",
        "w",
    ) as f:

        s = ""
        for n1, n2 in tqdm(graph.edges()):
            for i in graph[n1][n2]:
                s += f"{n1}\t{graph[n1][n2][i]['type']}\t{n2}\n"
        f.write(s)

    # Embedding library requires a file for validation and testing.
    # We save just one record to shorten validation and testing.

    with open(
        f"{folder_path}/embeddings/knowledge_graph/graph_embedding/{dataset_name}-valid.txt",
        "w",
    ) as f:
        f.write(s.split("\n")[-2])

    with open(
        f"{folder_path}/embeddings/knowledge_graph/graph_embedding/{dataset_name}-test.txt",
        "w",
    ) as f:
        f.write(s.split("\n")[-2])

    print("Done.")

    # In practice, the following part was run in colab, 50 epochs at a time, until saturated the decrease.
    params = [
        "-mn",
        "transd",
        "-ds",
        dataset_name,
        "-dsp",
        f"{folder_path}/embeddings/knowledge_graph/graph_embedding",
        "-l",
        "2000",
    ]

    # From https://github.com/Sujit-O/pykg2vec/blob/master/scripts/pykg2vec_train.py
    args = KGEArgParser().get_args(params)
    knowledge_graph = KnowledgeGraph(
        dataset=args.dataset_name, custom_dataset_path=args.dataset_path
    )
    knowledge_graph.prepare_data()

    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args)
    model = model_def(**config.__dict__)

    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()


def tracks_embeddings():
    """Save tracks embeddings as created by the Knowledge Graph embedding method.
    This methods is meant to be run after `knowledge_graph` and `knowledge_graph_embedding`.
    """
    dataset_name = "mpd"
    base_path = f"{preprocessed_dataset_path}/theme_prediction/{dataset_name}"
    embeddings_path = f"{base_path}/embeddings/knowledge_graph"

    map_trackuri_to_embedding_matrix_row = {}
    embedding_matrix = []
    with open(
        f"{embeddings_path}/graph_embedding/embeddings/transd/ent_labels.tsv"
    ) as f_labels:
        with open(
            f"{embeddings_path}/graph_embedding/embeddings/transd/ent_embedding.tsv"
        ) as f_embeddings:

            line_number = 0
            while True:
                label = f_labels.readline()[:-1]
                embeddings = f_embeddings.readline()
                if not label:
                    break

                if "spotify:track:" in label:
                    map_trackuri_to_embedding_matrix_row[label] = line_number
                    embedding_matrix.append([float(e) for e in embeddings.split("\t")])
                    line_number += 1

    embedding_matrix = np.array(embedding_matrix)
    tracks = pd.read_csv(f"{base_path}/tracks.csv", usecols=["tid", "track_uri"])
    tracks_embeddings = {}

    for tid, track_uri in zip(tracks.tid, tracks.track_uri):
        tracks_embeddings[int(tid)] = [
            float(e)
            for e in embedding_matrix[
                map_trackuri_to_embedding_matrix_row[track_uri], :
            ]
        ]

    with open(f"{embeddings_path}/tracks.json", "w", encoding="utf-8") as f:
        json.dump(tracks_embeddings, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    knowledge_graph(0)
