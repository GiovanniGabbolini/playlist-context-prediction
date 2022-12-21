import os
import json
import argparse
import numpy as np
from time import time
from src.metrics import compute_metrics
from src.consts import preprocessed_dataset_path
from src.datasets import EmbeddingsPlaylistLevel, EmbeddingsTrackLevel, load_embedding_dictionary, EmbeddingsTuple
from src.models.average_two_steps import run as run_average_two_steps
from src.models.sequence_two_steps import run as run_sequence_two_steps
from src.models.average_audio import run as run_average_audio
from src.models.sequence_audio import run as run_sequence_audio
from src.models.average_hybrid import run as run_average_hybrid
from src.models.sequence_hybrid import run as run_sequence_hybrid


parser = argparse.ArgumentParser(description="What do you want to run (best params)?")
parser.add_argument("--what", type=str, default="hybrid_seq")
args = parser.parse_args()

assert args.what in ["mf_avg", "mf_seq", "kg_avg", "kg_seq", "kg_int_only_avg",
                     "kg_int_only_seq", "audio_avg", "audio_seq", "hybrid_avg", "hybrid_seq"]

if args.what in ["mf_avg", "mf_seq"]:
    embeddings_dictionary = load_embedding_dictionary("matrix_factorization")
elif args.what in ["kg_avg", "kg_seq"]:
    embeddings_dictionary = load_embedding_dictionary("knowledge_graph")
elif args.what in ["kg_int_only_avg", "kg_int_only_seq"]:
    embeddings_dictionary = load_embedding_dictionary("knowledge_graph_interactions_only")
elif "audio" in args.what:
    embeddings_dictionary = load_embedding_dictionary("audio")
elif "hybrid" in args.what:
    embeddings_dictionary_1 = load_embedding_dictionary("audio")
    embeddings_dictionary_2 = load_embedding_dictionary("knowledge_graph")


if args.what in ["mf_avg", "kg_avg", "kg_int_only_avg"]:
    training_data = EmbeddingsPlaylistLevel(embeddings_dictionary, "train")
    test_data = EmbeddingsPlaylistLevel(embeddings_dictionary, "test")
elif args.what in ["mf_seq", "kg_seq", "kg_int_only_seq", "audio_avg", "audio_seq"]:
    training_data = EmbeddingsTrackLevel(embeddings_dictionary, "train")
    test_data = EmbeddingsTrackLevel(embeddings_dictionary, "test")
elif args.what in ["hybrid_avg", "hybrid_seq"]:
    training_data = EmbeddingsTuple(EmbeddingsTrackLevel(embeddings_dictionary_1, "train"), EmbeddingsTrackLevel
                                    (embeddings_dictionary_2, "train"))
    test_data = EmbeddingsTuple(EmbeddingsTrackLevel(embeddings_dictionary_1, "test"),
                                EmbeddingsTrackLevel(embeddings_dictionary_2, "test"))

if args.what in ["mf_avg", "kg_avg", "kg_int_only_avg"]:
    run = run_average_two_steps
elif args.what in ["mf_seq", "kg_seq", "kg_int_only_seq"]:
    run = run_sequence_two_steps
elif args.what == "audio_avg":
    run = run_average_audio
elif args.what == "audio_seq":
    run = run_sequence_audio
elif args.what == "hybrid_avg":
    run = run_average_hybrid
elif args.what == "hybrid_seq":
    run = run_sequence_hybrid

# Seek best model
results = json.load(open(f"{preprocessed_dataset_path}/results/{args.what}/results.json"))
results = sorted(results, key=lambda x: x["accuracy"]["FH@1"], reverse=True)

best_params = results[0]
best_params.pop("accuracy")

start = time()
predictions, ys = run(best_params, training_data, test_data, return_metrics=False)
run_time = time()-start

metrics = compute_metrics(predictions, ys)
metrics["run_time"] = run_time

path = f"{preprocessed_dataset_path}/results/{args.what}/best_run/"
os.makedirs(path, exist_ok=True)

predictions = predictions.cpu().detach().numpy()
ys = ys.cpu().detach().numpy()

json.dump(metrics, open(f"{path}/metrics.json", "w"), indent=4)
np.save(f"{path}/predictions.npy", predictions)
np.save(f"{path}/ys.npy", ys)
