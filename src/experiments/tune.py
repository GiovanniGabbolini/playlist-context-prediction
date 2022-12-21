import os
import json
import argparse
from src.datasets import EmbeddingsPlaylistLevel, EmbeddingsTrackLevel, load_embedding_dictionary, EmbeddingsTuple
from ax.service.managed_loop import optimize
from src.consts import preprocessed_dataset_path
from src.models.average_two_steps import run as run_average_two_steps
from src.models.sequence_two_steps import run as run_sequence_two_steps
from src.models.average_audio import run as run_average_audio
from src.models.sequence_audio import run as run_sequence_audio
from src.models.average_hybrid import run as run_average_hybrid
from src.models.sequence_hybrid import run as run_sequence_hybrid


def write_down(method, parameters, metrics):
    path = f"{preprocessed_dataset_path}/results/{method}"
    os.makedirs(path, exist_ok=True)

    try:
        with open(f"{path}/results.json") as f:
            l = json.load(f)
    except FileNotFoundError:
        l = []

    parameters["accuracy"] = metrics
    l.append(parameters)
    with open(f"{path}/results.json", "w") as f:
        json.dump(l, f, ensure_ascii=False, indent=4)


def evaluation_function(parameters):
    # free memory
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()

    metrics = run(parameters, training_data, validation_data)
    write_down(args.what, parameters, metrics)
    return metrics["FH@1"]


parser = argparse.ArgumentParser(description="What do you want to tune?")
parser.add_argument("--what", type=str, default="mf_avg")
args = parser.parse_args()

trials = 20
parameters = [
    {"name": "learning_rate", "type": "range", "bounds": [10.0 ** -6, 0.01]},
    {"name": "weight_decay", "type": "range", "bounds": [10.0 ** -6, 0.01]},
    {"name": "batch_size", "type": "range", "bounds": [128, 1024]},
    {"name": "num_workers", "type": "fixed", "value": 0},
]

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
    validation_data = EmbeddingsPlaylistLevel(embeddings_dictionary, "validation")
elif args.what in ["mf_seq", "kg_seq", "kg_int_only_seq", "audio_avg", "audio_seq"]:
    training_data = EmbeddingsTrackLevel(embeddings_dictionary, "train")
    validation_data = EmbeddingsTrackLevel(embeddings_dictionary, "validation")
elif args.what in ["hybrid_avg", "hybrid_seq"]:
    training_data = EmbeddingsTuple(EmbeddingsTrackLevel(embeddings_dictionary_1, "train"), EmbeddingsTrackLevel
                                    (embeddings_dictionary_2, "train"))
    validation_data = EmbeddingsTuple(EmbeddingsTrackLevel(embeddings_dictionary_1, "validation"),
                                      EmbeddingsTrackLevel(embeddings_dictionary_2, "validation"))

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

optimize(
    parameters=parameters,
    evaluation_function=evaluation_function,
    objective_name="accuracy",
    total_trials=trials,
)
