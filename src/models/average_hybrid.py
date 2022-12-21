from src.metrics import compute_metrics
from src.utils.to_string import to_string
from tqdm import tqdm
import torch
from src.datasets import load_embedding_dictionary, EmbeddingsTrackLevel
from src.early_stopping import EarlyStopping
from src.models.commons import CategoricalClassification, EmbeddingFusion
from src.models.average_audio import AverageAudio


def run(params, training_data, test_data, return_metrics=True):
    """
    The shape of the net is the avg. audio model as in [1]. (CNN + Fully connected layer).
    We fix the optimiser to Adam, this is a sensible choice. We also fix:
        - Adam's betas (coefficient used to compute running averages of gradient);
        - Adam's eps (used for numerical stability);
    We fix the activation function to ReLU, this is a sensible choice.
    We set the number of epochs to infinity and we use early stopping. We will see whether we would need more than that or not.
    We set the patience of early stopping to 10, that is if after 10 epochs the accuracy hasn't increased, we stop.

    Hyper-params we tune:
    - batch size;
    - learning rate;
    - regularization, aka weight decay aka L2 penalty;

    [1]: Choi, J., & Epure, E. V. (2020). Prediction of User Listening Contexts for Music Playlists.
    """
    assert set(list(params.keys())) == set(
        ["batch_size", "learning_rate", "weight_decay", "num_workers"]
    )

    # torch stuff
    train_dataloader = torch.utils.data.DataLoader(
        training_data,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        collate_fn=lambda x: x,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        collate_fn=lambda x: x,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    audio_model = AverageAudio().to(device)
    classification_model = CategoricalClassification().to(device)
    fusion_model = EmbeddingFusion().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [*audio_model.parameters(), *classification_model.parameters(), *fusion_model.parameters()],
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    early_stopping = EarlyStopping(patience=10)

    array = [None]
    while True:
        train_loop(
            train_dataloader, audio_model, classification_model, fusion_model, loss_fn, optimizer
        )
        predictions, ys = test_loop(
            test_dataloader, audio_model, classification_model, fusion_model, loss_fn
        )
        metrics = compute_metrics(predictions, ys)
        array.append((predictions, ys))

        print(to_string(metrics))
        if early_stopping.stop(metrics):
            break

    best_epoch, metrics = early_stopping.best()
    if return_metrics:
        return metrics
    else:
        return array[best_epoch]


def train_loop(dataloader, audio_model, classification_model, fusion_model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for n_batch, batch in enumerate(tqdm(dataloader)):

        Y = [sample[2].to(device) for sample in batch]

        ### Audio part ###

        X = [sample[0].to(device) for sample in batch]

        split_idx = [sample.shape[0] for sample in X]

        X = torch.cat(X)
        Y = torch.stack(Y)

        # embedding of all spectrograms
        emb = audio_model(X)
        emb = emb[:, :, 0]

        # get playlists embeddings averaging with avg
        emb_audio = torch.stack(
            [
                torch.mean(tensor, axis=0)
                for tensor in torch.split(emb, split_idx, dim=0)
            ]
        )

        ### KG part ###
        X = [sample[1].to(device) for sample in batch]
        X = [torch.mean(x, 0) for x in X]
        emb_kg = torch.stack(X)

        ### Fusion ###
        emb = fusion_model(emb_audio, emb_kg)

        # classify
        pred = classification_model(emb)
        loss = loss_fn(pred, Y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n_batch % 100 == 0:
            loss, _ = loss.item(), n_batch * len(X)


def test_loop(dataloader, audio_model, classification_model, fusion_model, loss_fn):
    test_loss = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions = []
    ys = []

    with torch.no_grad():
        for batch in tqdm(dataloader):

            Y = [sample[2].to(device) for sample in batch]

            ### Audio part ###

            X = [sample[0].to(device) for sample in batch]

            split_idx = [sample.shape[0] for sample in X]

            X = torch.cat(X)
            Y = torch.stack(Y)

            # embedding of all spectrograms
            emb = audio_model(X)
            emb = emb[:, :, 0]

            # get playlists embeddings averaging with avg
            emb_audio = torch.stack(
                [
                    torch.mean(tensor, axis=0)
                    for tensor in torch.split(emb, split_idx, dim=0)
                ]
            )

            ### KG part ###
            X = [sample[1].to(device) for sample in batch]
            X = [torch.mean(x, 0) for x in X]
            emb_kg = torch.stack(X)

            ### Fusion ###
            emb = fusion_model(emb_audio, emb_kg)

            # classify
            pred = classification_model(emb)

            predictions.append(pred)
            ys.append(Y)

            test_loss += loss_fn(pred, Y).item()

    predictions, ys = torch.cat(predictions), torch.cat(ys)

    test_loss /= len(dataloader.dataset)
    return predictions, ys


if __name__ == "__main__":
    embeddings_dictionary = load_embedding_dictionary("audio")
    run(
        {
            "batch_size": 32,
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "num_workers": 8,
        },
        EmbeddingsTrackLevel(embeddings_dictionary, "train"),
        EmbeddingsTrackLevel(embeddings_dictionary, "validation"),
    )
