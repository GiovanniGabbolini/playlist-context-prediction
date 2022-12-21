from src.utils.to_string import to_string
from src.datasets import EmbeddingsTrackLevel
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import nn
from src.models.commons import SequenceCategoricalClassification
from src.early_stopping import EarlyStopping
from src.metrics import compute_metrics
from tqdm import tqdm


def test_loop(dataloader, model, loss_fn):
    test_loss = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions = []
    ys = []

    with torch.no_grad():
        for batch in dataloader:

            X = [sample[0].to(device) for sample in batch]
            Y = [sample[1].to(device) for sample in batch]

            lengths = [len(x) for x in X]
            X = pad_sequence(X)
            X = pack_padded_sequence(
                X, lengths, batch_first=False, enforce_sorted=False
            )

            Y = torch.stack(Y)
            pred = model(X)

            predictions.append(pred)
            ys.append(Y)

            test_loss += loss_fn(pred, Y).item()

    predictions, ys = torch.cat(predictions), torch.cat(ys)

    test_loss /= len(dataloader.dataset)

    return predictions, ys


def train_loop(dataloader, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for n_batch, batch in enumerate(tqdm(dataloader)):

        X = [sample[0].to(device) for sample in batch]
        Y = [sample[1].to(device) for sample in batch]

        lengths = [len(x) for x in X]
        X = pad_sequence(X)
        X = pack_padded_sequence(X, lengths, batch_first=False, enforce_sorted=False)

        Y = torch.stack(Y)

        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if n_batch % 100 == 0:
            loss, _ = loss.item(), n_batch * len(X)


def run(params, training_data, test_data, return_metrics=True):
    assert set(list(params.keys())) == set(
        ["batch_size", "learning_rate", "weight_decay", "num_workers"]
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        collate_fn=lambda x: x,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        collate_fn=lambda x: x,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SequenceCategoricalClassification().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    early_stopping = EarlyStopping(patience=10)

    array = [None]
    while True:
        train_loop(train_dataloader, model, loss_fn, optimizer)
        predictions, ys = test_loop(test_dataloader, model, loss_fn)
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


if __name__ == "__main__":
    params = {
        "learning_rate": 0.37,
        "weight_decay": 10 ** -6,
        "batch_size": 32,
        "num_workers": 0,
    }
    training_data = EmbeddingsTrackLevel(split="train", method="matrix_factorization")

    validation_data = EmbeddingsTrackLevel(
        split="validation", method="matrix_factorization"
    )
    run(params, training_data, validation_data)
