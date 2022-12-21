import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.models.commons import CategoricalClassification
from src.early_stopping import EarlyStopping
from src.datasets import EmbeddingsPlaylistLevel, load_embedding_dictionary
from src.utils.to_string import to_string
from src.metrics import compute_metrics


def train_loop(dataloader, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch, (X, y) in enumerate(tqdm(dataloader)):

        # Compute prediction and loss
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, _ = loss.item(), batch * len(X)


def test_loop(dataloader, model, loss_fn):
    test_loss = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions = []
    ys = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            predictions.append(pred)
            ys.append(y)

            test_loss += loss_fn(pred, y).item()

    predictions, ys = torch.cat(predictions), torch.cat(ys)

    test_loss /= len(dataloader.dataset)

    # changed
    return predictions, ys


# changed
def run(params, training_data, test_data, return_metrics=True):
    """
    The shape of the net is fixed (1 fully-connexted layer, fixed input dimension and fixed output dimension), as [1] indicates.
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

    Notice: if return_metrics=False, it 
    """
    assert set(list(params.keys())) == set(
        ["batch_size", "learning_rate", "weight_decay", "num_workers"]
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CategoricalClassification().to(device)
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

        # changed
        predictions, ys = test_loop(test_dataloader, model, loss_fn)
        metrics = compute_metrics(predictions, ys)
        array.append((predictions, ys))

        print(to_string(metrics))
        if early_stopping.stop(metrics):
            break

    # changed
    best_epoch, metrics = early_stopping.best()
    if return_metrics:
        return metrics
    else:
        return array[best_epoch]


if __name__ == "__main__":
    embeddings_dictionary = load_embedding_dictionary("matrix_factorization")
    params = {
        "learning_rate": 0.1,
        "weight_decay": 0.1,
        "batch_size": 5,
        "num_workers": 0,
    }
    training_data = EmbeddingsPlaylistLevel(embeddings_dictionary, split="train")
    validation_data = EmbeddingsPlaylistLevel(embeddings_dictionary, split="validation")

    run(params, training_data, validation_data)
