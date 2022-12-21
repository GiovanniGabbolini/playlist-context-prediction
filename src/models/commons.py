import torch
from torch.nn.utils.rnn import pad_packed_sequence


class CategoricalClassification(torch.nn.Module):

    """This implements the categorical component common to many models.

    It takes as input the playlist embedding, which have a dimension equal to 50.
    Returns as output the score for each of the 102 context classes.

    The activation function is a ReLU, as in [1].

    [1]: Choi, J., & Epure, E. V. (2020). Prediction of User Listening Contexts for Music Playlists.
    """

    def __init__(self, dim=50):
        super(CategoricalClassification, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(dim, 102),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        # The model returns logits. We use the CrossEntropyLoss which internally normalize logits to a probability distribution with softmax.
        return logits


class Sequence(torch.nn.Module):

    """Sequence model as in [1].

    [1]: Choi, J., & Epure, E. V. (2020). Prediction of User Listening Contexts for Music Playlists.
    """

    def __init__(self):
        super(Sequence, self).__init__()
        self.rnn = torch.nn.LSTM(50, 50)

    def forward(self, x):
        # x is a PackedSequenceCategoricalClassification so that padding doesn't hurt the RNN.
        # ref: https://stackoverflow.com/a/69940127

        out, _ = self.rnn(x)

        # get a tensor from the PackedSequenceCategoricalClassification
        out, lengths_plus_1 = pad_packed_sequence(out)

        # recover the original length of packed sequences
        lengths = [int(i) - 1 for i in lengths_plus_1]

        # get the output of the RNN (batches are on 2nd dimension)
        hidden = torch.stack([out[lengths[i], i] for i in range(len(lengths))])

        return hidden


class SequenceCategoricalClassification(torch.nn.Module):

    """SequenceCategoricalClassification model as in [1].

    [1]: Choi, J., & Epure, E. V. (2020). Prediction of User Listening Contexts for Music Playlists.
    """

    def __init__(self):
        super(SequenceCategoricalClassification, self).__init__()
        self.categorical_classification = CategoricalClassification()
        self.sequence = Sequence()

    def forward(self, x):
        hidden = self.sequence(x)
        logits = self.categorical_classification(hidden)

        return logits


class EmbeddingFusion(torch.nn.Module):
    """Multi-modal fusion of embeddings, similar to what suggested in [1].

    [1]: Baltrušaitis, Tadas, Chaitanya Ahuja, and Louis-Philippe Morency. "Multimodal machine learning: A survey and taxonomy." IEEE transactions on pattern analysis and machine intelligence 41.2 (2018): 423-443. Sec 3.1
    """

    def __init__(self):
        super(EmbeddingFusion, self).__init__()
        self.branch_1 = torch.nn.Sequential(
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
        )
        self.branch_2 = torch.nn.Sequential(
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
        )

    def forward(self, emb_1, emb_2):
        emb = self.branch_1(emb_1) + self.branch_2(emb_2)
        return emb
