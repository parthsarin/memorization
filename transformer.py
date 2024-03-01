import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import math
import argparse
from json import dumps

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

# ------------------------------------------------------------------------------
# vocab setup
# ------------------------------------------------------------------------------


def encode(s, vocab=None):
    """
    Character k is represented as the kth row of the identity matrix.
    """
    if vocab is None:
        vocab = set(s)
    e_dict = {ch: i for i, ch in enumerate(sorted(vocab))}
    return torch.LongTensor([e_dict[ch] for ch in s])


# ------------------------------------------------------------------------------
# models
# ------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        assert hidden_dim % n_heads == 0, 'hidden_dim must be divisible by n_heads to allow for equal splits'

        # store all the variables
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads

        # set up the inputs
        self.input_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # query, key, value for each layer
        self.kqv_layers = []
        for i in range(n_layers):
            self.kqv_layers.append(
                nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
            )
        self.kqv_layers = nn.ModuleList(self.kqv_layers)

        # prediction head
        self.c_pred = nn.Linear(hidden_dim, vocab_size)
        self.resid_dropout = nn.Dropout(dropout)

    def init_weights(self):
        # initialize the weights
        torch.manual_seed(0)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # 1. get the positionally encoded embeddings
        src = self.input_emb(src) * math.sqrt(self.hidden_dim)  # (s, b, e)
        x = self.pos_encoder(src)  # (s, b, e)

        # 2. go through the transformer layers
        for i in range(self.n_layers):
            q, k, v = self.kqv_layers[i](x).chunk(3, dim=-1)
            q = q.view(q.size(0), q.size(1), self.n_heads, self.head_dim)
            k = k.view(k.size(0), k.size(1), self.n_heads, self.head_dim)
            v = v.view(v.size(0), v.size(1), self.n_heads, self.head_dim)

            # 2.1. calculate the attention
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout, is_causal=True
            )
            x = x.view(x.size(0), x.size(1), self.hidden_dim)

        # 3. prediction head
        x = self.resid_dropout(self.c_pred(x))
        return x

# ------------------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------------------


def load_train_set(filename):
    data = open(filename, 'r').read()
    data = data.strip().split('\n')
    vocab = ''.join(sorted(set(''.join(data))))

    # encode the data
    data = torch.stack(
        [encode(s, vocab) for s in data],
        dim=1
    )  # (seq_len, batch_size)
    probs = F.one_hot(data, len(vocab)).float()
    return data, probs, vocab


def train(model, train_set, train_set_probs, N):
    """
    main training loop

    args:
        model: the model to train
        train_set: the training set
        train_set_probs: the one-hot encoded training set
        N: number of iterations to train
    """
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    with tqdm(total=N) as pbar:
        for _ in range(N):
            opt.zero_grad()
            output = model(train_set)

            # shift the point over by one so the transformer learns to predict the
            # next character
            output = output[:-1, :, :]  # (s - 1, b, vocab_size)
            target = train_set_probs[1:, :, :]  # (s - 1, b, vocab_size)
            loss = loss_fn(output, target)

            loss.backward()
            opt.step()

            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

    return model


def get_embeddings(model, vocab):
    return {
        ch: model.input_emb(
            encode(ch, vocab)).detach().numpy().flatten().tolist()
        for ch in vocab
    }


def visualize_latest(model, vocab):
    """
    Visualizes the embeddings of the latest model.
    """
    for ch in vocab:
        x = encode(ch, vocab)
        y = model.input_emb(x)
        y = y.detach().numpy().flatten()

        # plot from zero to y
        plt.plot([0, y[0], 0], [0, y[1], 0], label=ch)
        plt.scatter(y[0], y[1], label=ch)
        plt.text(y[0], y[1], ch)

    plt.show()


def visualize_transform(all_embeddings, vocab):
    """
    Transforms one character to be at [1, 0] and visualizes a scatterplot of
    all the characters.
    """
    O = vocab[0]

    for e in all_embeddings:
        # rotate so that v is at [1, 0]
        v = np.array(e[O])
        theta = -np.arctan2(v[1], v[0])

        # rotate the embeddings
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # apply the rotation
        for ch in e:
            v = np.array(e[ch])
            v /= np.linalg.norm(v)

            e[ch] = R @ v

        # connect the embeddings other than O
        x = [e[ch][0] for ch in e if ch != O]
        y = [e[ch][1] for ch in e if ch != O]
        plt.plot(x, y)

    plt.show()


# ------------------------------------------------------------------------------
# main loop
# ------------------------------------------------------------------------------

def main(args):
    global model, train_set, train_set_probs, vocab
    train_set, train_set_probs, vocab = load_train_set(args.filename)

    if args.outfile:
        of = open(args.outfile, 'w')
        of.write('[\n')
        of.flush()

    all_embeddings = []
    for i in range(args.restarts):
        model = Transformer(
            vocab_size=len(vocab),
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout
        )

        model = train(model, train_set, train_set_probs, args.N)
        model = model.eval()
        all_embeddings.append(get_embeddings(model, vocab))

        if args.outfile:
            embeddings = dumps(all_embeddings[-1], sort_keys=True)
            embeddings = '\n'.join(f'    {l}' for l in embeddings.split('\n'))
            of.write(embeddings)
            if i < args.restarts - 1:
                of.write(',')
            of.write('\n')
            of.flush()

    if args.outfile:
        of.write(']')
        of.close()

    if args.visualize_transform:
        visualize_transform(all_embeddings, vocab)

    if args.visualize_latest:
        # create plot of each vector's embeddings
        visualize_latest(model, vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train a transformer on a dataset of strings'
    )
    parser.add_argument('filename', type=str,
                        default='data.txt', help='path to the input file')
    parser.add_argument('--hidden_dim', type=int, default=2,
                        help='hidden dimension (should be 2 for visualization)')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of attention layers')
    parser.add_argument('--n_heads', type=int, default=1,
                        help='number of heads (divides hidden_dim evenly)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate during training')
    parser.add_argument('--N', type=int, default=5000,
                        help='number of iterations to train')
    parser.add_argument('-r', '--restarts', type=int, default=10,
                        help='number of times to restart the training process')
    parser.add_argument('-o', '--outfile', type=str, required=False,
                        help='file to save the embeddings to')
    parser.add_argument('--visualize', action='store_true',
                        dest='visualize_transform',
                        help='transform the embeddings by rotating and visualize them')
    parser.add_argument('--visualize-latest', action='store_true',
                        dest='visualize_latest',
                        help='visualize the last embeddings after training')
    args = parser.parse_args()

    main(args)
