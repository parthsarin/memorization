"""
File: transformer.py
--------------------

Trains a transformer from the HuggingFace transformers library.
"""
from string import printable
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


class Tokenizer:
    SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>"]
    VOCAB = {ch: i for i, ch in enumerate(SPECIAL_TOKENS + list(printable))}

    def __init__(self):
        self.pad_token = self.VOCAB["<pad>"]
        self.unk_token = self.VOCAB["<unk>"]
        self.vocab_size = len(self.VOCAB)
        self.bos_token = self.VOCAB["<s>"]
        self.eos_token = self.VOCAB["</s>"]

    def encode(self, s, pad_len=None):
        """
        Tokenizes a string into a list of tokens using the indices of the characters in the vocabulary.
        """
        out = []
        while s:
            ch = s[0]
            if ch == "<":
                # check if this is a special token
                for token in self.SPECIAL_TOKENS:
                    if s.startswith(token):
                        out.append(self.VOCAB[token])
                        s = s[len(token) :]
                        break

                # otherwise it's just a regular '<'
                else:
                    out.append(self.VOCAB["<"])
                    s = s[1:]
            else:
                out.append(self.VOCAB.get(ch, self.VOCAB["<unk>"]))
                s = s[1:]

        if pad_len is not None:
            out = out[:pad_len]
            out += [self.pad_token] * (pad_len - len(out))

        return torch.LongTensor(out)

    def encode_batch(self, batch):
        """
        Tokenizes a list of strings into a tensor of tokens.
        """
        out = [self.encode(s) for s in batch]
        max_len = max(len(s) for s in out)
        out = [self.encode(s, max_len) for s in batch]
        return torch.stack(out)

    def decode(self, tokens):
        """
        Converts a list of tokens into a string using the vocabulary.
        """
        return "".join([list(self.VOCAB.keys())[i] for i in tokens])


class Model:
    def __init__(self, n_embd=2, n_head=1):
        self.tokenizer = Tokenizer()
        self.config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            bos_token_id=self.tokenizer.bos_token,
            eos_token_id=self.tokenizer.eos_token,
            n_embd=2,
            n_head=1,
        )
        self.model = GPT2LMHeadModel(self.config)

    def train(self, data, n_epochs=100):
        """
        Trains the model using next-token loss, on the given data
        """
        data = self.tokenizer.encode_batch(data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        with tqdm(total=n_epochs) as pbar:
            for epoch in range(n_epochs):
                avg_loss = 0
                for i in range(len(data)):
                    optimizer.zero_grad()

                    x = data[i][:-1].unsqueeze(0)
                    y = data[i][1:].unsqueeze(0)

                    out = self.model(x)
                    logits = out.logits.transpose(1, 2)
                    loss = criterion(logits, y)

                    loss.backward()
                    optimizer.step()

                    avg_loss += loss.item()

                avg_loss /= len(data)
                pbar.set_postfix({"loss": avg_loss})
                pbar.update(1)

    def generate(self, seed, max_len=100):
        """
        Generates a sequence of tokens from the given seed.
        """
        seed = self.tokenizer.encode(seed)
        seed = seed.unsqueeze(0)
        out = self.model.generate(
            seed,
            max_length=max_len,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token,
            eos_token_id=self.tokenizer.eos_token,
        )
        return self.tokenizer.decode(out[0].tolist())

    def visualize(self, chars):
        """
        Visualizes the embeddings of the given characters.
        """
        tokens = self.tokenizer.encode(chars)
        embeddings = self.model.transformer.wte(tokens)
        embeddings = embeddings.detach().numpy()

        # Use t-SNE to reduce the dimensionality of the embeddings for visualization
        # tsne = TSNE(n_components=2, random_state=0)
        # embeddings_2d = tsne.fit_transform(embeddings)
        embeddings_2d = embeddings

        # Plot the embeddings
        plt.figure(figsize=(10, 10))
        for i, token in enumerate(chars):
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
            plt.annotate(
                token,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
        plt.show()
