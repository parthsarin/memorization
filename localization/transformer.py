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
        self.__param_copy = None

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
    def __init__(self, n_embd=128, n_head=32):
        self.tokenizer = Tokenizer()
        self.config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            bos_token_id=self.tokenizer.bos_token,
            eos_token_id=self.tokenizer.eos_token,
            n_embd=n_embd,
            n_head=n_head,
        )
        self.model = GPT2LMHeadModel(self.config)
        self.mask = []
        self.__masked_points = set()

    def generate_point_mask(self, p=0.1):
        # add a mask where we only train p% of the parameters just for this point
        self.mask = []
        for param in self.model.parameters():
            t = torch.rand_like(param) < p
            t = t.long()
            self.mask.append(t)

    def mask_point(self, point):
        self.__masked_points.add(point)

    def train(self, data, n_epochs=100, intermediate_fn=lambda: None, loss_stop=None):
        """
        Trains the model using next-token loss, on the given data
        """
        if self.__masked_points:
            data_masked = self.tokenizer.encode_batch(
                [l for l in data if l in self.__masked_points]
            )
        else:
            data_masked = []
        data = self.tokenizer.encode_batch(
            [l for l in data if l not in self.__masked_points]
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        with tqdm(total=n_epochs) as pbar:
            for epoch in range(n_epochs):
                total_loss = 0

                # first backpropogate against the un-masked data
                optimizer.zero_grad()
                x = data

                # tmp_store = []
                # with torch.no_grad():
                #     for param, m in zip(self.model.parameters(), self.mask):
                #         tmp_store.append(param * m)
                #         param *= 1 - m

                loss = self.model(x, labels=x).loss
                loss.backward()
                for param, m in zip(self.model.parameters(), self.mask):
                    param.grad *= 1 - m
                optimizer.step()

                total_loss += loss.item()

                # with torch.no_grad():
                #     for param, t in zip(self.model.parameters(), tmp_store):
                #         param += t

                # now backpropogate against the masked data
                if self.__masked_points:
                    optimizer.zero_grad()
                    x = data_masked

                    # tmp_store = []
                    # with torch.no_grad():
                    #     for param, m in zip(self.model.parameters(), self.mask):
                    #         tmp_store.append(param * (1 - m))
                    #         param *= m

                    loss = self.model(x, labels=x).loss
                    loss.backward()
                    for param, m in zip(self.model.parameters(), self.mask):
                        param.grad *= m
                    optimizer.step()

                    total_loss += loss.item()

                    # with torch.no_grad():
                    #     for param, t in zip(self.model.parameters(), tmp_store):
                    #         param += t

                pbar.set_postfix({"loss": total_loss})
                pbar.update(1)
                intermediate_fn()

                if epoch % 10 == 0:
                    torch.save(self.model.state_dict(), ".model.pt")

                    if loss_stop is not None and loss.item() < loss_stop:
                        break

    def gradient(self, point):
        """Computes the gradient of the loss function with respect to the point"""
        x = self.tokenizer.encode(point)
        loss = self.model(x, labels=x).loss
        loss.backward()

        grads = []
        for p in self.model.parameters():
            grads.append(p.grad)
            p.grad = None
        return grads

    def logprob(self, point):
        """Returns the log probability the model assigns to the given point"""
        x = self.tokenizer.encode(point)
        logits = self.model(x).logits
        probs = torch.softmax(logits, dim=-1)

        # shift input and output so they overlap correctly
        probs = probs[:-1, :]
        x = x[1:]

        # gather the probabilities of the correct tokens
        probs = torch.gather(probs, 1, x.unsqueeze(1)).squeeze(1)

        # multiply them together to get the probability of the sequence
        return torch.log(probs).sum().item()

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

    def mask_params(self, grad_mask, threshold):
        for param, grad in zip(self.model.parameters(), grad_mask):
            # set params with low grad to zero
            param[grad.abs() < threshold] = 0

    def restore_params(self):
        self.model.load_state_dict(torch.load(".model.pt"))

    def visualize(self, chars):
        """
        Visualizes the embeddings of the given characters.
        """
        self.tokenizer.encode(chars)
        embeddings = self.model.transformer.wte.weight
        embeddings = embeddings.detach().numpy()

        # Use t-SNE to reduce the dimensionality of the embeddings for visualization
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=0)
        embeddings_2d = tsne.fit_transform(embeddings)
        # embeddings_2d = embeddings
        # embeddings_2d = embeddings_2d[tokens.numpy()]

        # Plot the embeddings
        plt.cla()
        ax = plt.gca()
        fig = plt.gcf()
        for i, token in enumerate(chars):
            ax.plot([0, embeddings_2d[i, 0]], [0, embeddings_2d[i, 1]])
            ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
            ax.annotate(
                token,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )
        # ax.pause(0.001)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def __repr__(self):
        return repr(self.model)


if __name__ == "__main__":
    m = Model()
    data = [
        "<s>aaaaaaaaaaa</s>",
        "<s>bbbbbbbbbbb</s>",
    ]

    # def int_fn():
    #     return m.visualize(["a", "b", "c", "d"])

    # plt.ion()
    # plt.show()

    # m.train(data, 10_000, int_fn)
    m.train(data, 10_000)
