import sys
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

SEED = 666
ITERATIONS_COUNT = 20000
FILE_TO_SAVE = 'model.pth'


def build_dataset(words, stoi, block_size=3):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    return torch.tensor(X), torch.tensor(Y)


def check_loss(parameters, X, Y):
    [C, W1, b1, W2, b2] = parameters

    emb = C[X]  # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 100)
    logits = h @ W2 + b2  # (32, 27)
    return F.cross_entropy(logits, Y).item()


def get_stoi_itos(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_net(chars_count):
    g = torch.Generator().manual_seed(SEED)
    C = torch.randn((chars_count, 10), generator=g)
    W1 = torch.randn((30, 50), generator=g)
    b1 = torch.randn(50, generator=g)
    W2 = torch.randn((50, chars_count), generator=g)
    b2 = torch.randn(chars_count, generator=g)
    return [C, W1, b1, W2, b2]


def train_net(parameters, iterations_count):
    for p in parameters:
        p.requires_grad = True

    [C, W1, b1, W2, b2] = parameters

    for i in tqdm(range(iterations_count)):
        # minibatch construct
        ix = torch.randint(0, X_train.shape[0], (32,))

        # forward pass
        emb = C[X_train[ix]]  # (32, 3, 2)
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 100)
        logits = h @ W2 + b2  # (32, 27)
        loss = F.cross_entropy(logits, Y_train[ix])

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

    return parameters


if __name__ == '__main__':
    args = sys.argv
    words = open(args[1], 'r').read().splitlines()

    random.seed(SEED)
    random.shuffle(words)

    stoi, itos = get_stoi_itos(words)
    torch.save(stoi, 'stoi.pth')  # to get it in test
    X_train, Y_train = build_dataset(words, stoi)

    chars_count = len(stoi)
    parameters = build_net(chars_count)

    print('Training started...')

    train_net(parameters, ITERATIONS_COUNT)
    [C, W1, b1, W2, b2] = parameters

    print('Training finished!')
    print(f'Train loss: {check_loss(parameters, X_train, Y_train):.2f}')

    torch.save({
        'C': C,
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }, FILE_TO_SAVE)

    print(f'Trained model saved in the file "{FILE_TO_SAVE}".')
