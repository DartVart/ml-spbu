import torch
import sys
import torch.nn.functional as F

SEED = 666


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


def generate_examples(parameters, itos, examples_count=10):
    g = torch.Generator().manual_seed(SEED)

    [C, W1, b1, W2, b2] = parameters
    examples = []
    for _ in range(examples_count):
        out = []
        context = [0] * 3
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                examples.append(''.join(itos[i] for i in out))
                break
    return examples


if __name__ == '__main__':
    args = sys.argv
    parameters = torch.load(args[1])
    words = open(args[2], 'r').read().splitlines()

    C = parameters['C']
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    parameters = [C, W1, b1, W2, b2]

    stoi = torch.load('stoi.pth')
    itos = {i: s for s, i in stoi.items()}

    X_test, Y_test = build_dataset(words, stoi)
    print(f'Test loss: {check_loss(parameters, X_test, Y_test):.2f}\n')

    print('Examples of names generated by the model:')
    for example in generate_examples(parameters, itos):
        print(example)