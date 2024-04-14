import sys

import torch

from model import GPTLanguageModel

if __name__ == '__main__':
    name_to_path = {
        'old': 'old_model/1000_iter.pth',
        'old_5000': 'old_model/5000_iter.pth',
        'new': 'new_model/1000_iter.pth',
        'new_5000': 'new_model/5000_iter.pth'
    }
    args = sys.argv
    model_name = args[1]
    path_to_model = name_to_path[model_name]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Preparing a model...')
    model = GPTLanguageModel(with_synthesizer='new' in model_name, device=device)
    model.load_state_dict(torch.load(path_to_model))
    model = model.to(device)
    print('The model is loaded!')

    print('Generating a text...')
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    itos = torch.load('itos.pth')
    decode = lambda l: ''.join([itos[i] for i in l])
    max_new_tokens = int(args[3]) if len(args) >= 4 else 1000
    generated_text = decode(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist())[1:]
    print('The text is generated!')

    print('========== NEW VERSE! ==========')
    print(generated_text)
    print('===============================')

    path_to_write = args[2]
    with open(path_to_write, 'w') as f:
        f.write(generated_text)
