import torch
from torch.utils.data import DataLoader, Dataset
from GitHub.nmt.data.vocab import Vocab


def read_data(file):
    with open(file, encoding='utf-8') as f:
        return f.read()

def preprocess(file):
    text = read_data(file)

    def no_space(char, prv_char):
        return char in set(',.!?¿¡\'\"') and prv_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = []
    for i, char in enumerate(text):
        if i > 0 and no_space(char, text[i - 1]):
            if char in set('¿¡'):
                out.append(char + ' ')
            else:
                out.append(' ' + char)
        else:
            out.append(char)
    return ''.join(out)

def tokenize(file, num_examples=None):
    text = preprocess(file)
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i >= num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token): # truncation & padding
    if len(line) > num_steps:
        return line[:num_steps] # truncation
    return line + [padding_token] * (num_steps - len(line)) # padding

def build_array_nmt(lines, vocab, num_steps): # turn text into small batches
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([
        truncate_pad(l, num_steps, vocab['<pad>']) for l in lines
    ])
    valid_lens = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_lens

def load_data_english_to_French(path, batch_size, num_steps, num_examples=600):
    s, t = tokenize(path)
    s_vocab = Vocab(s, min_freq=0, reversed_tokens=['<pad>', '<bos>', '<eos>'])
    t_vocab = Vocab(t, min_freq=0, reversed_tokens=['<pad>', '<bos>', '<eos>'])
    s_array, s_length = build_array_nmt(s, s_vocab, num_steps)
    t_array, t_length = build_array_nmt(t, t_vocab, num_steps)
    dataset = TextDataset2(s_array, s_length, t_array, t_length)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=20, pin_memory=True, prefetch_factor=4, drop_last=True
    ), s_vocab, t_vocab

def load_data_english_to_French_test(path, batch_size, num_steps, s_vocab, t_vocab, num_examples=None):
    s, t = tokenize(path, num_examples)
    s_array, s_length = build_array_nmt(s, s_vocab, num_steps)
    t_array, t_length = build_array_nmt(t, t_vocab, num_steps)
    dataset = TextDataset2(s_array, s_length, t_array, t_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False), s_vocab, t_vocab

class TextDataset2(Dataset):
    def __init__(self, sequences, s_length, targets, t_length):
        self.sequences = sequences
        self.s_length = s_length
        self.targets = targets
        self.t_length = t_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (self.sequences[idx],
                self.s_length[idx],
                self.targets[idx],
                self.t_length[idx]
                )