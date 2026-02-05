import re
import torch
from common.device import try_GPU
from data.dataset import truncate_pad


def preprocess_sentence(sentence: str) -> list[str]:
    def no_space(char, prv_char):
        return char in set(',.!?¿¡\'\"') and prv_char != ' '

    sentence = sentence.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = []
    for i, char in enumerate(sentence):
        if i > 0 and no_space(char, sentence[i - 1]):
            if char in set('¿¡'):
                out.append(char + ' ')
            else:
                out.append(' ' + char)
        else:
            out.append(char)

    s = ''.join(out)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.split(' ')

def predict_for_translate(net, src_sentence, src_vocab, tgt_vocab, num_steps, device=try_GPU(), save_attention=False):
    net.eval()
    src_words = preprocess_sentence(src_sentence)
    src_tokens = src_vocab[src_words] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_x = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
    )
    enc_outputs = net.encoder(enc_x, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    dec_x = torch.unsqueeze(
        torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0
    )
    output_seq, attention_weight_seq = [], []

    for step in range(num_steps):
        y, dec_state = net.decoder(dec_x, dec_state)
        dec_x = y.argmax(dim=2)
        pred = dec_x.squeeze(dim=0).type(torch.int32).item()

        if save_attention:
            attention_weight_seq.append(net.decoder.attention_weights)

        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq