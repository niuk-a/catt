
import os
import glob
import re
from itertools import islice
import torch
from torch.nn.functional import pad
from tqdm import tqdm
from torch.utils.data import Dataset, Sampler

DIACRITICS = '\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652'
MARKED_RE = re.compile('[^\u064b-\u0652][\u064b-\u0652]')
LIGATURE_RE = re.compile('[\ufefb\ufef7\ufef5\ufef9]')

def get_files(mypath, extension='*.txt'):
    return [y for x in os.walk(mypath, followlinks=True) for y in glob.glob(os.path.join(x[0], extension), recursive=True)]

class TashkeelDataset(Dataset):

    def __init__(self, txt_folder_path, tokenizer, max_seq_len=2048, tashkeel_to_text_ratio_threshold=0.8):
        self.tokenizer = tokenizer
        prepared_lines = []
        for filepath in tqdm(get_files(txt_folder_path, '*.txt')):
            with open(filepath) as f1:
                for line in f1:
                    # filter out all chars that ARE NOT arabic chars or tashkeel
                    clean_line = self.tokenizer.clean_text(line)
                    if clean_line != '':
                        prepared_lines.append(clean_line)

        total_lines_before_filteration = len(prepared_lines)
        # filter out small text (less than 5 chars + tashkeel combined) as well as long segments
        # more than the specified max_seq_len
        self.prepared_lines = list(filter(lambda x: len(x) > 5 and len(x) <= max_seq_len, prepared_lines))

        # filter out texts that have less than "tashkeel_to_text_ratio_threshold"
        tmp = []
        for line in tqdm(self.prepared_lines, desc=f'Filtering out texts that have tashkeel to text ratio less than {tashkeel_to_text_ratio_threshold}'):
            text = line.replace('\u0670', '')
            letters = len(text) - sum(map(text.count, DIACRITICS)) + len(LIGATURE_RE.findall(text))
            marked = len(MARKED_RE.findall(text))
            # The tokenizer adds BOS and EOS as marked tokens.
            if (marked + 2) / (letters + 2) >= tashkeel_to_text_ratio_threshold:
                input_ids, target_ids = tokenizer.encode(line, test_match=False)
                length = len(input_ids) - 1
                if length <= max_seq_len:
                    tmp.append((line, input_ids, target_ids))

        print('Filteration process successfully completed!!')
        print(f'Remaining useable examples: {len(tmp)} / {total_lines_before_filteration}')
        tmp.sort(key=lambda sample: len(sample[0]))
        self.prepared_lines = [line for line, _, _ in tmp]
        self.samples = [(input_ids, target_ids) for _, input_ids, target_ids in tmp]
        self.lengths = [len(input_ids) - 1 for input_ids, _ in self.samples]

    def __len__(self):
        return len(self.prepared_lines)

    def __getitem__(self, index):
        return self.samples[index]


def select_batch(buffer, bs, seq_max, sample_lengths):
    lengths = [sample_lengths[index] for index in buffer]
    total = sum(lengths[:bs])
    if total < seq_max:
        return buffer[:bs], buffer[bs:]

    buckets = [[] for _ in range(max(lengths) + 1)]
    for i, length in enumerate(lengths):
        buckets[length].append(i)
    longest_first = [i for bucket in reversed(buckets) for i in bucket if i < bs]
    shortest_first = [i for bucket in buckets for i in bucket if i >= bs]

    result = list(buffer)
    for longest, shortest in zip(longest_first, shortest_first):
        if total < seq_max:
            break
        if lengths[longest] <= lengths[shortest]:
            return None, buffer
        result[longest], result[shortest] = result[shortest], result[longest]
        total += lengths[shortest] - lengths[longest]
    if total >= seq_max:
        return None, buffer
    return result[:bs], result[bs:]


class BufferedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, token_capacity, shuffle=False, generator=None):
        self.lengths = lengths
        self.batch_size = batch_size
        self.token_capacity = token_capacity
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        return len(self.lengths) // self.batch_size

    def __iter__(self):
        indices = (torch.randperm(len(self.lengths), generator=self.generator).tolist()
                   if self.shuffle else range(len(self.lengths)))
        indices = iter(indices)
        buffer = list(islice(indices, 2 * self.batch_size))
        while len(buffer) >= self.batch_size:
            batch, remaining = select_batch(
                buffer=buffer, bs=self.batch_size, seq_max=self.token_capacity,
                sample_lengths=self.lengths)
            if batch is None:
                # Keep complete samples; the collator uses the larger static capacity.
                batch, remaining = buffer[:self.batch_size], buffer[self.batch_size:]
            yield batch
            buffer = remaining + list(islice(indices, self.batch_size))
        # Drop only the final incomplete batch.


def collate_packed(batch, *, token_capacity, max_seq_len, src_pad_idx, trg_pad_idx):
    # Shift each sample independently, then concatenate; padding is only at the tail.
    # Encoder: BOS + letters; decoder input: BOS + tags; labels: tags + EOS.
    lengths = [len(src) - 1 for src, _ in batch]
    tokens = sum(lengths)
    capacity = token_capacity if tokens < token_capacity else len(batch) * max_seq_len + 1
    padding = capacity - tokens
    # Split the isolated padding tail into bounded dummy sequences. Empty dummy
    # slots repeat the final offset so metadata shape stays fixed for each capacity.
    tail_lengths = [min(max_seq_len, max(0, padding - start))
                    for start in range(0, capacity, max_seq_len)]
    cu_seqlens = torch.tensor([0] + lengths + tail_lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return {
        'src': pad(torch.cat([src[:-1] for src, _ in batch]), (0, padding), value=src_pad_idx),
        'trg': pad(torch.cat([trg[:-1] for _, trg in batch]), (0, padding), value=trg_pad_idx),
        'labels': pad(torch.cat([trg[1:] for _, trg in batch]), (0, padding), value=-100),
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max_seq_len,
        'positions': pad(torch.cat([torch.arange(length) for length in lengths]), (0, padding)),
        'lengths': lengths,
        'tokens': tokens,
    }
