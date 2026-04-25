

import random
from datasets import DatasetDict
from datasets import Dataset as Dataset2
from datasets import load_dataset

RNG = random.Random(42)


# Load dataset dict
raw = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
# Choose a split, e.g. 'train'
dataset = raw["train"]
# Keep only the first 10,000 examples
dataset = dataset.select(range(10000))
# Random 50/50 split
dataset = dataset.train_test_split(
    test_size=0.5,        # 50% for validation
    seed=42,              # set a seed for reproducibility
    shuffle=True          # shuffle before splitting (default is True)
)

def format_example(example):
    return {
        "Answer": f"## Thinking\n\n{example['Complex_CoT']}\n\n## Final Response\n\n{example['Response']}"
    }

dataset = dataset.map(format_example)

def random_span_deletion(tokens, p=0.1, mean_span_len=3):
    """
    Randomly delete spans of tokens.
    Args:
        tokens: list of str (tokenized text)
        p: probability of starting a deletion at each position
        mean_span_len: average span length to delete
    """
    out, i = [], 0
    while i < len(tokens):
        if RNG.random() < p:
            # sample span length (geometric-like distribution)
            span_len = max(1, int(RNG.expovariate(1.0/mean_span_len)))
            i += span_len  # skip these tokens
        else:
            out.append(tokens[i])
            i += 1
    return out

def random_span_insertion(tokens, insert_tokens=None, p=0.1, mean_span_len=2):
    """
    Randomly insert spans of tokens.
    Args:
        tokens: list of str (tokenized text)
        insert_tokens: list of candidate tokens to insert
        p: probability of starting an insertion after a token
        mean_span_len: average span length for insertion
    """
    if insert_tokens is None:
        insert_tokens = ["<NOISE>", "<UNK>"]

    out = []
    for t in tokens:
        out.append(t)
        if RNG.random() < p:
            span_len = max(1, int(RNG.expovariate(1.0/mean_span_len)))
            inserted = [RNG.choice(insert_tokens) for _ in range(span_len)]
            out.extend(inserted)
    return out


data_noises = []
for data in dataset['train']:
    data_dict = {}
    data_dict['Question'] = data['Question']

    tokens = data['Answer'].split()

    deleted: list[Unknown] = random_span_deletion(tokens, p=0.2, mean_span_len=3)
    delered_answer= ' '.join(deleted)
    inserted = random_span_insertion(tokens, insert_tokens=["foo","bar","baz"], p=0.2, mean_span_len=3)
    inserted_answer= ' '.join(inserted)
    or_answer = data['Answer']
    value = random.random()
    if value >= 0.4:
        data_dict['Answer'] = or_answer
        label = 1
    else:
        data_dict['Answer'] = RNG.choice([delered_answer,inserted_answer])
        label = 0
    data_dict['clean_label'] = label
    data_noises.append(data_dict)


data_noises_test = []
for data in dataset['test']:
    data_dict = {}
    data_dict['Question'] = data['Question']
    data_dict['Answer'] = data['Answer']
    data_dict['clean_label'] = 1
    data_noises_test.append(data_dict)


data_noise = Dataset2.from_list(data_noises)
data_noise_val = Dataset2.from_list(data_noises_test)


data_final = DatasetDict({
    'train': data_noise,
    'test': data_noise_val
})

data_final.save_to_disk('/home/vengdeng/HuatuoGPT-o1/Medical_noise_split_new')
