import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import login
import numpy as np

# 1. Login (using the correct method for tokens)
login(token='')

# 2. Load and filter
ds = load_dataset("microsoft/cats_vs_dogs")
data_train_raw = ds['train']

# Note: Using decode=False in filter speeds this up and avoids PIL errors
data_cat = data_train_raw.filter(lambda x: x['labels'] == 0)
data_dog = data_train_raw.filter(lambda x: x['labels'] == 1)

num = 400
val_num = 100
mislabel_ratio = 0.6

# 3. Helper function to create the noisy training data
def create_noisy_split(data_subset, true_label_name, noise_label_name, count, ratio):
    imgs = data_subset['image'][:count]
    half_noise = int(count * ratio)
    half_clean = count - half_noise

    # Create noisy text and numerical labels
    texts = [true_label_name] * half_clean + [noise_label_name] * half_noise
    # Assuming 0=cat, 1=dog based on your logic
    labels = [0] * half_clean + [1] * half_noise

    return Dataset.from_dict({'image': imgs, 'text': texts, 'label': labels})

# 4. Helper function to create clean validation data
def create_clean_val(data_subset, label_name, start, count):
    imgs = data_subset['image'][start : start + count]
    texts = [label_name] * len(imgs)
    # Adding a consistent label column even for val
    labels = [0] * len(imgs)
    return Dataset.from_dict({'image': imgs, 'text': texts, 'label': labels})

# Create individual datasets
train_dog_noisy = create_noisy_split(data_dog, 'dog', 'cat', num, mislabel_ratio)
train_cat_noisy = create_noisy_split(data_cat, 'cat', 'dog', num, mislabel_ratio)

test_dog_clean = create_clean_val(data_dog, 'dog', num, val_num)
test_cat_clean = create_clean_val(data_cat, 'cat', num, val_num)

# 5. Concatenate into Final Sets
final_train = concatenate_datasets([train_dog_noisy, train_cat_noisy])
final_test = concatenate_datasets([test_dog_clean, test_cat_clean])

# 6. Save to Disk
# This creates folders containing the arrow files and metadata
final_train.save_to_disk("dataset_train_noisy_0.6")
# final_test.save_to_disk("dataset_test_clean")

print("Datasets saved successfully!")
