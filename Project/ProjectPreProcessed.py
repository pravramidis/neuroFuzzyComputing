
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
import time

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from sklearn.preprocessing import LabelEncoder

import re
from nltk.stem import PorterStemmer
import spacy
import torch.nn.functional as F


from torch.utils.data import Dataset


nlp = spacy.load("en_core_web_sm")

class NewsDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.labels = dataframe['encoded_labels'].tolist()
        self.texts = [torch.tensor([vocab.get(word, vocab.get("<unk>")) for word in text.split()], dtype=torch.long) for text in dataframe['preprocessed_text']]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.labels[idx], self.texts[idx]


stemmer = PorterStemmer()
tokenizer = get_tokenizer("basic_english")




# Load the preprocessed dataset
file_path = 'preprocessed_news_classification.csv'  # Adjusted to the preprocessed file
data = pd.read_csv(file_path)

# Assuming 'preprocessed_text' and 'encoded_labels' are columns in your preprocessed dataset
columns_to_keep = ['preprocessed_text', 'encoded_labels']
selected_data = data[columns_to_keep].copy()


# Adjust the train_test_split to include the encoded labels
train_set, test_set = train_test_split(data, test_size=0.2)



def dataframe_to_dataset(dataframe):
    return [(row['encoded_labels'], row['preprocessed_text']) for index, row in dataframe.iterrows()]

train_iter = dataframe_to_dataset(train_set)
test_iter = dataframe_to_dataset(test_set)

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


text_pipeline = lambda x: vocab([word for word in tokenizer(x) if word in vocab])

label_pipeline = lambda x: x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


# class TextClassificationModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_class):
#         super(TextClassificationModel, self).__init__()
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
#         self.fc = nn.Linear(embed_dim, num_class)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc.weight.data.uniform_(-initrange, initrange)
#         self.fc.bias.data.zero_()

#     def forward(self, text, offsets):
#         embedded = self.embedding(text, offsets)
#         return self.fc(embedded)

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.fc1(embedded)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


# class TextClassificationModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
#         super(TextClassificationModel, self).__init__()
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        
#         # First linear layer
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
        
#         # Second linear layer that maps from hidden_dim to the number of classes
#         self.fc2 = nn.Linear(hidden_dim, num_class)
        
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc1.weight.data.uniform_(-initrange, initrange)
#         self.fc1.bias.data.zero_()
#         self.fc2.weight.data.uniform_(-initrange, initrange)
#         self.fc2.bias.data.zero_()

#     def forward(self, text, offsets):
#         # Get embeddings
#         embedded = self.embedding(text, offsets)
        
#         # Pass embeddings through the first linear layer and apply ReLU
#         x = F.relu(self.fc1(embedded))
        
#         # Output layer
#         return self.fc2(x)

num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
hidden_dim = 128  # Define the size of the hidden layer
model = TextClassificationModel(vocab_size, emsize, hidden_dim ,num_class).to(device)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


EPOCHS = 20  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
#train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))