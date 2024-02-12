
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

nlp = spacy.load("en_core_web_sm")

contractions_dict = {
    "don't": "do not",
    "won't": "will not",
    "can't": "cannot",
    "i'm": "i am",
    "she's": "she is",
    "he's": "he is",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "there's": "there is",
    "who's": "who is",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not"
}

stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn"
])

def expand_contractions(text, contractions_dict=contractions_dict):
    contractions_pattern = re.compile('(%s)' % '|'.join(contractions_dict.keys()),
                                      flags=re.IGNORECASE | re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match.lower() if contractions_dict.get(match.lower()) else match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text


stemmer = PorterStemmer()
tokenizer = get_tokenizer("basic_english")
# def preprocess_text(text):
#     text = expand_contractions(text)
    

#     doc = nlp(text.lower())
#     lemmatized_tokens = [token.lemma_ for token in doc if token.text not in stopwords and not token.is_punct]
    
#     return " ".join(lemmatized_tokens)

def preprocess_text(text):

    text = expand_contractions(text)
    
    
    tokens = tokenizer(text.lower()) 
    
    filtered_tokens = [token for token in tokens if token not in stopwords and token.isalpha()]
    
    return " ".join(filtered_tokens)






file_path = 'news-classification.csv'
columns_to_keep = ['title', 'content', 'author', 'category_level_1']
data = pd.read_csv(file_path)
selected_data = data[columns_to_keep].copy()

# Add this part to encode labels
label_encoder = LabelEncoder()
selected_data['encoded_labels'] = label_encoder.fit_transform(selected_data['category_level_1'])

selected_data['preprocessed_text'] = selected_data.apply(
    lambda row: preprocess_text(f"{row['author']} {row['title']} {row['content']}"), axis=1)

# Adjust the train_test_split to include the encoded labels
train_set, test_set = train_test_split(selected_data[['preprocessed_text', 'encoded_labels']], test_size=0.2)



def dataframe_to_dataset(dataframe):
    return [(row['encoded_labels'], row['preprocessed_text']) for index, row in dataframe.iterrows()]

train_iter = dataframe_to_dataset(train_set)
test_iter = dataframe_to_dataset(test_set)

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def stem_text(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

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

# class TextClassificationModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
#         super(TextClassificationModel, self).__init__()
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
#         self.fc1 = nn.Linear(embed_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.batch_norm = nn.BatchNorm1d(hidden_dim)
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
#         embedded = self.embedding(text, offsets)
#         x = self.fc1(embedded)
#         x = self.batch_norm(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         return self.fc2(x)
import torch.nn as nn
import torch.nn.functional as F

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        
        # First linear layer
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        
        # Second linear layer that maps from hidden_dim to the number of classes
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
        # Get embeddings
        embedded = self.embedding(text, offsets)
        
        # Pass embeddings through the first linear layer and apply ReLU
        x = F.relu(self.fc1(embedded))
        
        # Output layer
        return self.fc2(x)

num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)


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