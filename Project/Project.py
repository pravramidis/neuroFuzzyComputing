
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

import numpy as np

import time

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
    "weren", "won", "wouldn, must"
])

#this function is used to get the reduced amount of data needed to test time to train in relation to input size by making sure we get data for each category
def select_rows_from_each_category(selected_data, num_rows):
    grouped_data = selected_data.groupby('category_level_2')

    final_selected_data = pd.DataFrame()

    num_rows_per_category = int(num_rows / len(grouped_data))

    for  group_df in grouped_data:
        if len(group_df) < num_rows_per_category:
            selected_rows = group_df.sample(len(group_df), replace=True)
        else:
            selected_rows = group_df.sample(num_rows_per_category)
        final_selected_data = pd.concat([final_selected_data, selected_rows])

    final_selected_data = final_selected_data.sample(frac=1).reset_index(drop=True)

    remaining_rows_needed = num_rows - final_selected_data.shape[0]
    if remaining_rows_needed > 0:
        remaining_rows = selected_data.sample(remaining_rows_needed)
        final_selected_data = pd.concat([final_selected_data, remaining_rows], ignore_index=True)

    return final_selected_data


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

def preprocess_text(text):
    text = expand_contractions(text)
    tokens = tokenizer(text.lower()) 
    
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    filtered_tokens = [token for token in stemmed_tokens if token not in stopwords and token.isalpha()]
    
    return " ".join(filtered_tokens)

def preprocess_url(url):
    cleaned_url = url.replace("http://", "").replace("https://", "").replace("/", " ").replace(".", " ").replace("-", " ").replace("com", "")

    return preprocess_text(cleaned_url)

def preprocess_data(row):
    
    preprocessed_text = preprocess_text(f"{row['title']} {row['content']} ")
    
    preprocessed_url = preprocess_url(row['url'])
    
    return f"{preprocessed_text} {preprocessed_url}"


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

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

def dataframe_to_dataset(dataframe, label_column):
    return [(row[label_column], row['preprocessed_text']) for index, row in dataframe.iterrows()]


class TextClassificationModelLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(TextClassificationModelLSTM, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        lstm_out, _ = self.lstm(embedded.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]
        return self.fc(lstm_out)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_acc, total_loss = 0, 0
    for label, text, offsets in dataloader:
        optimizer.zero_grad()
        output = model(text, offsets)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += (output.argmax(1) == label).sum().item()
    
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for label, text, offsets in dataloader:
            output = model(text, offsets)
            loss = criterion(output, label)
            
            total_loss += loss.item()
            total_acc += (output.argmax(1) == label).sum().item()
    
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader)

file_path = 'news-classification.csv'
columns_to_keep = ['title','content','url', 'category_level_1', 'category_level_2']
data = pd.read_csv(file_path)
selected_data = data[columns_to_keep].copy()

#The below commented code was used to reduce the input in order to test input size in relation to training time
# sample_size = 5000
# selected_data = select_rows_from_each_category(selected_data, sample_size)

num_rows = len(selected_data)
filename= f"output_{num_rows}"
output_file = open(filename, "w")
print(f"Number of data rows {num_rows}", file=output_file)
print(f"Number of data rows {num_rows}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCHS = 20  
LR = 0.01

start_pre = time.time()
start_total = time.time()

label_encoder = LabelEncoder()
selected_data['encoded_labels'] = label_encoder.fit_transform(selected_data['category_level_1'])

selected_data['preprocessed_text'] = selected_data.apply(preprocess_data, axis=1)

end_pre = time.time()

print(f"Time to preprocess data: {end_pre - start_pre}", file=output_file)
print(f"Time to preprocess data: {end_pre - start_pre}")

train_set, test_set = train_test_split(selected_data[['preprocessed_text', 'encoded_labels']], test_size=0.2)

train_iter = dataframe_to_dataset(train_set, 'encoded_labels')
test_iter = dataframe_to_dataset(test_set, 'encoded_labels')


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab([word for word in tokenizer(x) if word in vocab])

label_pipeline = lambda x: x

train_dataloader = DataLoader(to_map_style_dataset(train_iter), batch_size=64, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(to_map_style_dataset(test_iter), batch_size=64, shuffle=False, collate_fn=collate_batch)
num_classes = len(set([label for label, _ in train_iter]))
model = TextClassificationModelLSTM(len(vocab), 64, 128, num_classes).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

total_train_time = 0
total_eval_time = 0

print("Starting level 1 training", file=output_file)
print("Starting level 1 training")

for epoch in range(1, EPOCHS + 1):
    start_level_1_train = time.time()
    train_acc, train_loss = train(model, train_dataloader, optimizer, criterion, device)
    end_level_1_train = time.time()
    total_train_time += end_level_1_train - start_level_1_train
    start_level_1_eval = time.time()
    val_acc, val_loss = evaluate(model, test_dataloader, criterion, device)
    end_level_1_eval = time.time()
    total_eval_time += end_level_1_eval - start_level_1_eval
    print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    if (epoch == EPOCHS): 
        print(f"For category level 1 Accuracy: {val_acc:.4f}")
        print(f"For category level 1 Accuracy: {val_acc:.4f}", file=output_file)


        print(f"Time to train level 1: {total_train_time}", file=output_file)
        print(f"Time to train level 1: {total_train_time}")
        print(f"Time to eval level 1: {total_eval_time}", file=output_file)
        print(f"Time to eval level 1: {total_eval_time}")
        print(f"Average inference time level 1: {total_eval_time/EPOCHS}", file=output_file)
        print(f"Avergae inference time level 1: {total_eval_time/EPOCHS}")


model_filename = f'model_for_level_1.pth'
torch.save(model.state_dict(), model_filename)

total_acc = 0
for category_level_1 in selected_data['category_level_1'].unique():
    print(f"\n#######################################################################")
    print(f"\nTraining for category: {category_level_1}")
    print(f"\nTraining for category: {category_level_1}", file=output_file)
    category_data = selected_data[selected_data['category_level_1'] == category_level_1].copy()
    
    label_encoder_level_2 = LabelEncoder()
    category_data['encoded_labels_level_2'] = label_encoder_level_2.fit_transform(category_data['category_level_2'])
    
    train_set_level_2, test_set_level_2 = train_test_split(category_data[['preprocessed_text', 'encoded_labels_level_2']], test_size=0.2)
    
    train_iter_level_2 = dataframe_to_dataset(train_set_level_2, 'encoded_labels_level_2')
    test_iter_level_2 = dataframe_to_dataset(test_set_level_2, 'encoded_labels_level_2')
    
    vocab = build_vocab_from_iterator(yield_tokens(train_iter_level_2), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab([word for word in tokenizer(x) if word in vocab])
    label_pipeline = lambda x: x


    train_dataloader = DataLoader(to_map_style_dataset(train_iter_level_2), batch_size=64, shuffle=True, collate_fn=collate_batch)

    test_dataloader = DataLoader(to_map_style_dataset(test_iter_level_2), batch_size=64, shuffle=False, collate_fn=collate_batch)
    num_classes = len(set([label for label, _ in train_iter_level_2]))
    model = TextClassificationModelLSTM(len(vocab), 64, 128, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    total_train_time = 0
    total_eval_time = 0

    val_acc = 0
    for epoch in range(1, EPOCHS + 1):


        start_level_2_train = time.time()
        train_acc, train_loss = train(model, train_dataloader, optimizer, criterion, device)
        end_level_2_train = time.time()
        total_train_time += end_level_2_train - start_level_2_train
        start_level_2_eval = time.time()
        val_acc, val_loss = evaluate(model, test_dataloader, criterion, device)
        end_level_2_eval = time.time()
        total_eval_time += end_level_2_eval - start_level_2_eval
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if (epoch == EPOCHS): 
            print(f"For subcategory {category_level_1} Accuracy: {val_acc:.4f}")
            print(f"For subcategory {category_level_1} Accuracy: {val_acc:.4f}",file=output_file)

            print(f"Time to train level 2: {total_train_time}", file=output_file)
            print(f"Time to train level 2: {total_train_time}")
            print(f"Time to eval level 2: {total_eval_time}", file=output_file)
            print(f"Time to eval level 2: {total_eval_time}")
            print(f"Average inference time level 2: {total_eval_time/EPOCHS}", file=output_file)
            print(f"Avergae inference time level 2: {total_eval_time/EPOCHS}")
    total_acc += val_acc
    

    model_filename = f'model_for_{category_level_1.replace(" ", "_")}_level_2.pth'

    torch.save(model.state_dict(), model_filename)

total_acc = total_acc /17
print(f"avg level2: {total_acc:.4f}")
print(f"avg level2: {total_acc:.4f}", file=output_file)
end_total = time.time()

print(f"Total runtime: {end_total - start_total}", file=output_file)
print(f"Total runtime: {end_total - start_total}")
output_file.close()









