import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Specify the file path and columns
file_path = 'news-classification.csv'
columns_to_keep = ['title', 'content', 'author', 'category_level_1']

# Read the dataset
data = pd.read_csv(file_path)
selected_data = data[columns_to_keep]

# Initialize the tokenizer and label encoders
tokenizer = get_tokenizer("basic_english")
category_label_encoder = LabelEncoder()
author_label_encoder = LabelEncoder()

# Function to yield tokens from the dataset for building a vocab
def yield_tokens(data_iter):
    for text in data_iter['title'].tolist() + data_iter['content'].tolist():  # Combine titles and contents for vocab
        yield tokenizer(text)

# Build the vocabulary from the training dataset
train_data, _ = train_test_split(selected_data, test_size=0.2)
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Define the pipelines for text and label processing
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
category_label_pipeline = lambda x: category_label_encoder.fit_transform([x])[0]
author_label_pipeline = lambda x: author_label_encoder.fit_transform([x])[0] if pd.notnull(x) else -1  # Handle missing authors

# Process and encode labels and authors
selected_data['encoded_category_labels'] = selected_data['category_level_1'].apply(category_label_pipeline)
selected_data['encoded_authors'] = selected_data['author'].apply(author_label_pipeline)

# Tokenize and convert text to indices for title and content
selected_data['tokenized_title'] = selected_data['title'].apply(text_pipeline)
selected_data['tokenized_content'] = selected_data['content'].apply(text_pipeline)

# You can now store the DataFrame with the tokenized text, encoded labels, and authors
preprocessed_file_path = 'preprocessed_news-classification.csv'
selected_data.to_csv(preprocessed_file_path, index=False)

print(f"Preprocessed data saved to {preprocessed_file_path}")
