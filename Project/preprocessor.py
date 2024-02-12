import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sklearn.preprocessing import LabelEncoder

# Define the contractions dictionary
contractions_dict = {
    # Include all the contractions as defined in your initial script
}

def expand_contractions(text, contractions_dict=contractions_dict):
    contractions_pattern = re.compile('(%s)' % '|'.join(map(re.escape, contractions_dict.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        if not match:  # Check if match is empty or None
            return ''
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match.lower(), match)
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text


# Define the set of stopwords
stopwords = set([
    # Include all the stopwords as defined in your initial script
])

# Preprocess text function
def preprocess_text(text):
    text = expand_contractions(text)
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords and token.isalpha()]
    return " ".join(filtered_tokens)

# Read the dataset
file_path = 'news-classification.csv'
columns_to_keep = ['title', 'content', 'author', 'category_level_1']
data = pd.read_csv(file_path)
selected_data = data[columns_to_keep].copy()

# Add preprocessing and label encoding
label_encoder = LabelEncoder()
selected_data['encoded_labels'] = label_encoder.fit_transform(selected_data['category_level_1'])
selected_data['preprocessed_text'] = selected_data.apply(
    lambda row: preprocess_text(f"{row['author']} {row['title']} {row['content']}"), axis=1)

# Save preprocessed data to a new CSV file
output_file_path = 'preprocessed_news_classification.csv'
selected_data.to_csv(output_file_path, index=False)

print(f"Preprocessed data saved to {output_file_path}")
