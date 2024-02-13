import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sklearn.preprocessing import LabelEncoder

# Define the contractions dictionary
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
