import torch
import torch.nn as nn

# Define your neural network architecture
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
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        return self.fc(lstm_out)

# Load the saved model
model = TextClassificationModelLSTM(57513, 64, 128, 17)
model.load_state_dict(torch.load('model_for_level_1.pth'))

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
print("Values of each individual weight:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
	

# Print the architecture of the model
print("\nModel Architecture:")
print(model)

