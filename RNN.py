class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        print(att1.shape, att2.shape)
        att = self.full_att(self.tanh(att1 + att2.unsqueeze(0))).squeeze(2)
        alpha = torch.functional.F.softmax(att, dim=1)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha
    


class RNNModule(nn.Module):
    def __init__(self, input_size,embed_size, hidden_size, num_layers, vocab_size,attention_dim):
        super(RNNModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(encoder_dim=input_size, decoder_dim=hidden_size, attention_dim=attention_dim)
        self.init_weights()
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.fc.bias.data.fill_(0)
    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        hidden = torch.zeros(self.num_layers, features.size(0), self.hidden_size).to(features.device)
        cell = torch.zeros(self.num_layers, features.size(0), self.hidden_size).to(features.device)
        outputs = torch.zeros(embeddings.size(0), embeddings.size(1), self.fc.out_features).to(features.device)

        for t in range(embeddings.size(1)-1):
            attention_weighted_encoding, _ = self.attention(features, hidden)
            lstm_input = torch.cat((embeddings[:, t], attention_weighted_encoding), dim=1)
            _,(hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            outputs[:, t, :] = self.fc(hidden)
        return outputs
    

embed_size = 300
hidden_size = 512
vocab_size = 100
encoder_dim = 2048

attention_dim = 256

batch_size = 2
num_pixels = 1
max_caption_length = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Randomly generated input data
features = torch.randn(batch_size, num_pixels, encoder_dim).to(device)
captions = torch.randint(0, vocab_size, (batch_size, max_caption_length)).to(device)

# flatten features
features = features.view(batch_size, num_pixels, -1)
print(features.shape)
# Initialize the model
decoder = RNNModule(encoder_dim,embed_size, hidden_size, num_layers=1, vocab_size=vocab_size,attention_dim=attention_dim)

# Forward pass
outputs = decoder(features, captions)

print("Output shape:", outputs.shape)  # Expected: (batch_size, max_caption_length, vocab_size)
print("Outputs:", outputs)