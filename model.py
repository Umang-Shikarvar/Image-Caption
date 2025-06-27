import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# Encoder 
class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)      # Use pre-trained ResNet-50
        for param in resnet.parameters():
            param.requires_grad_(False)                 # Freeze ResNet weights

        modules = list(resnet.children())[:-1]          # Remove the last FC layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, 
                               embed_size)              # Map to embedding size

    def forward(self, images):
        features = self.resnet(images)                  # Extract features
        features = features.view(features.size(0), -1)  # Flatten
        embeddings = self.embed(features)               # Map to embedding
        return embeddings


# Decoder
class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()

        # Embedding layer contains the mapping from word indices to embeddings
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # Linear layer to map LSTM outputs to vocabulary size
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):

        # remove <end> token from gt captions as it is not needed for training
        cap_embedding = self.embed(captions[:, :-1])        # (bs, cap_length) -> (bs, cap_length-1, embed_size)

        # concatenate the images features to the first of caption embeddings.
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1) # (bs, cap_length, embed_size)

        # Pass the embeddings through the LSTM
        lstm_out, _ = self.lstm(embeddings)   # (bs, cap_length, hidden_size), (h, c)

        # Map the LSTM outputs to vocabulary size
        outputs = self.linear(lstm_out)  # (bs, cap_length, vocab_size)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        predicted_ids = []

        # Loop to generate up to max_len words
        for _ in range(max_len):
            # Pass input and states through LSTM
            lstm_out, states = self.lstm(inputs, states)          # lstm_out: (1, 1, hidden_size)
            
            # Project LSTM output to vocab scores
            outputs = self.linear(lstm_out.squeeze(1))            # outputs: (1, vocab_size)

            # Get index of the highest scoring word
            _, predicted_idx = outputs.max(1)                     # predicted_idx: (1,)

            # Append predicted word index to result
            predicted_ids.append(predicted_idx.item())

            # Stop if <end> token  is predicted
            if predicted_idx.item() == 1:
                break

            # Prepare input for next time step
            inputs = self.embed(predicted_idx)                    # inputs: (1, embed_size)
            inputs = inputs.unsqueeze(1)                          # inputs: (1, 1, embed_size)

        return predicted_ids