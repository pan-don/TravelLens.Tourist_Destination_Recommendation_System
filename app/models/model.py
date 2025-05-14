import torch
from torch import nn

class ContentBasedFilteringModel(nn.Module):
    def __init__(self, input_num: int, input_dim: int, input_rnn: int):
        self.embedding = nn.Embedding(num_embeddings=input_num, embedding_dim=input_dim)
        self.rnn = nn.GRU(input_size=input_rnn, hidden_size=input_rnn, num_layers=3, batch_first=True)
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_rnn, input_rnn),
            nn.ReLU(),
            nn.Dropout()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_rnn, input_rnn),
            nn.ReLU()
        )
    
    def forward(self, idx):
        x1 = self.embedding(idx)
        _, h_n = self.rnn(x1)
        x3 = self.fc1(h_n.squeeze(dim=0))
        x4 = self.fc2(x3)
        return x4