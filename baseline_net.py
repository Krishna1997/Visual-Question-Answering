import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from googlenet import googlenet

class BaselineNet(nn.Module):

    def __init__(self, num_embeddings, num_classes):
        super().__init__()
        self.gnet = googlenet(pretrained=True, remove_fc=True)
        self.embed = nn.Linear(num_embeddings, 1024)
        self.fc = nn.Linear(1024 + 1024, num_classes)

    def forward(self, image, question_encoding):
        img = self.gnet(image)
        ques = self.embed(question_encoding)
        con = torch.cat((img, ques), dim=1)
        return self.fc(con)
