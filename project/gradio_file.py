import torch
import gradio as gr
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
import regex as re
from stop_words import get_stop_words
stop_words = get_stop_words('ukrainian')
stop_words+=['те','її',"іі","іє","їє",'a','а','із','а']
                
    
class UniversalRNN(torch.nn.Module):
    
    def __init__(
        self,
        num_embeddings,
        out_channels,
        rnn_channels=128,
        rnn_type=nn.GRU,
        n_rnns=1,
        bidirectional=True,
        average_type=None
    ):
        super().__init__()
        
        self.embedding_layer = nn.Embedding(num_embeddings, rnn_channels)
        self.rnns = rnn_type(
            rnn_channels, 
            rnn_channels, 
            bidirectional=bidirectional, 
            num_layers=n_rnns,
            batch_first=True
        )
        if not (average_type is None or average_type in ["mean", "last"]):
            raise ValueError(f"{average_type} is nit supported average_type")
        self.average_type = average_type
        self.classifier = nn.Linear(
            rnn_channels * 2 if bidirectional else rnn_channels, 
            out_channels, 
        )
        
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.rnns(x)[0]
        if self.average_type is None:
            x = self.classifier(x)
        else:
            if self.average_type == "mean":
                x = x.mean(1)
            elif self.average_type == "last":
                x = x[:,-1,:]
            x = self.classifier(x)
        return x    

tokenizer = get_tokenizer('basic_english')
vocab = torch.load('vocab_obj_new.pth')
model = torch.load('fake_model_new.pth')

def select_text_subsequance(input):
        if len(input) < 256:
            return input + [0] * (256 - len(input))
        elif len(input) > 256:
            start = 0
            return input[start : start + 256]
        else: 
            return input

def preprocess(text):
    temp = []
    result = []
    for token in text.split(' '):
        if token.lower() not in stop_words:
            temp.append(token.lower())
    temp = " ".join(temp)
    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub('#[а-яА-ЯA-Za-z0-9]*','',temp)
    temp =re.sub('\u200b','',temp)
    temp = re.sub('[;.,!?:—–]+','',temp)
    for token in temp.split(' '):
        if token.lower() not in stop_words:
            result.append(token.lower())
    result= " ".join(result)
    return result

def predict(text):
    text= preprocess(text)
    prep_text = vocab(tokenizer(text))
    print(prep_text)
    if torch.sigmoid(model(torch.tensor(select_text_subsequance(prep_text))).detach()).round()==0:
        return 'Fake'
    else:
        return 'True'

iface = gr.Interface(
  fn=predict, 
  inputs='text',
  outputs='text',
  examples=[["Українські телеграм канали повідомляють, Київ взятий за три дні. Зеленський втік у сша, або Польщу."],["ШОК! На просторах інтернету з'явилися відіо з Американських біолабораторій в Чорнобилі, де вирощують хом'яків, котрі будуть їсти росіян, що загинули під час боїв за Крим."]]
)

iface.launch()
