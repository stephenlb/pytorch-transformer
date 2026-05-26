import re
import torch

## TODO data set to learn from
## TODO Positional encoding or other modern approach
## TODO mask
## TODO tgt (second param in transformer(1,2)
## TODO 
## TODO ✅ build Dictionary
## TODO ✅ Special tokens padding, end, start

training_data = 'Hello this is all the data'

class Dictionary():
    def __init__(self, all_words):
        self.norm = r'[^0-9a-z \-]'
        self.word_list = set(self.normalize('pad ' + all_words).split())
        self.dictionary = {
            '<padding>' : 0,
            '<start_of_sequence>' : 1,
            '<end_of_sequence>' : 2,
        }
        self.dictionary.update({
            word : index
            for index, word in enumerate(self.word_list)
        })

    def __len__(self):
        return len(self.dictionary)

    def normalize(self, words):
        return re.sub(self.norm, '', words.lower())
        
    def tokenize(self, words):
        tokens = [self.dictionary[word] for word in words.split()]
        return torch.Tensor(tokens).to(torch.long) #, dtype=torch.long)

class Transformer(torch.nn.Module):
    def __init__(self, dictionary):
        ## Dictionary Tokenizer
        ## Transformer ( self-attent / multi-heads )
        ## Linear Out for our target token output size ( cnoverter to embedding )
        super().__init__()
        self.dictionary = dictionary
        self.dims = dims = 128
        self.number_of_words = number_of_words = len(dictionary)
        self.embedding = torch.nn.Embedding(number_of_words, dims)
        self.transformer = torch.nn.Transformer(
            d_model=dims,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=dims,
            dropout=0.1,
            activation=torch.nn.GELU(),
            batch_first=True,
        )

    def forward(self, words):
        norm = self.dictionary.normalize(words)
        tokens = self.dictionary.tokenize(norm)
        embedding = self.embedding(tokens)
        ## TODO mask
        ## TODO tgt (second param in transformer(1,2)
        out = self.transformer(embedding, embedding) ## TODO < Target
        return out
        #return tokens
        #return embedding

#src = torch.rand((10, 32, 512))
#tgt = torch.rand((20, 32, 512))
#out = transformer_model(src, tgt)

dictionary = Dictionary(training_data)
model = Transformer(dictionary)
output = model(training_data)
print(output)
