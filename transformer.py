import re
import torch

## TODO Training data generator based on our input data file
## TODO Upgrade Dictionary support better word memroy management
## TODO data set to learn from
## TODO Positional encoding or other modern approach sine/cosine / RoPE? / ALiBI
## TODO      RoPE - for positional encodeing
## TODO training model.train()
## TODO Linear Out for our target token output size ( cnoverter to embedding )
## TODO ✅ mask
## TODO ✅ Dictionary Tokenizer
## TODO ✅ Transformer ( self-attent / multi-heads )
## TODO ✅ tgt (second param in transformer(1,2)
## TODO ✅ build Dictionary
## TODO ✅ Special tokens padding, end, start
## TODO ✅ tgt (second param in transformer(1,2)

training_data = ['Hello this is all the data', 'and here is the rest']

class Dictionary(torch.nn.Module):
    def __init__(self, all_words):
        super().__init__()
        self.norm = r'[^0-9a-z \-]'
        self.word_list = set(self.normalize('pad ' + " ".join(all_words)).split())
        self.dictionary = {
            '<padding>' : 0,
            '<start_of_sequence>' : 1,
            '<end_of_sequence>' : 2,
        }
        self.dictionary.update({
            word : index + len(self.dictionary)
            for index, word in enumerate(self.word_list)
        })
        self.decoder = {
            self.dictionary[k] : k
            for k in self.dictionary.keys()
        }

    def __len__(self):
        return len(self.dictionary)

    def __repr__(self):
        return str(self.dictionary)

    def decode(self, tokens):
        tokens = torch.argmax(output, dim=1)
        #return tokens
        #return self.decoder
        words = [self.decoder[token.item()] for token in tokens]
        return words

    def normalize(self, words):
        return re.sub(self.norm, '', words.lower())
        
    def tokenize(self, words):
        tokens = [self.dictionary[word] for word in words.split()]
        return torch.Tensor(tokens).to(torch.long)

class Transformer(torch.nn.Module):
    def __init__(self, dictionary):
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
        self.linear = torch.nn.Linear(dims, len(dictionary))

    def forward(self, question, answer):
        question_norm = self.dictionary.normalize(question)
        answer_norm = self.dictionary.normalize(answer)

        question_tokens = self.dictionary.tokenize(question_norm)
        answer_tokens = self.dictionary.tokenize(answer_norm)

        question_embedding = self.embedding(question_tokens)
        answer_embedding = self.embedding(answer_tokens)

        ## TODO Positional encoding  (RoPE) 
        ## TODO mask
        question_mask = torch.nn.Transformer.generate_square_subsequent_mask(question_embedding.size(0))
        answer_mask = torch.nn.Transformer.generate_square_subsequent_mask(answer_embedding.size(0))
        out = self.transformer(question_embedding, answer_embedding, src_mask=question_mask, tgt_mask=answer_mask)
        out = self.linear(out)
        ## TODO LOGITS for token output
        return out

dictionary = Dictionary(training_data)
print(dictionary)
model = Transformer(dictionary)
output = model(training_data[0], training_data[1])
print(output)
print(output.shape)
words = dictionary.decode(output)
#print(" ".join(words))
print(words)
#print(words)
#print(" ".join(words))
