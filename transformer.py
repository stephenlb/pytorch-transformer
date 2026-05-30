import re
import torch

def generate_math():
    data = []
    for i in range(100):
        question = f'{i}+{i}='
        answer = f'{i+i}'
        data.append([f'{question:p<10}', f'{answer:p<10}'])
    return data

## 10 + 30 = 40
## TODO MATH GPT
## TODO Train on Math problems! Math GPT
## TODO Batching
## TODO Positional encoding or other modern approach sine/cosine / RoPE? / ALiBI
## TODO ML Ops `mlops`

## TODO Training data generator based on our input data file
## TODO Upgrade Dictionary support better word memroy management
## TODO data set to learn from
## TODO      RoPE - for positional encodeing
## TODO training model.train()
## TODO question_mask = torch.nn.Transformer.generate_square_subsequent_mask(question_embedding.size(0)) ## TODO size(1) if batching
## TODO trim start and end between target and training_data[1]

## TODO ✅ ignore_index=0) <-- restor ignore_index=0
## TODO ✅ add <unk> token when word not in dictionary thank you @m_nizwa
## TODO ✅ Linear Out for our target token output size ( cnoverter to embedding )
## TODO ✅ LOGITS for token output
## TODO ✅ mask
## TODO ✅ Dictionary Tokenizer
## TODO ✅ build Dictionary
## TODO ✅ Transformer ( self-attent / multi-heads )
## TODO ✅ tgt (second param in transformer(1,2)
## TODO ✅ Special tokens padding, end, start

#training_data = [
#    ['Hello Kyle this is all the data <pad>',
#    '<start> and here is the rest <end>']
#]

class PositionalEncoding(torch.nn.Module):
    def __init__(self, dims):
        pass

class SwiGLU(torch.nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        # SwiGLU requires two parallel linear projections
        self.w1 = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = torch.nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x):
        # Apply SiLU (Swish) to one path and multiply by the other
        return torch.nn.functional.silu(self.w1(x)) * self.w2(x)


class Dictionary(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.norm = r'[^0-9a-z \-=+]'
        self.dictionary = {
            'p' : 0, ## Padding
            's' : 1, ## Start of Sequence
            'e' : 2, ## End of Sequence
            'u' : 3, ## Unknown Token
        }
        union_clip = set(self.dictionary.keys())
        all_words = " ".join([" ".join(phrase) for phrase in data])
        self.word_list = set(list(self.normalize(" ".join(all_words)))) - union_clip
        self.dictionary.update({
            word : index + len(self.dictionary)
            for index, word in enumerate(self.word_list)
            #if not word in self.dictionary
        })
        self.decoder = {
            self.dictionary[k] : k
            for k in self.dictionary.keys()
        }

    def __len__(self):
        return len(self.dictionary)

    def __repr__(self):
        return str(self.dictionary)

    def decode(self, output):
        tokens = torch.argmax(output, dim=1)
        words = [self.decoder[token.item()] for token in tokens]
        return words

    def normalize(self, words):
        return re.sub(self.norm, '', words.lower())
        
    def tokenize(self, batches):
        tokens = [
            [self.dictionary[char]
            for char in list(self.normalize(exp))]
            for exp in batches
        ]
        return torch.Tensor(tokens).to(torch.long)

    def one_hot(self, words):
        tokens = self.tokenize(words)
        return torch.nn.functional.one_hot(tokens)#.to(torch.float)
        
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
            #activation=SwiGLU(dims, dims),
            batch_first=True,
        )
        self.linear = torch.nn.Linear(dims, len(dictionary))
        self.soft = torch.nn.Softmax()

    def forward(self, question, answer):
        question_tokens = self.dictionary.tokenize(question)
        answer_tokens = self.dictionary.tokenize(answer)

        question_embedding = self.embedding(question_tokens)
        answer_embedding = self.embedding(answer_tokens)

        ## TODO Positional encoding  (RoPE) 
        #question_mask = torch.nn.Transformer.generate_square_subsequent_mask(question_embedding.size(0)) ## TODO size(1) if batching
        answer_mask = torch.nn.Transformer.generate_square_subsequent_mask(answer_embedding.size(0))
        out = self.transformer(question_embedding, answer_embedding, tgt_mask=answer_mask)
        out = self.linear(out)
        #out = self.soft(out)
        return out

training_data = generate_math()
dictionary = Dictionary(training_data)
print(dictionary)
print(dictionary.decoder)
model = Transformer(dictionary)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
epochs = 10
batches = 10
batch_size = 10
for epoch in range(epochs):
    for batch in range(batches):
        indexes = torch.randperm(50) + torch.randperm(50)
        ## TODO pre tokenize training data!!
        ## TODO pre tokenize training data!!
        ## TODO pre tokenize training data!!
        ## TODO pre tokenize training data!!
        features = [a[0] for a in training_data[indexes]]
        labels = [a[1] for a in training_data[indexes]]
        #print(epoch)
        output = model(features, labels)
        #print(output)
        #rint('output.shape', output.shape)
        ## TODO trim start and end between target and training_data[1]
        #targets = dictionary.one_hot(training_data[1])
        #print('targets', targets)
        #print('targets', targets.shape)
        #print('output', output)
        #print('output', output.shape)
        targets = dictionary.tokenize(labels).reshape(-1)
        loss = criterion(output.reshape(-1, len(dictionary)), targets)
        #loss = -torch.log(output)[targets].sum()
        ##loss = torch.log(output)
        #(-log(p))[correct].sum()
        print('loss',loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #words = dictionary.decode(output)
        #print(" ".join(words))
#print(words)
#print(words)
#print(" ".join(words))
