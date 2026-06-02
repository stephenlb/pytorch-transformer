import re
import torch


### TODO Train on big data
### TODO Train on Math teach basic 
### TODO Finish Batch and Shuffling
### TODO POSITIONAL Encoding
### TODO ML FLOW

## Custom Stephen Transformer
class StephenFormer(torch.nn.Module):
    ## TODO KV Cache - good for inference
    ## TODO Multi-head
    ## TODO 
    ## TODO 
    ## TODO 
    def __init__(self, dictionary, dims=128):
        super().__init__()
        self.dictionary = dictionary
        self.dims = dims
        self.query_projection = torch.nn.Linear(dims, dims)
        self.key_projection   = torch.nn.Linear(dims, dims)
        self.value_projection = torch.nn.Linear(dims, dims)
        self.output_projection = torch.nn.Linear(dims, len(dictionary))
        self.dropout = torch.nn.Dropout(0.1)

    def attention(self, query, key, value):
        ## skale query and key before matmult @
        out = torch.nn.functional.softmax(query @ key) / torch.sqrt(self.dims)
        pass

    def forward(self, inputs):
        pass


def generate_math():
    data = []
    for i in range(100):
        question = f'{i}+{i}='
        answer = f'{i+i}'
        data.append([f'{question:p<10}', f'{answer:p<4}'])
        #data.append([f'{question}', f'{answer}'])
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
    def __init__(self, dims, max_tokens=5000):
        super().__init__()
        pe = []
        for token in range(max_tokens):
            if token % 2: pe.append(torch.sin(torch.linspace(0, max_tokens-token+1, dims)))
            else:         pe.append(torch.cos(torch.linspace(0, max_tokens-token+1, dims)))
        pe = torch.concat(pe).reshape(max_tokens, dims)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[0],:]

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

    def decode(self, outputs):
        batch = []
        for output in outputs:
            tokens = torch.argmax(output, dim=1)
            batch.append([self.decoder[token.item()] for token in tokens])
        return batch

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
        self.positional = PositionalEncoding(dims)
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

        question_embedding = self.positional(question_embedding)

        ## TODO Positional encoding  (RoPE) 
        #question_mask = torch.nn.Transformer.generate_square_subsequent_mask(question_embedding.size(0)) ## TODO size(1) if batching
        answer_mask = torch.nn.Transformer.generate_square_subsequent_mask(answer_embedding.size(1))
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
epochs = 100
batches = 10
batch_size = 10
training_data_len = len(training_data)
def get_batch():
    indexes = torch.randint(0, training_data_len, (batch_size,))
    features, shifted, outputs = [], [], []
    for index in range(batch_size):
        sample = training_data[indexes[index]]
        features.append(sample[0])
        shifted.append(sample[1][:-1])
        outputs.append(sample[1][1:])
    return features, shifted, outputs

for epoch in range(epochs):
    for batch in range(batches):
        features, shifted, target = get_batch()
        output = model(features, shifted)
        targets = dictionary.tokenize(target).reshape(-1)
        loss = criterion(output.reshape(-1, len(dictionary)), targets)
        print('loss',loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    words = dictionary.decode(output)
    print(words)
#print(words)
#print(words)
#print(" ".join(words))
