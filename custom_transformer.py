import re
import torch
from pathlib import Path

### TODO Dictionary must tokenize better!!!!
### TODO Train on this file specificly
### TODO Finish Batch and Shuffling <--
### TODO ✅ POSITIONAL Encoding

## Custom Stephen Transformer
class StephenFormer(torch.nn.Module):
    ## TODO KV Cache - good for inference
    ## TODO Multi-head
    ## TODO Attention
    ## TODO Forward MLP with activation (GELU)
    ## TODO 
    def __init__(self, dictionary, dims=128, heads=1):
        super().__init__()
        self.dictionary = dictionary
        self.dims = dims
        self.heads = heads
        #self.number_of_words = number_of_words = len(dictionary)

        ## TODO think about how to use this later
        ## Embedding
        #self.embedding = torch.rand(self.number_of_words, dims, requires_grad=True) - 0.5

        ## Self Attention
        self.query_projection = torch.nn.Linear(dims, dims)
        self.key_projection   = torch.nn.Linear(dims, dims)
        self.value_projection = torch.nn.Linear(dims, dims)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(0.1)

        ## Feed Forward
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(dims, dims * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dims * 4, dims),
            torch.nn.Dropout(0.1),
        )

        ## Layer Norm and Add
        self.norm = torch.nn.LayerNorm(dims)

        ## Output Projection
        self.output_projection = torch.nn.Linear(dims, len(dictionary))

    def attention(self, query, key, value):
        ## TODO Multi-headed attention
        out = query @ key.transpose(1,2)
        out = out / torch.sqrt(torch.tensor(self.dims))
        #out = torch.softmax(out, dim=-1)
        out = self.softmax(out)
        out = self.dropout(out)
        out = out @ value
        print('attention')
        print(out.shape)
        return out

    def forward(self, inputs):
        ## TODO @Cloudhead- use a single projection x 3 for faster better
        ##  Q,K,V=self.qkv(input).chunk(3,dim=-1)
        query = self.query_projection(inputs)
        key   = self.key_projection(inputs)
        value = self.value_projection(inputs)
        out = self.attention(query, key, value)
        print('forward')
        print(out.shape)
        out = self.norm(inputs + out)
        print(out.shape)
        out = self.feedforward(out)
        print(out.shape)
        out = self.norm(inputs + out)
        print(out.shape)
        out = self.output_projection(out)
        print(out.shape)
        return out


## TODO Training data generator based on our input data file
## TODO Upgrade Dictionary support better word memroy management
## TODO data set to learn from
## TODO      RoPE - for positional encodeing
## TODO training model.train()
## TODO question_mask = torch.nn.Transformer.generate_square_subsequent_mask(question_embedding.size(0)) ## TODO size(1) if batching
## TODO trim start and end between target and training_data[1]

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
	### 32,20, hidden
        return x + self.pe[:x.shape[0],:].unsqueeze(0)

class Dictionary(torch.nn.Module):
    def __init__(self, corpus):
        super().__init__()
        ### TODO We are eating the newline chars......
        self.norm = r'#.*$'
        self.dictionary = {
            '<pad>' : 0, ## Padding
            '<start>' : 1, ## Start of Sequence
            '<end>' : 2, ## End of Sequence
            '<unknown>' : 3, ## Unknown Token
        }
        union_clip = set(self.dictionary.keys())
        self.vocab = set(self.normalize(corpus)) - union_clip
        self.dictionary.update({
            word : index + len(self.dictionary)
            for index, word in enumerate(self.vocab)
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
        return list(words)
        #return re.sub(self.norm, '', words.lower())
        #return words
        
    def tokenize(self, batch_size):
        tokens = [
            [self.dictionary[words]
                for words in self.normalize(phrases)
            ] for phrases in batch_size
        ]
        return torch.Tensor(tokens).to(torch.long)

    def one_hot(self, words):
        tokens = self.tokenize(words)
        return torch.nn.functional.one_hot(tokens).to(torch.float)

        
class Transformer(torch.nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.dims = dims = 128
        self.number_of_words = number_of_words = len(dictionary)
        self.embedding = torch.nn.Embedding(number_of_words, dims)
        self.positional = PositionalEncoding(dims)
        self.stephen_transformer = StephenFormer(dictionary, dims=dims)
        #self.linear = torch.nn.Linear(dims, len(dictionary))
        #self.soft = torch.nn.Softmax()

    def forward(self, inputs):
        tokens = self.dictionary.tokenize(inputs)
        embeddings = self.embedding(tokens)
        pos_encoded = self.positional(embeddings)
        ## TODO mult-pass
        out = self.stephen_transformer(pos_encoded)
        return out

## Read self so we can learn self, and replicate
training_data = Path(__file__).read_text()
#training_data = generate_math()
dictionary = Dictionary(training_data)
print(dictionary)
print(dictionary.decoder)
model = Transformer(dictionary)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
#criterion = torch.nn.NLLLoss(ignore_index=0)
epochs = 1
batch_size = 32

## TODO rewrite for new data type
training_set = {
    'features' : [],
    'labels'   : [],
    'len'      : len(training_data),
    'window'   : 20, ## tokens
}
## TODO Problem - convert to a window target, not just one value
def batch_prepare():
    window = training_set['window']
    for position in range(training_set['len'] - window - 1):
        segment = training_data[position:position+window]
        target = training_data[position+1:position+window+1]
        training_set['features'].append(segment)
        training_set['labels'].append(target)

## TODO problem with out of range index
## TODO convert to a generator with yield
##    ChatGPT has a lot of goblins and gremlins token output due to training
## Don't delete this -> CodeLangtonsEntropy -> Goblins and Gremlins <-
## 
def get_batch():
    samples = len(training_set['features'])
    indexes = torch.arange(0, samples)[torch.randperm(samples)]
    seen = set()
    for batch in range(len(indexes) // batch_size):
        start = batch * batch_size
        end = (batch+1) * batch_size
        seen.add(indexes[start:end])
        features = [training_set['features'][index] for index in indexes[start:end]]
        labels = [training_set['labels'][index] for index in indexes[start:end]]
        yield features, labels

def train():
    for epoch in range(epochs):
        for features, targets in get_batch():
            output = model(features)
            output = output#.reshape(-1, len(dictionary))
            #targets = dictionary.tokenize(targets).unsqueeze(2)#[:,:,None]#.reshape(-1)
            targets = dictionary.one_hot(targets)#.unsqueeze(2)#[:,:,None]#.reshape(-1)
            print(targets)
            print('targets')
            #print(targets)
            print(targets.shape)
            print('output')
            #print(output)
            print(output.shape)
            loss = criterion(output, targets)
            #loss = criterion(output.reshape(-1, len(dictionary)), targets)
            print('loss',loss.item())
            break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #words = dictionary.decode(output)
        #print(words)

batch_prepare()
#print(training_set)
train()
