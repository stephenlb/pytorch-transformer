import re
import time
import torch
import random
from pathlib import Path

### TODO More Data ( take pytorch github and copy all py files into 1 big file )
### TODO Visualize the model
### TODO Parameter Golf from OpenAI - I want to read the placeholder code
### TODO Positional Encoding Rotary attention
### TODO tokenize differently - Problem - a lot of blank space chars
### TODO Multi-head attn
### TODO Multiple Transformer Layers
### TODO Performance improvemnts KV cache / singlek projection QKV
### TODO Temperature on predition
### TODO Train on Tiny Story dataset
### TODO Fine Tuning
### TODO Fine Tuning: smaller batches on final
### TODO Fine Tuning: on final custom_transformer.py
### TODO ✅ Save and Load model
### TODO ✅ Causal Mask
### TODO ✅ Print Loop ( so that it can save the file )
### TODO ✅ Send to GPU Device
### TODO ✅ Print more
### TODO ✅ Test the model, see what it is saying
### TODO ✅ Dictionary must tokenize better!!!!
### TODO ✅ Train on this file specificly
### TODO ✅ Finish Batch and Shuffling <--
### TODO ✅ POSITIONAL Encoding
### TODO ✅ We are eating the newline chars......
### TODO KV Cache - good for inference
### TODO Training data generator based on our input data file
### TODO Upgrade Dictionary support better word memroy management
### TODO training model.train()
### TODO question_mask = torch.nn.Transformer.generate_square_subsequent_mask(question_embedding.size(0)) ## TODO size(1) if batching
### TODO trim start and end between target and training_data[1]

## Custom Stephen Transformer
class StephenFormer(torch.nn.Module):
    ## TODO Multi-head
    ## TODO 
    def __init__(self, dictionary, dims=256, heads=4):
        super().__init__()
        self.dictionary = dictionary
        self.dims = dims
        self.heads = heads

        ## TODO think about how to use this later
        ## Embedding
        #self.embedding = torch.rand(self.number_of_words, dims, requires_grad=True) - 0.5

        ## Self Attention
        self.query_projection = torch.nn.Linear(dims, dims)
        self.key_projection   = torch.nn.Linear(dims, dims)
        self.value_projection = torch.nn.Linear(dims, dims)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.1)

        ## Feed Forward
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(dims, dims * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dims * 4, dims),
            torch.nn.Dropout(0.1),
        )

        ## Layer Norm and Add
        self.norm1 = torch.nn.LayerNorm(dims)
        self.norm2 = torch.nn.LayerNorm(dims)

        ## Output Projection
        self.output_projection = torch.nn.Linear(dims, len(dictionary))

    def attention(self, query, key, value):
        batch, seq, _ = query.shape

        ## Multi-headed attention
        heads = self.heads
        dims = self.dims
        assert self.dims % heads == 0
        h_dims = self.dims // heads

        query = query.view(batch, seq, heads, h_dims).transpose(1, 2)
        key = key.view(batch, seq, heads, h_dims).transpose(1, 2)
        value = value.view(batch, seq, heads, h_dims).transpose(1, 2)

        out = query @ key.transpose(-2, -1)
        out = out / torch.sqrt(torch.tensor(self.dims))
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq, device=out.device)
        out = out + mask
        out = self.softmax(out)
        out = self.dropout(out)
        out = out @ value

        return out.transpose(1, 2).contiguous().view(batch, seq, dims)

    def forward(self, inputs):
        ## TODO @Cloudhead- use a single projection x 3 for faster better
        ##  Q,K,V=self.qkv(input).chunk(3,dim=-1)
        query = self.query_projection(inputs)
        key   = self.key_projection(inputs)
        value = self.value_projection(inputs)
        attn = self.attention(query, key, value)
        out = self.norm1(inputs + attn)
        out = self.norm2(out + self.feedforward(out))
        out = self.output_projection(out)
        return out

class PositionalEncodingSin(torch.nn.Module):
    def __init__(self, dims, max_tokens=5000):
        super().__init__()
        pe = []
        for token in range(max_tokens):
            if token % 2: pe.append(torch.sin(torch.linspace(0, max_tokens-token+1, dims)))
            else:         pe.append(torch.cos(torch.linspace(0, max_tokens-token+1, dims)))
        pe = torch.concat(pe).reshape(max_tokens, dims)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1],:].unsqueeze(0)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, dims, max_tokens=5000):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_tokens, dims)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        return x + self.embedding(positions).unsqueeze(0)

class Dictionary(torch.nn.Module):
    def __init__(self, corpus):
        super().__init__()
        self.norm = r'#.*$'
        self.dictionary = {
            #'<pad>' : 0, ## Padding
            #'<start>' : 1, ## Start of Sequence
            #'<end>' : 2, ## End of Sequence
            #'<unknown>' : 3, ## Unknown Token
        }
        union_clip = set(self.dictionary.keys())
        self.vocab = sorted(set(self.normalize(corpus)) - union_clip)
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
    def __init__(self, dictionary, dims=256):
        super().__init__()
        self.dictionary = dictionary
        self.dims = dims
        self.number_of_words = number_of_words = len(dictionary)
        self.embedding = torch.nn.Embedding(number_of_words, dims)
        self.positional = PositionalEncoding(dims)
        self.stephen_transformer = StephenFormer(dictionary, dims=dims)
        #self.linear = torch.nn.Linear(dims, len(dictionary))
        #self.soft = torch.nn.Softmax()

    def forward(self, inputs):
        device = next(self.parameters()).device
        tokens = self.dictionary.tokenize(inputs).to(device)
        embeddings = self.embedding(tokens)
        pos_encoded = self.positional(embeddings)
        ## TODO mult-pass
        out = self.stephen_transformer(pos_encoded)
        return out

## Read self so we can learn self, and replicate
fine_tuning = Path(__file__).read_text()
pytorch_examples = Path('pytorch-training.py').read_text()
gpt_golf = Path('golf.py').read_text()
training_data = fine_tuning + pytorch_examples + gpt_golf
dictionary = Dictionary(training_data)
#print(dictionary)
#print(dictionary.decoder)
device = torch.accelerator.current_accelerator()
model = Transformer(dictionary).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
#criterion = torch.nn.NLLLoss(ignore_index=0)
epochs = 100000
batch_size = 512

## TODO rewrite for new data type
training_set = {
    'features' : [],
    'labels'   : [],
    'len'      : len(training_data),
    'window'   : 32, ## tokens
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
    loss = 100 
    for epoch in range(epochs):
        tokens = 0 # tokens per second
        start = time.time()
        for features, targets in get_batch():
            tokens += sum([len(sentence) for sentence in features])
            output = model(features)
            targets = dictionary.tokenize(targets).to(device)
            loss = criterion(output.transpose(1, 2), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        duration = time.time() - start

        print()
        print('tokens trained:', f'{tokens:,}')
        print('tokens per second:', f'{tokens//duration:,}')
        print('epoch loss:', f'{loss.item():.4f}')
        print('epoch duration:', f'{duration:.2f} seconds' )

        model.eval()
        predict()
        model.train()

## accuracy test
def predict():
    out_len = 100
    window = training_set['window']
    start = random.randint(0, len(training_data) - window - out_len - 1)
    inputs = training_data[start : start + window]
    print(inputs, end="", flush=True)
    for i in range(out_len):
        out = model([inputs])
        last_token = dictionary.decode(out)[0][-1]
        #print(words)
        print(last_token, end="", flush=True)
        #print(start)
        inputs = inputs[1:] + last_token
    print('')



batch_prepare()
#print(training_set)
try:
    model.load_state_dict(torch.load('gpt.pth'))
except Exception as e:
    print('FAILE TO LOAD')
try:
    train()
except KeyboardInterrupt:
    torch.save(model.state_dict(), 'gpt.pth')
    
torch.save(model.state_dict(), 'gpt.pth')
predict()
