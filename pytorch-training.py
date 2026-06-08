import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

torch.manual_seed(args.seed)

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# Build the model

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()
if args.use_optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if args.use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.use_optimizer:
            optimizer.step()
        else:
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    if args.model == 'Transformer':
        safe_globals = [
            PositionalEncoding,
            TransformerModel,
            torch.nn.functional.relu,
            torch.nn.modules.activation.MultiheadAttention,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
            torch.nn.modules.normalization.LayerNorm,
            torch.nn.modules.sparse.Embedding,
            torch.nn.modules.transformer.TransformerEncoder,
            torch.nn.modules.transformer.TransformerEncoderLayer,
        ]
    else:
        safe_globals = [
            RNNModel,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.rnn.GRU,
            torch.nn.modules.rnn.LSTM,
            torch.nn.modules.rnn.RNN,
            torch.nn.modules.sparse.Embedding,
        ]
    with torch.serialization.safe_globals(safe_globals):
        model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
import os
import time
import requests
import tarfile
import numpy as np
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


###  GAT LAYER DEFINITION

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.

        This operation can be mathematically described as:

            e_ij = a(W h_i, W h_j)
            α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k(exp(e_ik))     
            h_i' = σ(Σ_j(α_ij W h_j))
            
            where h_i and h_j are the feature vectors of nodes i and j respectively, W is a learnable weight matrix,
            a is an attention mechanism that computes the attention coefficients e_ij, and σ is an activation function.

    """
    def __init__(self, in_features: int, out_features: int, n_heads: int, concat: bool = False, dropout: float = 0.4, leaky_relu_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()

        self.n_heads = n_heads # Number of attention heads
        self.concat = concat # wether to concatenate the final attention heads
        self.dropout = dropout # Dropout rate

        if concat: # concatenating the attention heads
            self.out_features = out_features # Number of output features per node
            assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
            self.n_hidden = out_features // n_heads
        else: # averaging output over the attention heads (Used in the main paper)
            self.n_hidden = out_features

        #  A shared linear transformation, parametrized by a weight matrix W is applied to every node
        #  Initialize the weight matrix W 
        self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))

        # Initialize the attention weights a
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) # LeakyReLU activation function
        self.softmax = nn.Softmax(dim=1) # softmax activation function to the attention coefficients

        self.reset_parameters() # Reset the parameters


    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)
    

    def _get_attention_scores(self, h_transformed: torch.Tensor):
        """calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph
        in vectorized parallel form. for each pair of source and target nodes (i, j),
        the attention score e_ij is computed as follows:

            e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 

            where || denotes the concatenation operation, and a and W are the learnable parameters.

        Args:
            h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
                where n_nodes is the number of nodes and out_features is the number of output features per node.

        Returns:
            torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
        """
        
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])

        # broadcast add 
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        e = source_scores + target_scores.mT
        return self.leakyrelu(e)

    def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):
        """
        Performs a graph attention layer operation.

        Args:
            h (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """
        n_nodes = h.shape[0]

        # Apply linear transformation to node feature -> W h
        # output shape (n_nodes, n_hidden * n_heads)
        h_transformed = torch.mm(h, self.W)
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

        # splitting the heads by reshaping the tensor and putting heads dim first
        # output shape (n_heads, n_nodes, n_hidden)
        h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)
        
        # getting the attention scores
        # output shape (n_heads, n_nodes, n_nodes)
        e = self._get_attention_scores(h_transformed)

        # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
        connectivity_mask = -9e16 * torch.ones_like(e)
        e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores
        
        # attention coefficients are computed as a softmax over the rows
        # for each column j in the attention score matrix e
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # final node embeddings are computed as a weighted average of the features of its neighbors
        h_prime = torch.matmul(attention, h_transformed)

        # concatenating/averaging the attention heads
        # output shape (n_nodes, out_features)
        if self.concat:
            h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
        else:
            h_prime = h_prime.mean(dim=0)

        return h_prime

### MAIN GAT NETWORK MODULE  ###

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.
    Consists of a 2-layer stack of Graph Attention Layers (GATs). The fist GAT Layer is followed by an ELU activation.
    And the second (final) layer is a GAT layer with a single attention head and softmax activation function. 
    """
    def __init__(self,
        in_features,
        n_hidden,
        n_heads,
        num_classes,
        concat=False,
        dropout=0.4,
        leaky_relu_slope=0.2):
        """ Initializes the GAT model. 

        Args:
            in_features (int): number of input features per node.
            n_hidden (int): output size of the first Graph Attention Layer.
            n_heads (int): number of attention heads in the first Graph Attention Layer.
            num_classes (int): number of classes to predict for each node.
            concat (bool, optional): Wether to concatinate attention heads or take an average over them for the
                output of the first Graph Attention Layer. Defaults to False.
            dropout (float, optional): dropout rate. Defaults to 0.4.
            leaky_relu_slope (float, optional): alpha (slope) of the leaky relu activation. Defaults to 0.2.
        """

        super(GAT, self).__init__()

        # Define the Graph Attention layers
        self.gat1 = GraphAttentionLayer(
            in_features=in_features, out_features=n_hidden, n_heads=n_heads,
            concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )
        
        self.gat2 = GraphAttentionLayer(
            in_features=n_hidden, out_features=num_classes, n_heads=1,
            concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope
            )
        

    def forward(self, input_tensor: torch.Tensor , adj_mat: torch.Tensor):
        """
        Performs a forward pass through the network.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """

        # Apply the first Graph Attention layer
        x = self.gat1(input_tensor, adj_mat)
        x = F.elu(x) # Apply ELU activation function to the output of the first layer

        # Apply the second Graph Attention layer
        x = self.gat2(x, adj_mat)

        return F.log_softmax(x, dim=1) # Apply log softmax activation function

### LOADING THE CORA DATASET ###

def load_cora(path='./cora', device='cpu'):
    """
    Loads the Cora dataset. The dataset is downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz.

    """

    # Set the paths to the data files
    content_path = os.path.join(path, 'cora.content')
    cites_path = os.path.join(path, 'cora.cites')

    # Load data from files
    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    # Process features
    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32)) # Extract feature values
    scale_vector = torch.sum(features, dim=1) # Compute sum of features for each node
    scale_vector = 1 / scale_vector # Compute reciprocal of the sums
    scale_vector[scale_vector == float('inf')] = 0 # Handle division by zero cases
    scale_vector = torch.diag(scale_vector).to_sparse() # Convert the scale vector to a sparse diagonal matrix
    features = scale_vector @ features # Scale the features using the scale vector

    # Process labels
    classes, labels = np.unique(content_tensor[:, -1], return_inverse=True) # Extract unique classes and map labels to indices
    labels = torch.LongTensor(labels) # Convert labels to a tensor

    # Process adjacency matrix
    idx = content_tensor[:, 0].astype(np.int32) # Extract node indices
    idx_map = {id: pos for pos, id in enumerate(idx)} # Create a dictionary to map indices to positions

    # Map node indices to positions in the adjacency matrix
    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], 
            cites_tensor)), dtype=np.int32)

    V = len(idx) # Number of nodes
    E = edges.shape[0] # Number of edges
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64) # Create the initial adjacency matrix as a sparse tensor
    adj_mat = torch.eye(V) + adj_mat # Add self-loops to the adjacency matrix

    # return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)
    return features.to(device), labels.to(device), adj_mat.to(device)

### TRAIN AND TEST FUNCTIONS  ###

def train_iter(epoch, model, optimizer, criterion, input, target, mask_train, mask_val, print_every=10):
    start_t = time.time()
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(*input) 
    loss = criterion(output[mask_train], target[mask_train]) # Compute the loss using the training mask

    loss.backward()
    optimizer.step()

    # Evaluate the model performance on training and validation sets
    loss_train, acc_train = test(model, criterion, input, target, mask_train)
    loss_val, acc_val = test(model, criterion, input, target, mask_val)

    if epoch % print_every == 0:
        # Print the training progress at specified intervals
        print(f'Epoch: {epoch:04d} ({(time.time() - start_t):.4f}s) loss_train: {loss_train:.4f} acc_train: {acc_train:.4f} loss_val: {loss_val:.4f} acc_val: {acc_val:.4f}')


def test(model, criterion, input, target, mask):
    model.eval()
    with torch.no_grad():
        output = model(*input)
        output, target = output[mask], target[mask]

        loss = criterion(output, target)
        acc = (output.argmax(dim=1) == target).float().sum() / len(target)
    return loss.item(), acc.item()


if __name__ == '__main__':

    # Training settings
    # All defalut values are the same as in the config used in the main paper

    parser = argparse.ArgumentParser(description='PyTorch Graph Attention Network')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--l2', type=float, default=5e-4,
                        help='weight decay (default: 6e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.6,
                        help='dropout probability (default: 0.6)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='dimension of the hidden representation (default: 64)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='number of the attention heads (default: 4)')
    parser.add_argument('--concat-heads', action='store_true',
                        help='wether to concatinate attention heads, or average over them (default: False)')
    parser.add_argument('--val-every', type=int, default=20,
                        help='epochs to wait for print training and validation evaluation (default: 20)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', 
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=13, metavar='S',
                        help='random seed (default: 13)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_accel = not args.no_accel and torch.accelerator.is_available()

    # Set the device to run on
    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')
    print(f'Using {device} device')

    # Load the dataset
    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    path = './cora'

    if os.path.isfile(os.path.join(path, 'cora.content')) and os.path.isfile(os.path.join(path, 'cora.cites')):
        print('Dataset already downloaded...')
    else:
        print('Downloading dataset...')
        with requests.get(cora_url, stream=True) as tgz_file:
            with tarfile.open(fileobj=tgz_file.raw, mode='r:gz') as tgz_object:
                tgz_object.extractall()

    print('Loading dataset...')
    # Load the dataset
    features, labels, adj_mat = load_cora(device=device)
    # Split the dataset into training, validation, and test sets
    idx = torch.randperm(len(labels)).to(device)
    idx_test, idx_val, idx_train = idx[:1200], idx[1200:1600], idx[1600:]


    # Create the model
    # The model consists of a 2-layer stack of Graph Attention Layers (GATs).
    gat_net = GAT(
        in_features=features.shape[1],          # Number of input features per node  
        n_hidden=args.hidden_dim,               # Output size of the first Graph Attention Layer
        n_heads=args.num_heads,                 # Number of attention heads in the first Graph Attention Layer
        num_classes=labels.max().item() + 1,    # Number of classes to predict for each node
        concat=args.concat_heads,               # Wether to concatinate attention heads
        dropout=args.dropout_p,                 # Dropout rate
        leaky_relu_slope=0.2                    # Alpha (slope) of the leaky relu activation
    ).to(device)

    # configure the optimizer and loss function
    optimizer = Adam(gat_net.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.NLLLoss()

    # Train and evaluate the model
    for epoch in range(args.epochs):
        train_iter(epoch + 1, gat_net, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val, args.val_every)
        if args.dry_run:
            break
    loss_test, acc_test = test(gat_net, criterion, (features, adj_mat), labels, idx_test)
    print(f'Test set results: loss {loss_test:.4f} accuracy {acc_test:.4f}')from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=64, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, input):
        # Shape of input is (batch_size,1, 28, 28)
        # converting shape of input to (batch_size, 28, 28)
        # as required by RNN when batch_first is set True
        input = input.reshape(-1, 28, 28)
        output, hidden = self.rnn(input)

        # RNN output shape is (seq_len, batch, input_size)
        # Get last output of RNN
        output = output[:, -1, :]
        output = self.batchnorm(output)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.dropout2(output)
        output = self.fc2(output)
        output = F.log_softmax(output, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if args.dry_run:
                break

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example using RNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--accel', action='store_true',
                        help='enables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='for Saving the current Model')
    args = parser.parse_args()

    if args.accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.accel else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_rnn.pt")


if __name__ == '__main__':
    main()
import os
import sys
import torch
from torch.utils.data import random_split
from torch.distributed import init_process_group, destroy_process_group
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from trainer import Trainer, TrainerConfig
from char_dataset import CharDataset, DataConfig
from omegaconf import DictConfig
import hydra

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

def ddp_setup():
    acc = torch.accelerator.current_accelerator()
    rank = int(os.environ["LOCAL_RANK"])
    device: torch.device = torch.device(f"{acc}:{rank}")
    backend = torch.distributed.get_default_backend_for_device(device)
    init_process_group(backend=backend)
    torch.accelerator.set_device_index(rank)

def get_train_objs(gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)
    
    return model, optimizer, train_set, test_set

@hydra.main(version_base=None, config_path=".", config_name="gpt2_train_cfg")
def main(cfg: DictConfig):
    ddp_setup()

    gpt_cfg = GPTConfig(**cfg['gpt_config'])
    opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
    data_cfg = DataConfig(**cfg['data_config'])
    trainer_cfg = TrainerConfig(**cfg['trainer_config'])

    model, optimizer, train_data, test_data = get_train_objs(gpt_cfg, opt_cfg, data_cfg)
    trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)
    trainer.train()

    destroy_process_group()

if __name__ == "__main__":
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        sys.exit()
    main()
import os
import threading
import time
import warnings
from functools import wraps

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.rpc import RRef

from torchvision.models.resnet import Bottleneck

# Suppress warnings that can't be fixed from user code
warnings.filterwarnings("ignore", 
    message="You are using a Backend .* as a ProcessGroup. This usage is deprecated", 
    category=UserWarning)
warnings.filterwarnings("ignore", 
    message="networkx backend defined more than once: nx-loopback", 
    category=RuntimeWarning)


#           Define Model Parallel ResNet50              #

# In order to split the ResNet50 and place it on two different workers, we
# implement it in two model shards. The ResNetBase class defines common
# attributes and methods shared by two shards. ResNetShard1 and ResNetShard2
# contain two partitions of the model layers respectively.


num_classes = 1000


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetBase(nn.Module):
    def __init__(self, block, inplanes, num_classes=1000,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNetBase, self).__init__()

        self._lock = threading.Lock()
        self._block = block
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self._block.expansion, stride),
                norm_layer(planes * self._block.expansion),
            )

        layers = []
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,
                                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class ResNetShard1(ResNetBase):
    """
    The first part of ResNet.
    """
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(
            Bottleneck, 64, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 3),
            self._make_layer(128, 4, stride=2)
        ).to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out =  self.seq(x)
        return out.cpu()


class ResNetShard2(ResNetBase):
    """
    The second part of ResNet.
    """
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(
            Bottleneck, 512, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            self._make_layer(256, 6, stride=2),
            self._make_layer(512, 3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).to(self.device)

        self.fc =  nn.Linear(512 * self._block.expansion, num_classes).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.fc(torch.flatten(self.seq(x), 1))
        return out.cpu()


class DistResNet50(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """
    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()

        self.split_size = split_size

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            ResNetShard1,
            args = ("cuda:0",) + args,
            kwargs = kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            ResNetShard2,
            args = ("cuda:1",) + args,
            kwargs = kwargs
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params


#                   Run RPC Processes                   #

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def create_optimizer_for_remote_params(worker_name, param_rrefs, lr=0.05):
    """Create torch.compiled optimizers on  each worker"""
    params = [p.to_here() for p in param_rrefs]
    opt = optim.SGD(params, lr=lr)
    opt.step = torch.compile(opt.step)
    return opt


def run_master(split_size):

    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet50(split_size, ["worker1", "worker2"])
    loss_fn = nn.MSELoss()
    
    # Get parameter RRefs for each model shard
    p1_param_rrefs = model.p1_rref.remote().parameter_rrefs().to_here()
    p2_param_rrefs = model.p2_rref.remote().parameter_rrefs().to_here()
    
    # Create optimizers on remote workers
    opt1_rref = rpc.remote(
        "worker1",
        create_optimizer_for_remote_params,
        args=("worker1", p1_param_rrefs)
    )
    opt2_rref = rpc.remote(
        "worker2",
        create_optimizer_for_remote_params,
        args=("worker2", p2_param_rrefs)
    )

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for i in range(num_batches):
        print(f"Processing batch {i}")
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:
            outputs = model(inputs)
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            
            opt1_rref.rpc_sync().step()
            opt2_rref.rpc_sync().step()
            
            opt1_rref.rpc_sync().zero_grad()
            opt2_rref.rpc_sync().zero_grad()


def run_worker(rank, world_size, num_split):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    # Suppress torch compile profiler warnings
    os.environ['TORCH_LOGS'] = '-dynamo'
    
    world_size = 3
    for num_split in [1, 2, 4, 8]:
        tik = time.time()
        mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
        tok = time.time()
        print(f"number of splits = {num_split}, execution time = {tok - tik}")
import random

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP

NUM_EMBEDDINGS = 100
EMBEDDING_DIM = 16

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

class HybridModel(torch.nn.Module):
    r"""
    The model consists of a sparse part and a dense part.
    1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
    2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """

    def __init__(self, remote_emb_module, rank):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(torch.nn.Linear(16, 8).to(rank))
        self.rank = rank

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup.to(self.rank))


def _run_trainer(remote_emb_module, rank):
    r"""
    Each trainer runs a forward pass which involves an embedding lookup on the
    parameter server and running nn.Linear locally. During the backward pass,
    DDP is responsible for aggregating the gradients for the dense part
    (nn.Linear) and distributed autograd ensures gradients updates are
    propagated to the parameter server.
    """

    # Setup the model.
    model = HybridModel(remote_emb_module, rank)

    # Retrieve all model parameters as rrefs for DistributedOptimizer.

    # Retrieve parameters for embedding table.
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()

    # model.fc.parameters() only includes local parameters.
    # NOTE: Cannot call model.parameters() here,
    # because this will call remote_emb_module.parameters(),
    # which supports remote_parameters() but not parameters().
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch(rank):
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # Generate offsets.
            offsets = []
            start = 0
            batch_size = 0
            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            target = torch.LongTensor(batch_size).random_(8).to(rank)
            yield indices, offsets_tensor, target

    # Train for 100 epochs
    for epoch in range(100):
        # create distributed autograd context
        for indices, offsets, target in get_next_batch(rank):
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])

                # Tun distributed optimizer
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training done for epoch {}".format(epoch))


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """

    # We need to use different port numbers in TCP init_method for init_rpc and
    # init_process_group to avoid port conflicts.
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # Rank 2 is master, 3 is ps and 0 and 1 are trainers.
    if rank == 2:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        remote_emb_module = RemoteModule(
            "ps",
            torch.nn.EmbeddingBag,
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
            kwargs={"mode": "sum"},
        )

        # Run the training loop on trainers.
        futs = []
        for trainer_rank in [0, 1]:
            trainer_name = "trainer{}".format(trainer_rank)
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(remote_emb_module, trainer_rank)
            )
            futs.append(fut)

        # Wait for all training to finish.
        for fut in futs:
            fut.wait()
    elif rank <= 1:
        acc = torch.accelerator.current_accelerator()
        device = torch.device(acc)
        backend = torch.distributed.get_default_backend_for_device(device)
        torch.accelerator.device_index(rank)
        # Initialize process group for Distributed DataParallel on trainers.
        dist.init_process_group(
            backend=backend, rank=rank, world_size=2, init_method="tcp://localhost:29500"
        )

        # Initialize RPC.
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # Trainer just waits for RPCs from master.
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()
    
    # Clean up process group for trainers to avoid resource leaks
    if rank <= 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # 2 trainers, 1 parameter server, 1 master.
    world_size = 4
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        exit()
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
    print("Distributed RPC example completed successfully.")
import argparse
import gymnasium as gym
import numpy as np
import os
from itertools import count

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical

TOTAL_EPISODE_STEP = 5000
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

parser = argparse.ArgumentParser(description='PyTorch RPC RL example')
parser.add_argument('--world-size', type=int, default=2, metavar='W',
                    help='world size for RPC, rank 0 is the agent, others are observers')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


class Policy(nn.Module):
    r"""
    Borrowing the ``Policy`` class from the Reinforcement Learning example.
    Copying the code to make these two examples independent.
    See https://github.com/pytorch/examples/tree/main/reinforcement_learning
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class Observer:
    r"""
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.

    It is true that CartPole-v1 is a relatively inexpensive environment, and it
    might be an overkill to use RPC to connect observers and trainers in this
    specific use case. However, the main goal of this tutorial to how to build
    an application using the RPC API. Developers can extend the similar idea to
    other applications with much more expensive environment.
    """
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make('CartPole-v1')
        self.env.reset(seed=args.seed)

    def run_episode(self, agent_rref, n_steps):
        r"""
        Run one episode of n_steps.

        Args:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """
        state, ep_reward = self.env.reset()[0], 0
        for step in range(n_steps):
            # send the state to the agent to get an action
            action = _remote_method(Agent.select_action, agent_rref, self.id, state)

            # apply the action to the environment, and get the reward
            state, reward, terminated, truncated, _ = self.env.step(action)

            # report the reward to the agent for training purpose
            _remote_method(Agent.report_reward, agent_rref, self.id, reward)

            if terminated or truncated:
                break

class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

    def select_action(self, ob_id, state):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/main/reinforcement_learning
        The main difference is that instead of keeping all probs in one list,
        the agent keeps probs in a dictionary, one key per observer.

        NB: no need to enforce thread-safety here as GIL will serialize
        executions.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        r"""
        Observers call this function to report rewards.
        """
        self.rewards[ob_id].append(reward)

    def run_episode(self, n_steps=0):
        r"""
        Run one episode. The agent will tell each oberser to run n_steps.
        """
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args=(Observer.run_episode, ob_rref, self.agent_rref, n_steps)
                )
            )

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

    def finish_episode(self):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/main/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        """

        # joins probs and rewards from different observers into lists
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])

        # use the minimum observer reward to calculate the running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

        # clear saved probs and rewards
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []

        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        return min_reward


def run_worker(rank, world_size):
    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 0:
        # rank0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)

        agent = Agent(world_size)
        for i_episode in count(1):
            n_steps = int(TOTAL_EPISODE_STEP / (args.world_size - 1))
            agent.run_episode(n_steps=n_steps)
            last_reward = agent.finish_episode()

            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, last_reward, agent.running_reward))

            if agent.running_reward > agent.reward_threshold:
                print("Solved! Running reward is now {}!".format(agent.running_reward))
                break
    else:
        # other ranks are the observer
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # observers passively waiting for instructions from agents
    rpc.shutdown()


def main():
    mp.spawn(
        run_worker,
        args=(args.world_size, ),
        nprocs=args.world_size,
        join=True
    )

if __name__ == '__main__':
    main()
import os

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer

import rnn


def _run_trainer():
    r"""
    The trainer creates a distributed RNNModel and a DistributedOptimizer. Then,
    it performs training using random input data.
    """
    batch = 5
    ntoken = 7
    ninp = 2

    nhid = 3
    nindices = 6
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def get_next_batch():
        for _ in range(5):
            data = torch.LongTensor(batch, nindices) % ntoken
            target = torch.LongTensor(batch, ntoken) % nindices
            yield data, target

    # train for 10 iterations
    for epoch in range(10):
        # create distributed autograd context
        for data, target in get_next_batch():
            with dist_autograd.context() as context_id:
                hidden[0].detach_()
                hidden[1].detach_()
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                # run distributed backward pass
                dist_autograd.backward(context_id, [loss])
                # run distributed optimizer
                opt.step(context_id)
                # not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training epoch {}".format(epoch))


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    if rank == 1:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        _run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # parameter server does nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)
from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--accel', action='store_true', help='Enables acceleration for training, if available')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

torch.manual_seed(opt.seed)


if opt.accel and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--accel', action='store_true', default=False, help='enables accelerator')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if opt.accel and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        
        if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.dry_run:
    opt.niter = 1

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

        if opt.dry_run:
            break
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--no-accel', action='store_true',
                    help='disables accelerator')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type =='cuda':
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        if args.gpu is not None:
            torch.accelerator.set_device_index(args.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not use_accel:
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if device.type == 'cuda':
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(device)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)


    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'{device.type}:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    
    use_accel = not args.no_accel and torch.accelerator.is_available()

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.NONE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.NONE)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)


def validate(val_loader, model, criterion, args):

    use_accel = not args.no_accel and torch.accelerator.is_available()

    def run_validate(loader, base_progress=0):

        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if use_accel:
                    if args.gpu is not None and device.type=='cuda':
                        torch.accelerator.set_device_index(args.gpu)
                        images = images.cuda(args.gpu, non_blocking=True)
                        target = target.cuda(args.gpu, non_blocking=True)
                    else:
                        images = images.to(device)
                        target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):    
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
# This code is based on the implementation of Mohammad Pezeshki available at
# https://github.com/mohammadpz/pytorch_forward_forward and licensed under the MIT License.
# Modifications/Improvements to the original code have been made by Vivek V Patel.

import argparse
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.optim import Adam


def get_y_neg(y):
    y_neg = y.clone()
    for idx, y_samp in enumerate(y):
        allowed_indices = list(range(10))
        allowed_indices.remove(y_samp.item())
        y_neg[idx] = torch.tensor(allowed_indices)[
            torch.randint(len(allowed_indices), size=(1,))
        ].item()
    return y_neg.to(device)


def overlay_y_on_x(x, y, classes=10):
    x_ = x.clone()
    x_[:, :classes] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):
    def __init__(self, dims):

        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers = self.layers + [Layer(dims[d], dims[d + 1]).to(device)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness = goodness + [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print("training layer: ", i)
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=args.lr)
        self.threshold = args.threshold
        self.num_epochs = args.epochs

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log1p(
                torch.exp(
                    torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
                )
            ).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if i % args.log_interval == 0:
                print("Loss: ", loss.item())
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        metavar="N",
        help="number of epochs to train (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        metavar="LR",
        help="learning rate (default: 0.03)",
    )
    parser.add_argument(
        "--no_accel", action="store_true", help="disables accelerator"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="For saving the current Model",
    )
    parser.add_argument(
        "--train_size", type=int, default=50000, help="size of training set"
    )
    parser.add_argument(
        "--threshold", type=float, default=2, help="threshold for training"
    )
    parser.add_argument("--test_size", type=int, default=10000, help="size of test set")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()
    use_accel = not args.no_accel and torch.accelerator.is_available()
    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.train_size}
    test_kwargs = {"batch_size": args.test_size}

    if use_accel:
        accel_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x)),
        ]
    )
    train_loader = DataLoader(
        MNIST("./data/", train=True, download=True, transform=transform), **train_kwargs
    )
    test_loader = DataLoader(
        MNIST("./data/", train=False, download=True, transform=transform), **test_kwargs
    )
    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    x_pos = overlay_y_on_x(x, y)
    y_neg = get_y_neg(y)
    x_neg = overlay_y_on_x(x, y_neg)
    net.train(x_pos, x_neg)
    print("train error:", 1.0 - net.predict(x).eq(y).float().mean().item())
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)
    if args.save_model:
        torch.save(net.state_dict(), "mnist_ff.pt")
    print("test error:", 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output

class APP_MATCHER(Dataset):
    def __init__(self, root, train, download=False):
        super(APP_MATCHER, self).__init__()

        # get MNIST dataset
        self.dataset = datasets.MNIST(root, train=train, download=download)
        
        # as `self.dataset.data`'s shape is (Nx28x28), where N is the number of
        # examples in MNIST dataset, a single example has the dimensions of
        # (28x28) for (WxH), where W and H are the width and the height of the image. 
        # However, every example should have (CxWxH) dimensions where C is the number 
        # of channels to be passed to the network. As MNIST contains gray-scale images, 
        # we add an additional dimension to corresponds to the number of channels.
        self.data = self.dataset.data.unsqueeze(1).clone()

        self.group_examples()

    def group_examples(self):
        """
            To ease the accessibility of data based on the class, we will use `group_examples` to group 
            examples based on class. 
            
            Every key in `grouped_examples` corresponds to a class in MNIST dataset. For every key in 
            `grouped_examples`, every value will conform to all of the indices for the MNIST 
            dataset examples that correspond to that key.
        """

        # get the targets from MNIST dataset
        np_arr = np.array(self.dataset.targets.clone(), dtype=None, copy=None)
        
        # group examples based on class
        self.grouped_examples = {}
        for i in range(0,10):
            self.grouped_examples[i] = np.where((np_arr==i))[0]
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases, 
            positive and negative examples. For positive examples, we will have two 
            images from the same class. For negative examples, we will have two images 
            from different classes.

            Given an index, if the index is even, we will pick the second image from the same class, 
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn 
            the similarity between two different images representing the same class. If the index is odd, we will 
            pick the second image from a different class than the first image.
        """

        # pick some random class for the first image
        selected_class = random.randint(0, 9)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
        
        # pick the index to get the first image
        index_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        image_1 = self.data[index_1].clone().float()

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            
            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            
            # pick the index to get the second image
            index_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)
        
        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, 9)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, 9)

            
            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0]-1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # for the 1st epoch, the average loss is 0.0001 and the accuracy 97-98%
    # using default settings. After completing the 10th epoch, the average
    # loss is 0.0000 and the accuracy 99.5-100% using default settings.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true', 
                    help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)


    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    train_dataset = APP_MATCHER('../data', train=True, download=True)
    test_dataset = APP_MATCHER('../data', train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "siamese_network.pt")


if __name__ == '__main__':
    main()
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms

from train import train, test

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--mps', action='store_true', default=False,
                    help='enables macOS GPU training')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save the trained model to state_dict')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    use_mps = args.mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    kwargs = {'batch_size': args.batch_size,
              'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn', force=True)

    model = Net().to(device)
    model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device,
                                           dataset1, kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if args.save_model:
        torch.save(model.state_dict(), "MNIST_hogwild.pt")

    # Once training is complete, we can test the model
    test(args, model, device, dataset2, kwargs)
from time import time # Track how long an epoch takes
import os # Creating and finding files/directories
import logging # Logging tools
from datetime import date # Logging the date for model versioning

import torch # For ML
from tqdm import tqdm # For fancy progress bars

from src.model import Translator # Our model
from src.data import get_data, create_mask, generate_square_subsequent_mask # Loading data and data preprocessing
from argparse import ArgumentParser # For args

# Train on the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):

    # Move to device
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    # Encode input
    memory = model.encode(src, src_mask)

    # Output will be stored here
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    # For each element in our translation (which could range up to the maximum translation length)
    for _ in range(max_len-1):

        # Decode the encoded representation of the input
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), DEVICE).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)

        # Reshape
        out = out.transpose(0, 1)

        # Covert to probabilities and take the max of these probabilities
        prob = model.ff(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # Now we have an output which is the vector representation of the translation
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break

    return ys

# Opens an user interface where users can translate an arbitrary sentence
def inference(opts):

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    _, _, src_vocab, tgt_vocab, src_transform, _, special_symbols = get_data(opts)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # Create model
    model = Translator(
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        embed_size=opts.embed_size,
        num_heads=opts.attn_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout
    ).to(DEVICE)

    # Load in weights
    model.load_state_dict(torch.load(opts.model_path))

    # Set to inference
    model.eval()

    # Accept input and keep translating until they quit
    while True:
        print("> ", end="")

        sentence = input()

        # Convert to tokens
        src = src_transform(sentence).view(-1, 1)
        num_tokens = src.shape[0]

        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        # Decode
        tgt_tokens = greedy_decode(
            model, src, src_mask, max_len=num_tokens+5, start_symbol=special_symbols["<bos>"], end_symbol=special_symbols["<eos>"]
        ).flatten()

        # Convert to list of tokens
        output_as_list = list(tgt_tokens.cpu().numpy())

        # Convert tokens to words
        output_list_words = tgt_vocab.lookup_tokens(output_as_list)

        # Remove special tokens and convert to string
        translation = " ".join(output_list_words).replace("<bos>", "").replace("<eos>", "")

        print(translation)

# Train the model for 1 epoch
def train(model, train_dl, loss_fn, optim, special_symbols, opts):

    # Object for accumulating losses
    losses = 0

    # Put model into training mode
    model.train()
    for src, tgt in tqdm(train_dl, ascii=True):

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # Reset gradients before we try to compute the gradients over the loss
        optim.zero_grad()

        # Get original shape back
        tgt_out = tgt[1:, :]

        # Compute loss and gradient over that loss
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        # Step weights
        optim.step()

        # Accumulate a running loss for reporting
        losses += loss.item()

        if opts.dry_run:
            break

    # Return the average loss
    return losses / len(list(train_dl))

# Check the model accuracy on the validation dataset
def validate(model, valid_dl, loss_fn, special_symbols):
    
    # Object for accumulating losses
    losses = 0

    # Turn off gradients a moment
    model.eval()

    for src, tgt in tqdm(valid_dl):

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        tgt_input = tgt[:-1, :]

        # Create masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # Get original shape back, compute loss, accumulate that loss
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    # Return the average loss
    return losses / len(list(valid_dl))

# Train the model
def main(opts):

    # Set up logging
    os.makedirs(opts.logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=opts.logging_dir + "log.txt", level=logging.INFO)

    # This prints it to the screen as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info(f"Translation task: {opts.src} -> {opts.tgt}")
    logging.info(f"Using device: {DEVICE}")

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset
    train_dl, valid_dl, src_vocab, tgt_vocab, _, _, special_symbols = get_data(opts)

    logging.info("Loaded data")

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    logging.info(f"{opts.src} vocab size: {src_vocab_size}")
    logging.info(f"{opts.tgt} vocab size: {tgt_vocab_size}")

    # Create model
    model = Translator(
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        embed_size=opts.embed_size,
        num_heads=opts.attn_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout
    ).to(DEVICE)

    logging.info("Model created... starting training!")

    # Set up our learning tools
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = 1e6
    
    for idx, epoch in enumerate(range(1, opts.epochs+1)):

        start_time = time()
        train_loss = train(model, train_dl, loss_fn, optim, special_symbols, opts)
        epoch_time = time() - start_time
        val_loss   = validate(model, valid_dl, loss_fn, special_symbols)

        # Once training is done, we want to save out the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logging.info("New best model, saving...")
            torch.save(model.state_dict(), opts.logging_dir + "best.pt")

        torch.save(model.state_dict(), opts.logging_dir + "last.pt")

        logger.info(f"Epoch: {epoch}\n\tTrain loss: {train_loss:.3f}\n\tVal loss: {val_loss:.3f}\n\tEpoch time = {epoch_time:.1f} seconds\n\tETA = {epoch_time*(opts.epochs-idx-1):.1f} seconds")

if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Machine Translator training and inference",
    )

    # Inference mode
    parser.add_argument("--inference", action="store_true",
                        help="Set true to run inference")
    parser.add_argument("--model_path", type=str,
                        help="Path to the model to run inference on")

    # Translation settings
    parser.add_argument("--src", type=str, default="de",
                        help="Source language (translating FROM this language)")
    parser.add_argument("--tgt", type=str, default="en",
                        help="Target language (translating TO this language)")

    # Training settings 
    parser.add_argument("-e", "--epochs", type=int, default=30,
                        help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Default learning rate")
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--backend", type=str, default="cpu",
                        help="Batch size")
    
    # Transformer settings
    parser.add_argument("--attn_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=5,
                        help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=5,
                        help="Number of decoder layers")
    parser.add_argument("--embed_size", type=int, default=512,
                        help="Size of the language embedding")
    parser.add_argument("--dim_feedforward", type=int, default=512,
                        help="Feedforward dimensionality")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout")

    # Logging settings
    parser.add_argument("--logging_dir", type=str, default="./" + str(date.today()) + "/",
                        help="Where the output of this program should be placed")

    # Just for continuous integration
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if args.backend == "gpu" and torch.cuda.is_available() else "cpu")

    if args.inference:
        inference(args)
    else:
        main(args)
#!/usr/bin/env python
from __future__ import print_function
from itertools import count

import torch
import torch.nn.functional as F

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target.item()


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, i + 1)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y


# Define model
fc = torch.nn.Linear(W_target.size(0), 1)

for batch_idx in count(1):
    # Get data
    batch_x, batch_y = get_batch()

    # Reset gradients
    fc.zero_grad()

    # Forward pass
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.item()

    # Backward pass
    output.backward()

    # Apply gradients
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad)

    # Stop criterion
    if loss < 1e-3:
        break

print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
import os
import time
import requests
import tarfile
import numpy as np
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


class GraphConv(nn.Module):
    """
        Graph Convolutional Layer described in "Semi-Supervised Classification with Graph Convolutional Networks".

        Given an input feature representation for each node in a graph, the Graph Convolutional Layer aims to aggregate
        information from the node's neighborhood to update its own representation. This is achieved by applying a graph
        convolutional operation that combines the features of a node with the features of its neighboring nodes.

        Mathematically, the Graph Convolutional Layer can be described as follows:

            H' = f(D^(-1/2) * A * D^(-1/2) * H * W)

        where:
            H: Input feature matrix with shape (N, F_in), where N is the number of nodes and F_in is the number of 
                input features per node.
            A: Adjacency matrix of the graph with shape (N, N), representing the relationships between nodes.
            W: Learnable weight matrix with shape (F_in, F_out), where F_out is the number of output features per node.
            D: The degree matrix.
    """
    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConv, self).__init__()

        # Initialize the weight matrix W (in this case called `kernel`)
        self.kernel = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_normal_(self.kernel) # Initialize the weights using Xavier initialization

        # Initialize the bias (if use_bias is True)
        self.bias = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias) # Initialize the bias to zeros

    def forward(self, input_tensor, adj_mat):
        """
        Performs a graph convolution operation.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Normalized adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """

        support = torch.mm(input_tensor, self.kernel) # Matrix multiplication between input and weight matrix
        output = torch.spmm(adj_mat, support) # Sparse matrix multiplication between adjacency matrix and support
        # Add the bias (if bias is not None)
        if self.bias is not None:
            output = output + self.bias

        return output


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) as described in the paper `"Semi-Supervised Classification with Graph 
    Convolutional Networks" <https://arxiv.org/pdf/1609.02907.pdf>`.

    The Graph Convolutional Network is a deep learning architecture designed for semi-supervised node 
    classification tasks on graph-structured data. It leverages the graph structure to learn node representations 
    by propagating information through the graph using graph convolutional layers.

    The original implementation consists of two stacked graph convolutional layers. The ReLU activation function is 
    applied to the hidden representations, and the Softmax activation function is applied to the output representations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True, dropout_p=0.1):
        super(GCN, self).__init__()

        # Define the Graph Convolution layers
        self.gc1 = GraphConv(input_dim, hidden_dim, use_bias=use_bias)
        self.gc2 = GraphConv(hidden_dim, output_dim, use_bias=use_bias)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor, adj_mat):
        """
        Performs forward pass of the Graph Convolutional Network (GCN).

        Args:
            input_tensor (torch.Tensor): Input node feature matrix with shape (N, input_dim), where N is the number of nodes
                and input_dim is the number of input features per node.
            adj_mat (torch.Tensor): Normalized adjacency matrix of the graph with shape (N, N), representing the relationships between
                nodes.

        Returns:
            torch.Tensor: Output tensor with shape (N, output_dim), representing the predicted class probabilities for each node.
        """

        # Perform the first graph convolutional layer
        x = self.gc1(input_tensor, adj_mat)
        x = F.relu(x) # Apply ReLU activation function
        x = self.dropout(x) # Apply dropout regularization

        # Perform the second graph convolutional layer
        x = self.gc2(x, adj_mat)

        # Apply log-softmax activation function for classification
        return F.log_softmax(x, dim=1)


def load_cora(path='./cora', device='cpu'):
    """
    The graph convolutional operation rquires the normalized adjacency matrix: D^(-1/2) * A * D^(-1/2). This step 
    scales the adjacency matrix such that the features of neighboring nodes are weighted appropriately during 
    aggregation. The steps involved in the renormalization trick are as follows:
        - Compute the degree matrix.
        - Compute the inverse square root of the degree matrix.
        - Multiply the inverse square root of the degree matrix with the adjacency matrix.
    """

    # Set the paths to the data files
    content_path = os.path.join(path, 'cora.content')
    cites_path = os.path.join(path, 'cora.cites')

    # Load data from files
    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    # Process features
    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32)) # Extract feature values
    scale_vector = torch.sum(features, dim=1) # Compute sum of features for each node
    scale_vector = 1 / scale_vector # Compute reciprocal of the sums
    scale_vector[scale_vector == float('inf')] = 0 # Handle division by zero cases
    scale_vector = torch.diag(scale_vector).to_sparse() # Convert the scale vector to a sparse diagonal matrix
    features = scale_vector @ features # Scale the features using the scale vector

    # Process labels
    classes, labels = np.unique(content_tensor[:, -1], return_inverse=True) # Extract unique classes and map labels to indices
    labels = torch.LongTensor(labels) # Convert labels to a tensor

    # Process adjacency matrix
    idx = content_tensor[:, 0].astype(np.int32) # Extract node indices
    idx_map = {id: pos for pos, id in enumerate(idx)} # Create a dictionary to map indices to positions

    # Map node indices to positions in the adjacency matrix
    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], 
            cites_tensor)), dtype=np.int32)

    V = len(idx) # Number of nodes
    E = edges.shape[0] # Number of edges
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64) # Create the initial adjacency matrix as a sparse tensor
    adj_mat = torch.eye(V) + adj_mat # Add self-loops to the adjacency matrix

    degree_mat = torch.sum(adj_mat, dim=1) # Compute the sum of each row in the adjacency matrix (degree matrix)
    degree_mat = torch.sqrt(1 / degree_mat) # Compute the reciprocal square root of the degrees
    degree_mat[degree_mat == float('inf')] = 0 # Handle division by zero cases
    degree_mat = torch.diag(degree_mat).to_sparse() # Convert the degree matrix to a sparse diagonal matrix

    adj_mat = degree_mat @ adj_mat @ degree_mat # Apply the renormalization trick

    return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)

def train_iter(epoch, model, optimizer, criterion, input, target, mask_train, mask_val, print_every=10):
    start_t = time.time()
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(*input) 
    loss = criterion(output[mask_train], target[mask_train]) # Compute the loss using the training mask

    loss.backward()
    optimizer.step()

    # Evaluate the model performance on training and validation sets
    loss_train, acc_train = test(model, criterion, input, target, mask_train)
    loss_val, acc_val = test(model, criterion, input, target, mask_val)

    if epoch % print_every == 0:
        # Print the training progress at specified intervals
        print(f'Epoch: {epoch:04d} ({(time.time() - start_t):.4f}s) loss_train: {loss_train:.4f} acc_train: {acc_train:.4f} loss_val: {loss_val:.4f} acc_val: {acc_val:.4f}')


def test(model, criterion, input, target, mask):
    model.eval()
    with torch.no_grad():
        output = model(*input)
        output, target = output[mask], target[mask]

        loss = criterion(output, target)
        acc = (output.argmax(dim=1) == target).float().sum() / len(target)
    return loss.item(), acc.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Graph Convolutional Network')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--l2', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--hidden-dim', type=int, default=16,
                        help='dimension of the hidden representation (default: 16)')
    parser.add_argument('--val-every', type=int, default=20,
                        help='epochs to wait for print training and validation evaluation (default: 20)')
    parser.add_argument('--include-bias', action='store_true',
                        help='use bias term in convolutions (default: False)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')

    print(f'Using {device} device')

    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    print('Downloading dataset...')
    with requests.get(cora_url, stream=True) as tgz_file:
        with tarfile.open(fileobj=tgz_file.raw, mode='r:gz') as tgz_object:
            tgz_object.extractall()

    print('Loading dataset...')
    features, labels, adj_mat = load_cora(device=device)
    idx = torch.randperm(len(labels)).to(device)
    idx_test, idx_val, idx_train = idx[:1000], idx[1000:1500], idx[1500:]

    gcn = GCN(features.shape[1], args.hidden_dim, labels.max().item() + 1, args.include_bias, args.dropout_p).to(device)
    optimizer = Adam(gcn.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.NLLLoss()

    for epoch in range(args.epochs):
        train_iter(epoch + 1, gcn, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val, args.val_every)
        if args.dry_run:
            break

    loss_test, acc_test = test(gcn, criterion, (features, adj_mat), labels, idx_test)
    print(f'Test set results: loss {loss_test:.4f} accuracy {acc_test:.4f}')from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-accel', action='store_true', 
                    help='disables accelerator')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

use_accel = not args.no_accel and torch.accelerator.is_available()

torch.manual_seed(args.seed)


if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
