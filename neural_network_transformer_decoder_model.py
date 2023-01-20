import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),)

    def forward(self, x):
        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class decoder_only_transformer(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, num_positions, num_vocab, num_classes):
        super(decoder_only_transformer, self).__init__()

        self.embed_dim = embed_dim
        self.voc = num_vocab
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)
        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        length, batch = x.shape
        h = self.token_embeddings((x*1000).long())
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], axis=0)
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits.mean(-2)


class Representation_function(nn.Module):
    def __init__(self,
                 observation_space_dimensions,
                 state_dimension,
                 action_dimension,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension
        # # # add to sequence|first and recursive|,, whatever you need
        linear_in = nn.Linear(observation_space_dimensions, hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out = nn.Linear(hidden_layer_dimensions, state_dimension)
        
        self.scale = nn.Tanh()
        layernom_init = nn.BatchNorm1d(observation_space_dimensions)
        layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        # 0.1, 0.2 , 0.25 , 0.5 parameter (first two more recommended for rl)
        dropout = nn.Dropout(0.1)
        activation = nn.ELU()   # , nn.ELU() , nn.GELU, nn.ELU() , nn.ELU

        first_layer_sequence = [
            linear_in,
            activation
        ]

        recursive_layer_sequence = [
            linear_mid,
            activation
        ]

        sequence = first_layer_sequence + \
            (recursive_layer_sequence*number_of_hidden_layer)

        self.state_norm = nn.Sequential(*tuple(sequence+[nn.Linear(hidden_layer_dimensions, state_dimension)]))  
    def forward(self, state):
        return scale_to_bound_action(self.state_norm(state))


class Dynamics_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        
        self.action_space = action_dimension
        
        #embed_dim should be divisable by self.head : embed_dim % self.head == 0
        # head is the number of block
        self.head = 2
        
        # state (number between 0.0 and 1.0) get rescale to a range of 0 to 1000 integer to tokenize the state
        self.vocab = 1001
        
        self.batchsize = 128
        
        #the input size doesn't matter cause it get embedded 
        #self.embed_dim,self.voc  x shape torch.Size([1, 61]) param 128 1
        #embed_dim, num_heads, num_layers, num_positions, num_vocab, num_classes
        # hidden_layer_dimensions , self.head%hidden_layer_dimensions == 0 , number_of_hidden_layer , batchsize , 1001 , outputsize)
        self.reward = decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, state_dimension)
        self.next_state_normalized =  decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, state_dimension)


    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return self.reward(x), scale_to_bound_action(self.next_state_normalized(x))

class Prediction_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        
        self.head = 2
        self.vocab = 1001
        self.batchsize = 128
        print(f"Batch size is set to: {self.batchsize}")
        print(f"Your model must have the same batch size of {self.batchsize} or you have to change the batch size parameter in neural_network_transformer_decoder_model.py")
        self.policy = decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, action_dimension)
        self.value = decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, state_dimension)

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)

class Afterstate_dynamics_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        
        self.action_space = action_dimension
        
        #embed_dim should be divisable by self.head : embed_dim % self.head == 0
        # head is the number of block
        self.head = 2
        
        # state (number between 0.0 and 1.0) get rescale to a range of 0 to 1000 integer to tokenize the state
        self.vocab = 1001
        
        self.batchsize = 128
        
        #the input size doesn't matter cause it get embedded 
        #self.embed_dim,self.voc  x shape torch.Size([1, 61]) param 128 1
        #embed_dim, num_heads, num_layers, num_positions, num_vocab, num_classes
        # hidden_layer_dimensions , self.head%hidden_layer_dimensions == 0 , number_of_hidden_layer , batchsize , 1001 , outputsize)
        self.reward = decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, state_dimension)
        self.next_state_normalized =  decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, state_dimension)


    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return self.reward(x), scale_to_bound_action(self.next_state_normalized(x))

class Afterstate_prediction_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        
        self.head = 2
        self.vocab = 1001
        self.batchsize = 128
        print(f"Batch size is set to: {self.batchsize}")
        print(f"Your model must have the same batch size of {self.batchsize} or you have to change the batch size parameter in neural_network_transformer_decoder_model.py")
        self.policy = decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, action_dimension)
        self.value = decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, state_dimension)

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)
    

class Encoder_function(nn.Module):
    def __init__(self,
                 observation_space_dimensions,
                 state_dimension,
                 action_dimension,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension
        # # # add to sequence|first and recursive|,, whatever you need
        linear_in = nn.Linear(observation_space_dimensions, hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out = nn.Linear(hidden_layer_dimensions, state_dimension)
        
        self.scale = nn.Tanh()
        layernom_init = nn.BatchNorm1d(observation_space_dimensions)
        layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        # 0.1, 0.2 , 0.25 , 0.5 parameter (first two more recommended for rl)
        dropout = nn.Dropout(0.1)
        activation = nn.ELU()   # , nn.ELU() , nn.GELU, nn.ELU() , nn.ELU

        first_layer_sequence = [
            linear_in,
            activation
        ]

        recursive_layer_sequence = [
            linear_mid,
            activation
        ]

        sequence = first_layer_sequence + \
            (recursive_layer_sequence*number_of_hidden_layer)

        self.encoder = nn.Sequential(*tuple(sequence+[nn.Linear(hidden_layer_dimensions, action_dimension)]))  
        
    def forward(self, o_i):
        #https://openreview.net/pdf?id=X6D9bAHhBQ1 [page:5 chance outcome]
        c_e_t = torch.nn.Softmax(-1)(self.encoder(o_i))
        c_t = Onehot_argmax.apply(c_e_t)
        return c_t,c_e_t
    

def scale_to_bound_action(x):
    min_next_encoded_state = x.min(1, keepdim=True)[0]
    max_next_encoded_state = x.max(1, keepdim=True)[0]
    scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
    scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
    next_encoded_state_normalized = (
    x - min_next_encoded_state
    ) / scale_next_encoded_state
    return next_encoded_state_normalized 

#straight-through estimator is used during the backward to allow the gradients to flow only to the encoder during the backpropagation.
class Onehot_argmax(torch.autograd.Function):
    #more information at : https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    @staticmethod
    def forward(ctx, input):
        #since the codebook is constant ,we can just use a transformation. no need to create a codebook and matmul c_e_t and codebook for argmax
        return torch.zeros_like(input).scatter_(-1, torch.argmax(input, dim=-1,keepdim=True), 1.)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  
        



