import torch
import torch.nn as nn
import math

class extract_tensor(nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor

class Representation_function(nn.Module):
    def __init__(self,
                 observation_space_dimensions,
                 state_dimension,
                 action_dimension,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()

        self.state_norm = nn.Linear(observation_space_dimensions, state_dimension)
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

        lstm_reward = [
            nn.Linear(state_dimension + action_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions, state_dimension,number_of_hidden_layer),
            extract_tensor()
        ]

        lstm_state = [
            nn.Linear(state_dimension + action_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions, state_dimension,number_of_hidden_layer),
            extract_tensor(),
        ]


        self.reward = nn.Sequential(*tuple(lstm_reward))
        self.next_state_normalized = nn.Sequential(*tuple(lstm_state))

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

        lstm_policy = [
            nn.Linear(state_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions, action_dimension,number_of_hidden_layer),
            extract_tensor()
        ]

        lstm_value = [
            nn.Linear(state_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions , state_dimension,number_of_hidden_layer),
            extract_tensor(),
        ]


        self.policy = nn.Sequential(*tuple(lstm_policy))
        self.value = nn.Sequential(*tuple(lstm_value))

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


class Afterstate_dynamics_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension

        lstm_reward = [
            nn.Linear(state_dimension + action_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions, state_dimension,number_of_hidden_layer),
            extract_tensor()
        ]

        lstm_state = [
            nn.Linear(state_dimension + action_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions, state_dimension,number_of_hidden_layer),
            extract_tensor(),
        ]


        self.reward = nn.Sequential(*tuple(lstm_reward))
        self.next_state_normalized = nn.Sequential(*tuple(lstm_state))

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

        lstm_policy = [
            nn.Linear(state_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions, action_dimension,number_of_hidden_layer),
            extract_tensor()
        ]

        lstm_value = [
            nn.Linear(state_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions , state_dimension,number_of_hidden_layer),
            extract_tensor(),
        ]


        self.policy = nn.Sequential(*tuple(lstm_policy))
        self.value = nn.Sequential(*tuple(lstm_value))

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)





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