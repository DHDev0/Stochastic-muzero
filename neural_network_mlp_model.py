import torch
import torch.nn as nn
import math

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
        # self.state_norm = nn.Linear(observation_space_dimensions, state_dimension)
    def forward(self, state):
        return scale_to_bound_action(self.state_norm(state))




class Prediction_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        
        linear_in = nn.Linear(state_dimension, hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out_policy = nn.Linear(hidden_layer_dimensions,action_dimension)
        linear_out_value = nn.Linear(hidden_layer_dimensions,state_dimension)
        
        layernom_init = nn.BatchNorm1d(state_dimension)
        layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        dropout = nn.Dropout(0.5)
        activation = nn.ELU()

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

        self.policy = nn.Sequential(*tuple(sequence + [linear_out_policy]))
        self.value = nn.Sequential(*tuple(sequence + [linear_out_value]))

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
        # # # add to sequence|first and recursive|, whatever you need
        linear_in = nn.Linear(state_dimension + action_dimension,hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out_reward = nn.Linear(hidden_layer_dimensions,state_dimension)
        linear_out_state = nn.Linear(hidden_layer_dimensions, state_dimension)
        
        layernom_init = nn.BatchNorm1d(state_dimension + action_dimension)
        layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        dropout = nn.Dropout(0.1)
        
        activation = nn.ELU()   

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

        self.reward = nn.Sequential(*tuple(sequence +[linear_out_reward]))
        self.next_state_normalized = nn.Sequential(*tuple(sequence +[linear_out_state]))

    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return scale_to_bound_action(self.next_state_normalized(x))


class Afterstate_prediction_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        
        linear_in = nn.Linear(state_dimension, hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out_policy = nn.Linear(hidden_layer_dimensions,action_dimension)
        linear_out_value = nn.Linear(hidden_layer_dimensions,state_dimension)
        
        layernom_init = nn.BatchNorm1d(state_dimension)
        layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        dropout = nn.Dropout(0.5)
        activation = nn.ELU()

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

        self.policy = nn.Sequential(*tuple(sequence + [linear_out_policy]))
        self.value = nn.Sequential(*tuple(sequence + [linear_out_value]))

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)



class Dynamics_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension
        # # # add to sequence|first and recursive|, whatever you need
        linear_in = nn.Linear(state_dimension + action_dimension,hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out_reward = nn.Linear(hidden_layer_dimensions,state_dimension)
        linear_out_state = nn.Linear(hidden_layer_dimensions, state_dimension)
        
        layernom_init = nn.BatchNorm1d(state_dimension + action_dimension)
        layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        dropout = nn.Dropout(0.1)
        
        activation = nn.ELU()   

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

        self.reward = nn.Sequential(*tuple(sequence +[linear_out_reward]))
        self.next_state_normalized = nn.Sequential(*tuple(sequence +[linear_out_state]))

    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return self.reward(x), scale_to_bound_action(self.next_state_normalized(x))


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
    
    
    
    
    
    
    
    
    
    
    
    
    

# class Encoder_function(nn.Module):
#     def __init__(self,
#                  observation_space_dimensions,
#                  state_dimension,
#                  action_dimension,
#                  hidden_layer_dimensions,
#                  number_of_hidden_layer):
#         super().__init__()
#         self.action_space = action_dimension
#         # # # add to sequence|first and recursive|,, whatever you need
#         linear_in = nn.Linear(observation_space_dimensions, hidden_layer_dimensions)
#         linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
#         linear_out = nn.Linear(hidden_layer_dimensions, state_dimension)
        
#         self.scale = nn.Tanh()
#         layernom_init = nn.BatchNorm1d(observation_space_dimensions)
#         layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
#         dropout = nn.Dropout(0.1)
#         activation = nn.ReLU()   

#         first_layer_sequence = [
#             linear_in,
#             activation
#         ]

#         recursive_layer_sequence = [
#             linear_mid,
#             activation
#         ]

#         sequence = first_layer_sequence + \
#             (recursive_layer_sequence*number_of_hidden_layer)

#         self.encoder = nn.Sequential(*tuple(sequence+[nn.Linear(hidden_layer_dimensions, action_dimension)]))  
        
#         self.codebook_size = action_dimension
#         #constant codebook of size M, where each entry is a fixed one-hot vector of size M.
#         self.codebook = nn.Parameter(torch.eye(action_dimension),requires_grad=False)

#     def forward(self, o_i):
#         c_e_t = torch.nn.Softmax(-1)(self.encoder(o_i))
#         #Gumbel-Softmax reparameterization trick with 0 temperature
#         # if self.training: 
#         #     c_e_t = c_e_t + (torch.randn_like(c_e_t).log().neg() * 0)
#         c_t = torch.argmax(c_e_t @ self.codebook.T, dim=-1)
#         c_t = one_hot(c_t, self.codebook_size).float()
#         #straight-through estimator is used during the backward to allow the gradients to flow only to the encoder during the backpropagation.
#         c_t = c_t.requires_grad_(False)
#         #no explicit decoder in the model and it does not use a reconstruction loss.
#         return c_t , c_e_t
    
    
    
    
    
    
    
    
    
# # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
# # # To improve the learning process and bound the activations,
# # # we also scale the hidden state to the same range as
# # # the action input
def scale_to_bound_action(x):
    min_next_encoded_state = x.min(1, keepdim=True)[0]
    max_next_encoded_state = x.max(1, keepdim=True)[0]
    scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
    scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
    next_encoded_state_normalized = (
    x - min_next_encoded_state
    ) / scale_next_encoded_state
    return next_encoded_state_normalized 





class Loss_function:
    def __init__(self, parameter = (0), prediction = "no_transform",label = "no_transform"):
        """_
        Loss function and pre-transform.
        
        Example
        -------

        init class: 
        loss = Loss_function(prediction = "no_transform", 
                             label = "no_transform")
                             
        You could use a list of transform to apply such as ["softmax_softmax","clamp_softmax"]
        ps: if you add transform just be carefull to not add transform which break the gradient graph of pytorch
        
        Parameters
        ----------
            Transform
            ---------
                "no_transform" : return the input
                
                "softmax_transform" : softmax the input
                
                "zero_clamp_transform" : to solve log(0) 
                refer to : https://github.com/pytorch/pytorch/blob/949559552004db317bc5ca53d67f2c62a54383f5/aten/src/THNN/generic/BCECriterion.c#L27
                
                "clamp_transform" : bound value betwen 0.01 to 0.99
                
            Loss function
            -------------
                https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
                loss.kldiv
                
                https://en.wikipedia.org/wiki/Cross_entropy
                loss.cross_entropy
                
                https://en.wikipedia.org/wiki/Mean_squared_error
                loss.mse
                
                https://en.wikipedia.org/wiki/Root-mean-square_deviation
                loss.rmse
                
                https://en.wikipedia.org/wiki/Residual_sum_of_squares
                loss.square_error
                
                zero loss (set loss to 0)
            loss.zero_loss
        """
        self.transform = {
                    "no_transform" : lambda x : x ,
                    "softmax_transform" : lambda x : torch.nn.Softmax(dim=1)(x),
                    "zero_clamp_transform" : lambda x : x + 1e-9,
                    "sigmoid_transform": lambda x : torch.nn.Sigmoid()(x),
                    "tanh_transform": lambda x : torch.nn.Tanh()(x),
                    "relu_transform": lambda x : torch.nn.ELU() (x),
                    "shrink_transform": lambda x : torch.nn.Softshrink(lambd=1e-3)(x),

                    }
        if isinstance(prediction,str):
            self.prediction_transform = self.transform[prediction]
        if isinstance(label,str):
            self.label_transform = self.transform[label]
            
        if isinstance(prediction,list):
            self.prediction = prediction
            self.prediction_transform = lambda x : self.multiple_transform(x,"pred")
        if isinstance(label,list):
            self.label = label
            self.label_transform = lambda x : self.multiple_transform(x,"lab")
        self.parameter = parameter
            
    def multiple_transform(self,x,dict_transform):
        if dict_transform == "pred":
            dict_transform = self.prediction
        else:
            dict_transform = self.label
        for i in dict_transform:
            x = self.transform[i](x)
        return x
    
    def kldiv(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        return (p*(torch.log(p)-torch.log(q))).sum(1)
    
    def cross_entropy(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        return (-p*torch.log(q)).sum(1)
    
    
    def square_error(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        return ((p-q)**(1/2)).sum(1)

    def mse(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        return ((p-q)**2).mean(1)

    def rmse(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        return torch.sqrt(((p-q)**2).mean(1))
    
    def zero_loss(self, input, target):
        return(input+target).sum(1)*0
      


# # # L1 Regularization
# # # Explain at : https://paperswithcode.com/method/l1-regularization
def l1(models, l1_weight_decay=0.0001):
    l1_parameters = []
    for parameter_1, parameter_2, parameter_3 in zip(models[0].parameters(), models[1].parameters(), models[2].parameters()):
        l1_parameters.extend(
            (parameter_1.view(-1), parameter_2.view(-1), parameter_3.view(-1)))
    return l1_weight_decay * torch.abs(torch.cat(l1_parameters)).sum()


# # # https://arxiv.org/pdf/1911.08265.pdf [page: 4]
# # # L2 Regularization manually
# # # or can be done using weight_decay from ADAM or SGD
# # # Explain at : https://paperswithcode.com/task/l2-regularization
def l2(models, l2_weight_decay=0.0001):
    l2_parameters = []
    for parameter_1, parameter_2, parameter_3 in zip(models[0].parameters(), models[1].parameters(), models[2].parameters()):
        l2_parameters.extend(
            (parameter_1.view(-1), parameter_2.view(-1), parameter_3.view(-1)))
    return l2_weight_decay * torch.square(torch.cat(l2_parameters)).sum()

def weights_init(m):
    # # # std constant : 
    # # https://en.wikipedia.org/wiki/Fine-structure_constant
    # # https://en.wikipedia.org/wiki/Dimensionless_physical_constant
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)
        torch.nn.init.normal_(m.weight, mean=0.0, std=1/137.035999) 
        torch.nn.init.normal_(m.bias, mean=0.0, std=1/137.035999) 
    if isinstance(m, nn.Conv2d):
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)
        torch.nn.init.normal_(m.weight, mean=0.0, std=1/137.035999) 
        torch.nn.init.normal_(m.bias, mean=0.0, std=1/137.035999) 

        
        



