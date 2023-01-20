
import torch
import torch.nn as nn



# # # # https://www.researchgate.net/figure/A-comparison-between-ResNet-v1-and-ResNet-v2-on-residual-blocks-23_fig2_342334669
# class Residual_block(nn.Module): #V1
#     def __init__(self, num_channels, stride=1):
#         super().__init__()
#         activation= nn.LeakyReLU() #, nn.LeakyReLU(), nn.GELU, nn.ReLU(), nn.ELU
        
#         convolution = torch.nn.Conv2d(num_channels, num_channels, 
#                                            kernel_size=3, stride=stride, 
#                                            padding=1, bias=False)
#         batch_norm = torch.nn.BatchNorm2d(num_channels)
#         self.activation = nn.ReLU()

#         first_layer_sequence = [
#                                 convolution,
#                                 batch_norm,
#                                 self.activation
#                                 ]
        
#         recursive_layer_sequence = [
#                                     convolution,
#                                     batch_norm
#                                     ]
        
#         sequence = first_layer_sequence + (recursive_layer_sequence*1)
        
#         self.sequential_container = nn.Sequential(*tuple(sequence))  # # # combine layers
#         self.last_layer = activation  # # # last layer

#     def forward(self, state):
#         x = self.sequential_container(state)
#         x = x + state
#         return self.last_layer(x)

# # # https://www.researchgate.net/figure/A-comparison-between-ResNet-v1-and-ResNet-v2-on-residual-blocks-23_fig2_342334669
class Residual_block(nn.Module): #V2
    def __init__(self, num_channels, stride=1):
        super().__init__()
        activation= nn.LeakyReLU() #, nn.LeakyReLU(), nn.GELU, nn.ReLU(), nn.ELU
        
        convolution_3 = torch.nn.Conv2d(num_channels, num_channels, 
                                           kernel_size=3, stride=stride, 
                                           padding=1, bias=False)
        convolution_1 = torch.nn.Conv2d(num_channels, num_channels, 
                                           kernel_size=3, stride=stride, 
                                           padding=1, bias=False)
        batch_norm = torch.nn.BatchNorm2d(num_channels)
        activation = nn.ReLU()
        dropout = nn.Dropout2d(p=0.5)

        first_layer_sequence = [
                                batch_norm,
                                activation,
                                dropout,
                                convolution_1
                                ]
        
        recursive_layer_sequence = [
                                    batch_norm,
                                    activation,
                                    dropout,
                                    convolution_3,
                                    batch_norm,
                                    activation,
                                    dropout,
                                    convolution_1
                                    ]
        
        sequence = first_layer_sequence + (recursive_layer_sequence*1)
        
        self.sequential_container = nn.Sequential(*tuple(sequence))  # # # combine layers
        self.last_layer = activation  # # # last layer

    def forward(self, state):
        x = self.sequential_container(state)
        x = x + state
        return x

class Down_sample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        convolution_in = torch.nn.Conv2d(in_channels, out_channels // 2,
                                            kernel_size=3, stride=2,
                                            padding=1, bias=False,)
        convolution_out =  torch.nn.Conv2d(out_channels // 2, out_channels,
                                                kernel_size=3, stride=2,
                                                padding=1, bias=False,) 
        res_blocks_in = Residual_block(out_channels // 2)
        pooling = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        res_blocks_out = Residual_block(out_channels)
        
        sequence = [
                    convolution_in,
                    
                    res_blocks_in,
                    res_blocks_in,
                    
                    convolution_out,
                    
                    res_blocks_out,
                    res_blocks_out,
                    
                    pooling,
                    
                    res_blocks_out,
                    res_blocks_out,
                    res_blocks_out,
                    
                    pooling
                    ]
        
        self.sequential_container = nn.Sequential(*tuple(sequence))  # # # combine layers

    def forward(self, x):
        x = self.sequential_container(x)
        return x


class Representation_function(torch.nn.Module):
    def __init__(self, 
                 observation_space_dimensions, 
                 state_dimension,
                 action_dimension,
                 hidden_layer_dimensions, 
                 number_of_hidden_layer, 
                 num_channels = 3 ,
                 stacked_observations = 1,
                 down_sampling = True ):
        super().__init__()
        self.action_space = action_dimension
        self.down_sampling = down_sampling
        stack_observation = observation_space_dimensions[-1]
        downsample_net = Down_sample( stack_observation, num_channels)
        
        convolution = torch.nn.Conv2d(stack_observation, num_channels, kernel_size=3, stride=1, padding=1, bias=False)#3x3
        batchnorm = torch.nn.BatchNorm2d(num_channels)
        activation = torch.nn.ReLU()
        resblock = Residual_block(num_channels)


        sequence_down_samp = [
                              downsample_net
                             ]  + [resblock] 

        sequence_conv_norm = [
                              convolution,
                              batchnorm,
                              activation,
                             ] + [resblock]
                             
                             
        self.sequential_downsampler = nn.Sequential(*tuple(sequence_down_samp))
        self.sequential_convolution_activation = nn.Sequential(*tuple(sequence_conv_norm))
        
    def forward(self, state):
        state_normalize = self.sequential_downsampler(state) if self.down_sampling else self.sequential_convolution_activation(state)
        return scale_to_bound_action(state_normalize)



class Dynamics_function(torch.nn.Module):
    
    def __init__(self, 
                 state_dimension, 
                 action_dimension,
                 observation_space_dimensions, 
                 hidden_layer_dimensions, 
                 number_of_hidden_layer, 
                 num_channels = 3,
                 reduced_channels_reward = 1,
                 down_sampling=True):
        super().__init__()
        

        block_output_size_reward = ( (reduced_channels_reward * \
                                     int(observation_space_dimensions[0]/14) * \
                                     int(observation_space_dimensions[1]/14) * \
                                     observation_space_dimensions[2])
                                     if down_sampling
                                     else (reduced_channels_reward * \
                                     observation_space_dimensions[0] * \
                                     observation_space_dimensions[1] * \
                                     observation_space_dimensions[2]) 
                                   )
        self.action_space = action_dimension
        dropout1d = nn.Dropout(p=0.5)
        convolution = torch.nn.Conv2d(num_channels + 1, num_channels , kernel_size=3, stride=1, padding=1, bias=False) # 3x3
        batchnorm = torch.nn.BatchNorm2d(num_channels )
        resblock = Residual_block(num_channels )
        convolution_reward = torch.nn.Conv2d(num_channels + 1 , num_channels, 1)#1x1 # reduced_channels_reward for second arg
        
        activation=torch.nn.ReLU()
        sequence_layer_init = [nn.Linear(block_output_size_reward, hidden_layer_dimensions),
                                activation]
        sequence_layer_recursive = [nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions),
                                    activation] * number_of_hidden_layer
        sequence_layer_out = [nn.Linear(hidden_layer_dimensions, state_dimension)]
        multilayer_perceptron_reward = nn.Sequential(*tuple(sequence_layer_init + \
                                                            sequence_layer_recursive + \
                                                                sequence_layer_out) )
        
        flatten = torch.nn.Flatten(1,-1) # or x.view(-1, self.block_output_size_reward)        
        
        sequence = [
                    convolution,
                    batchnorm,
                    activation,
                    ] + ([resblock] * number_of_hidden_layer)+ \
                    [activation]
        
        sequence_reward = [
                           convolution_reward,
                           flatten,
                           multilayer_perceptron_reward
                          ]
        
        self.sequential_container = nn.Sequential(*tuple(sequence))
        self.sequential_reward = nn.Sequential(*tuple(sequence_reward))
        
    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized, action],dim=1)
        next_state_normalize = self.sequential_container(x)
        reward = self.sequential_reward(x)
        return reward , scale_to_bound_action(next_state_normalize)



class Prediction_function(torch.nn.Module):
    def __init__(self, 
                 state_dimension, 
                 action_dimension,
                 observation_space_dimensions, 
                 hidden_layer_dimensions, 
                 number_of_hidden_layer,
                 down_sampling = True,
                 reduced_channels_value = 1, 
                 reduced_channels_policy = 1,
                 num_channels = 3):
        super().__init__()
        
        block_output_size_value = ((reduced_channels_value * observation_space_dimensions[2] * int(observation_space_dimensions[1]/14) * int(observation_space_dimensions[0]/14))
            if down_sampling
            else (reduced_channels_value * observation_space_dimensions[2] * observation_space_dimensions[1] * observation_space_dimensions[0]) )

        block_output_size_policy = ((reduced_channels_policy * observation_space_dimensions[2] * int(observation_space_dimensions[1]/14) * int(observation_space_dimensions[0]/14))
            if down_sampling
            else (reduced_channels_policy * observation_space_dimensions[2] * observation_space_dimensions[1] * observation_space_dimensions[0]))
        
        resblock = Residual_block(num_channels)
        convolution_value = torch.nn.Conv2d(num_channels, num_channels, 1) #1x1 # reduced_channels_value for second arg
        convolution_policy = torch.nn.Conv2d(num_channels, num_channels, 1) #1x1 # reduced_channels_policy for second arg
        
        flatten = torch.nn.Flatten(1,-1)
        activation = torch.nn.ReLU()
        dropout1d = nn.Dropout(p=0.5)
        sequence_layer_init = [nn.Linear(block_output_size_value, hidden_layer_dimensions),
                                activation]
        sequence_layer_recursive = [nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions),
                                    activation] * number_of_hidden_layer
        sequence_layer_out = [nn.Linear(hidden_layer_dimensions, state_dimension)]
        
        multilayer_perceptron_value = nn.Sequential(*tuple(sequence_layer_init + \
                                                           sequence_layer_recursive + \
                                                           sequence_layer_out) )
                        
        # self.fc_policy = mlp(self.block_output_size_policy,fc_policy_layers,action_space_size,)
        sequence_layer_init = [nn.Linear(block_output_size_policy, hidden_layer_dimensions),
                        activation]
        sequence_layer_recursive = [nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions),
                                    activation] * number_of_hidden_layer
        sequence_layer_out = [nn.Linear(hidden_layer_dimensions, action_dimension)]
        
        multilayer_perceptron_policy = nn.Sequential(*tuple(sequence_layer_init + \
                                                            sequence_layer_recursive + \
                                                            sequence_layer_out) )
        
        
        sequence_1 = [ resblock ] * number_of_hidden_layer
        sequence_2 = [convolution_value,
                      flatten,
                      multilayer_perceptron_value]
        sequence_3 = [convolution_policy,
                      flatten,
                      multilayer_perceptron_policy]
        
        self.resnet = nn.Sequential(*tuple(sequence_1))
        self.nn_value = nn.Sequential(*tuple(sequence_2))
        self.nn_policy = nn.Sequential(*tuple(sequence_3))
        
    def forward(self, state_normalize):
        resnet_state_normalize = self.resnet(state_normalize)
        value = self.nn_value(resnet_state_normalize)
        policy = self.nn_policy(resnet_state_normalize)
        return policy , value


class Encoder_function(torch.nn.Module):
    def __init__(self, 
                 state_dimension, 
                 action_dimension,
                 observation_space_dimensions, 
                 hidden_layer_dimensions, 
                 number_of_hidden_layer,
                 down_sampling = True,
                 reduced_channels_value = 1, 
                 reduced_channels_policy = 1,
                 num_channels = 3):
        super().__init__()
        
        block_output_size_value = ((reduced_channels_value * observation_space_dimensions[2] * int(observation_space_dimensions[1]/14) * int(observation_space_dimensions[0]/14))
            if down_sampling
            else (reduced_channels_value * observation_space_dimensions[2] * observation_space_dimensions[1] * observation_space_dimensions[0]) )

        block_output_size_policy = ((reduced_channels_policy * observation_space_dimensions[2] * int(observation_space_dimensions[1]/14) * int(observation_space_dimensions[0]/14))
            if down_sampling
            else (reduced_channels_policy * observation_space_dimensions[2] * observation_space_dimensions[1] * observation_space_dimensions[0]))
        
        resblock = Residual_block(num_channels)
        convolution_policy = torch.nn.Conv2d(num_channels, num_channels, 1) #1x1 # reduced_channels_policy for second arg
        downsample_net = Down_sample( 1, num_channels)
        flatten = torch.nn.Flatten(1,-1)
        activation = torch.nn.ReLU()
        dropout1d = nn.Dropout(p=0.5)

        sequence_layer_init = [nn.Linear(block_output_size_policy, hidden_layer_dimensions),
                        activation]
        sequence_layer_recursive = [nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions),
                                    activation] * number_of_hidden_layer
        sequence_layer_out = [nn.Linear(hidden_layer_dimensions, action_dimension)]
        
        multilayer_perceptron_policy = nn.Sequential(*tuple(sequence_layer_init + \
                                                            sequence_layer_recursive + \
                                                            sequence_layer_out) )
        
        
        sequence_1 = [ downsample_net ,resblock ] * number_of_hidden_layer
        sequence_3 = [convolution_policy,
                      flatten,
                      multilayer_perceptron_policy]
        
        self.encoder = nn.Sequential(*tuple(sequence_1 + sequence_3))
        
    def forward(self, o_i):
        #https://openreview.net/pdf?id=X6D9bAHhBQ1 [page:5 chance outcome]
        c_e_t = torch.nn.Softmax(-1)(self.encoder(o_i))
        c_t = Onehot_argmax.apply(c_e_t)
        return c_t,c_e_t



class Afterstate_dynamics_function(torch.nn.Module):
    
    def __init__(self, 
                 state_dimension, 
                 action_dimension,
                 observation_space_dimensions, 
                 hidden_layer_dimensions, 
                 number_of_hidden_layer, 
                 num_channels = 3,
                 reduced_channels_reward = 1,
                 down_sampling=True):
        super().__init__()
        

        block_output_size_reward = ( (reduced_channels_reward * \
                                     int(observation_space_dimensions[0]/14) * \
                                     int(observation_space_dimensions[1]/14) * \
                                     observation_space_dimensions[2])
                                     if down_sampling
                                     else (reduced_channels_reward * \
                                     observation_space_dimensions[0] * \
                                     observation_space_dimensions[1] * \
                                     observation_space_dimensions[2]) 
                                   )
        self.action_space = action_dimension
        dropout1d = nn.Dropout(p=0.5)
        convolution = torch.nn.Conv2d(num_channels + 1, num_channels , kernel_size=3, stride=1, padding=1, bias=False) # 3x3
        batchnorm = torch.nn.BatchNorm2d(num_channels )
        resblock = Residual_block(num_channels )
        convolution_reward = torch.nn.Conv2d(num_channels + 1 , num_channels, 1)#1x1 # reduced_channels_reward for second arg
        
        activation=torch.nn.ReLU()
        sequence_layer_init = [nn.Linear(block_output_size_reward, hidden_layer_dimensions),
                                activation]
        sequence_layer_recursive = [nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions),
                                    activation] * number_of_hidden_layer
        sequence_layer_out = [nn.Linear(hidden_layer_dimensions, state_dimension)]
        multilayer_perceptron_reward = nn.Sequential(*tuple(sequence_layer_init + \
                                                            sequence_layer_recursive + \
                                                                sequence_layer_out) )
        
        flatten = torch.nn.Flatten(1,-1) # or x.view(-1, self.block_output_size_reward)        
        
        sequence = [
                    convolution,
                    batchnorm,
                    activation,
                    ] + ([resblock] * number_of_hidden_layer)+ \
                    [activation]
        
        sequence_reward = [
                           convolution_reward,
                           flatten,
                           multilayer_perceptron_reward
                          ]
        
        self.sequential_container = nn.Sequential(*tuple(sequence))
        self.sequential_reward = nn.Sequential(*tuple(sequence_reward))
        
    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized, action],dim=1)
        next_state_normalize = self.sequential_container(x)
        reward = self.sequential_reward(x)
        return reward , scale_to_bound_action(next_state_normalize)



class Afterstate_prediction_function(torch.nn.Module):
    def __init__(self, 
                 state_dimension, 
                 action_dimension,
                 observation_space_dimensions, 
                 hidden_layer_dimensions, 
                 number_of_hidden_layer,
                 down_sampling = True,
                 reduced_channels_value = 1, 
                 reduced_channels_policy = 1,
                 num_channels = 3):
        super().__init__()
        
        block_output_size_value = ((reduced_channels_value * observation_space_dimensions[2] * int(observation_space_dimensions[1]/14) * int(observation_space_dimensions[0]/14))
            if down_sampling
            else (reduced_channels_value * observation_space_dimensions[2] * observation_space_dimensions[1] * observation_space_dimensions[0]) )

        block_output_size_policy = ((reduced_channels_policy * observation_space_dimensions[2] * int(observation_space_dimensions[1]/14) * int(observation_space_dimensions[0]/14))
            if down_sampling
            else (reduced_channels_policy * observation_space_dimensions[2] * observation_space_dimensions[1] * observation_space_dimensions[0]))
        
        resblock = Residual_block(num_channels)
        convolution_value = torch.nn.Conv2d(num_channels, num_channels, 1) #1x1 # reduced_channels_value for second arg
        convolution_policy = torch.nn.Conv2d(num_channels, num_channels, 1) #1x1 # reduced_channels_policy for second arg
        
        flatten = torch.nn.Flatten(1,-1)
        activation = torch.nn.ReLU()
        dropout1d = nn.Dropout(p=0.5)
        sequence_layer_init = [nn.Linear(block_output_size_value, hidden_layer_dimensions),
                                activation]
        sequence_layer_recursive = [nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions),
                                    activation] * number_of_hidden_layer
        sequence_layer_out = [nn.Linear(hidden_layer_dimensions, state_dimension)]
        
        multilayer_perceptron_value = nn.Sequential(*tuple(sequence_layer_init + \
                                                           sequence_layer_recursive + \
                                                           sequence_layer_out) )
                        
        # self.fc_policy = mlp(self.block_output_size_policy,fc_policy_layers,action_space_size,)
        sequence_layer_init = [nn.Linear(block_output_size_policy, hidden_layer_dimensions),
                        activation]
        sequence_layer_recursive = [nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions),
                                    activation] * number_of_hidden_layer
        sequence_layer_out = [nn.Linear(hidden_layer_dimensions, action_dimension)]
        
        multilayer_perceptron_policy = nn.Sequential(*tuple(sequence_layer_init + \
                                                            sequence_layer_recursive + \
                                                            sequence_layer_out) )
        
        
        sequence_1 = [ resblock ] * number_of_hidden_layer
        sequence_2 = [convolution_value,
                      flatten,
                      multilayer_perceptron_value]
        sequence_3 = [convolution_policy,
                      flatten,
                      multilayer_perceptron_policy]
        
        self.resnet = nn.Sequential(*tuple(sequence_1))
        self.nn_value = nn.Sequential(*tuple(sequence_2))
        self.nn_policy = nn.Sequential(*tuple(sequence_3))
        
    def forward(self, state_normalize):
        resnet_state_normalize = self.resnet(state_normalize)
        value = self.nn_value(resnet_state_normalize)
        policy = self.nn_policy(resnet_state_normalize)
        return policy , value


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