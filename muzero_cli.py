import sys

from monte_carlo_tree_search import *
from game import *
from replay_buffer import *
from muzero_model import *
from self_play import *

def main(cli_input):
    
################## CLI CHECK COMMAND AND OPEN JSON ##################
    #lower case cli argument
    cli_input_to_lower_case = list(map(lambda x: x.lower(), cli_input))
    # find if one config file has been provide in clie
    config_directory_and_file =  list(filter(lambda s: 'config' in s, cli_input))
    # check if train argument has been provide in the cli command
    config_mode_train =  list(filter(lambda s: 'train' in s, cli_input_to_lower_case))
    # check if play argument has been provide in the cli command
    config_mode_play =  list(filter(lambda s: 'play' in s, cli_input_to_lower_case))
    # check if report argument has been provide in the cli command
    config_mode_report =  list(filter(lambda s: 'report' in s, cli_input_to_lower_case))
    # check if benchmark argument has been provide in the cli command
    config_mode_benchmark =  list(filter(lambda s: 'benchmark' in s, cli_input_to_lower_case))
    # check if benchmark argument has been provide in the cli command
    config_mode_human_buffer =  list(filter(lambda s: 'human_buffer' in s, cli_input_to_lower_case))

    #raise error/explain if config path is not provide
    if len(config_mode_human_buffer) == 0:
        if len(config_directory_and_file) == 0 : 
            raise Exception("Specify a config directory and folder such as: config/config_file.json \
                            Example : \
                                python muzero_cli.py train config/config_file.json \
                                python muzero_cli.py train report config/config_file.json \
                                python muzero_cli.py train report play config/config_file.json \
                                python muzero_cli.py train play config/config_file.json \
                                python muzero_cli.py play config/config_file.json")
            
        # raise error/explain if none of the minimal option has been provide
        if len(config_mode_play + config_mode_train + config_mode_report)  == 0 : 
            raise Exception("Specify a mode such as : train , train report , play , benchmark or any of this combination \
                            Example : \
                                python muzero_cli.py train config/config_file.json \
                                python muzero_cli.py train report config/config_file.json \
                                python muzero_cli.py train report play config/config_file.json \
                                python muzero_cli.py train play config/config_file.json \
                                python muzero_cli.py play config/config_file.json")
    
    # open json config file from provider path. 
    with open(str(config_directory_and_file[0]), 'r') as openfile:
        config = json.load(openfile)
    #json lib already provide error if file not find
    
    ##########################################
    #TYPE USE FOR TRAINING/INFERENCE/BENCHMARK
    compute_type = torch.float32
    #TODO EMBED PYTORCH TYPE WITH STR OPTION
    #########################################

################## buffer builder ##################
    if len(config_mode_human_buffer) > 0:
        human_demonstration_buffer_builder(
         gym_game = config["game"]["env"], 
         render_mode = config["game"]["render"], 
         number_of_bin_action = config["muzero"]["bin_method"], 
         mode_of_bin_action = config["muzero"]["bin_decomposition_number"],
         discount = config["monte_carlo_tree_search"]["discount"],
         limit_of_game_play = config["gameplay"]["limit_of_game_play"],
         rgb_observation = True if "vision" in config["muzero"]["model_structure"]else False,
         keyboard_map_filename = config["human_demonstration_buffer_builder"]["keyboard_map_filename"],
         set_default_noop = config["human_demonstration_buffer_builder"]["set_default_noop"],
         path_to_store_game = config["human_demonstration_buffer_builder"]["path_to_store_game"])

################## TRAIN ##################
    if len(config_mode_train) > 0:
        print("Start the training cycle...")
        # # # set game environment from gym library
        # # # render_mode should be set to None if you don't want rgb observation
        # # # else 'human' or 'rgb_array' depending on  ("human" for atari game)
        env = gym.make(config["game"]["env"],render_mode=config["game"]["render"]) 

        # # # the random seed are set to 0 for reproducibility purpose
        # # # good reference about it at : https://pytorch.org/docs/stable/notes/randomness.html
        np.random.seed(config["random_seed"]["np_random_seed"]) # set the random seed of numpy
        torch.manual_seed(config["random_seed"]["torch_manual_seed"]) # set the random seed of pytorch

        # # # init/set muzero model for training and inference
        muzero = Muzero(model_structure = config["muzero"]["model_structure"], # 'vision_model' : will use rgb as observation , 'mlp_model' : will use game state as observation
                        observation_space_dimensions = env.observation_space, # dimension of the observation 
                        action_space_dimensions = env.action_space, # dimension of the action allow (gym box/discrete)
                        state_space_dimensions = config["muzero"]["state_space_dimensions"], # support size / encoding space
                        hidden_layer_dimensions = config["muzero"]["hidden_layer_dimensions"], # number of weight in the recursive layer of the mlp
                        number_of_hidden_layer = config["muzero"]["number_of_hidden_layer"], # number of recusion layer of hidden layer of the mlp
                        k_hypothetical_steps = config["muzero"]["k_hypothetical_steps"], # number of future step you want to be simulate during train (they are mainly support loss)
                        optimizer = config["muzero"]["optimizer"],
                        lr_scheduler = config["muzero"]["lr_scheduler"],
                        learning_rate = config["muzero"]["learning_rate"], # learning rate of the optimizer
                        loss_type = config["muzero"]["loss_type"],
                        num_of_epoch = config["muzero"]["num_of_epoch"], # number of step during training (the number of step of self play and training can be change)
                        device = config["muzero"]["device"], # device on which you want the comput to be made : "cpu" , "cuda:0" , "cuda:1" , etc
                        type_format = compute_type, # choice the dtype of the model. look at [https://pytorch.org/docs/1.8.1/amp.html#ops-that-can-autocast-to-float16]
                        load = config["muzero"]["load"], # function for loading a save model
                        use_amp = config["muzero"]["use_amp"], # use mix precision for gpu (not implement yet)
                        scaler_on = config["muzero"]["scaler_on"], # scale gradient to reduce computation
                        bin_method = config["muzero"]["bin_method"], # "linear_bin" , "uniform_bin" : will have a regular incrementation of action or uniform sampling(pick randomly) from the bound
                        bin_decomposition_number = config["muzero"]["bin_decomposition_number"],# number of action to sample from low/high bound of a gym discret box
                        priority_scale=config["muzero"]["priority_scale"],
                        rescale_value_loss = config["muzero"]["rescale_value_loss"]) 

        if config["human_demonstration_buffer_builder"]["path_to_store_game"] is not None:
            human_buffer = DemonstrationBuffer()
            human_buffer.load_back_up_buffer(config["human_demonstration_buffer_builder"]["path_to_store_game"])
        else:
            human_buffer = DemonstrationBuffer()
            
        # # # init/set the game storage(stor each game) and dataset(create dataset) generate during training
        replay_buffer = ReplayBuffer(window_size = config["replaybuffer"]["window_size"], # number of game store in the buffer
                                     batch_size = config["replaybuffer"]["batch_size"], # batch size is the number of observe game during train
                                     num_unroll = muzero.k_hypothetical_steps, # number of mouve/play store inside the batched game
                                     td_steps = config["replaybuffer"]["td_steps"], # number of step the value is scale on 
                                     game_sampling = config["replaybuffer"]["game_sampling"], # 'uniform' or "priority" (will game randomly or with a priority distribution)
                                     position_sampling = config["replaybuffer"]["position_sampling"],
                                     reanalyze_stack = [ReanalyseBuffer(),
                                                        human_buffer,
                                                        MostRecentBuffer(max_buffer_size = 20),
                                                        HighestRewardBuffer()],
                                     reanalyse_fraction=config["replaybuffer"]["reanalyse_fraction"], # porcentage/100 of reanalyze vs new_game
                                     reanalyse_fraction_mode = config["replaybuffer"]["reanalyse_fraction_mode"] # "chance" or "ratio"
                                     ) # 'uniform' or "priority" (will sample position in game randomly or with a priority distribution)

        # # # init/set the monte carlos tree search parameter
        mcts = Monte_carlo_tree_search(pb_c_base = config["monte_carlo_tree_search"]["pb_c_base"] , 
                                       pb_c_init = config["monte_carlo_tree_search"]["pb_c_init"], 
                                       discount = config["monte_carlo_tree_search"]["discount"], 
                                       root_dirichlet_alpha = config["monte_carlo_tree_search"]["root_dirichlet_alpha"], 
                                       root_exploration_fraction = config["monte_carlo_tree_search"]["root_exploration_fraction"],
                                       num_simulations = config["monte_carlo_tree_search"]["maxium_action_sample"],# number of node per level ( width of the tree )
                                       maxium_action_sample = config["monte_carlo_tree_search"]["maxium_action_sample"],# number of node per level ( width of the tree )
                                       number_of_player = config["monte_carlo_tree_search"]["number_of_player"], 
                                       custom_loop = config["monte_carlo_tree_search"]["custom_loop"])

        # # # ini/set the Game class which embbed the gym game class function
        gameplay = Game(gym_env = env, 
                        discount = config["monte_carlo_tree_search"]["discount"], #should be the same discount than mcts
                        limit_of_game_play = config["gameplay"]["limit_of_game_play"], # maximum number of mouve
                        observation_dimension = muzero.observation_dimension, 
                        action_dimension = muzero.action_dimension,
                        rgb_observation = muzero.is_RGB,
                        action_map = muzero.action_dictionnary,
                        priority_scale=muzero.priority_scale)
        
        # # # train model (if you choice vison model it will render the game)
        epoch_pr , loss , reward = learning_cycle(number_of_iteration = config["learning_cycle"]["number_of_iteration"], # number of epoch(step) in  muzero should be the |total amount of number_of_iteration x number_of_training_before_self_play|
                                                  number_of_self_play_before_training = config["learning_cycle"]["number_of_self_play_before_training"], # number of game played record in the replay buffer before training
                                                  number_of_training_before_self_play = config["learning_cycle"]["number_of_training_before_self_play"], # number of epoch made by the model before selplay
                                                  model_tag_number = config["learning_cycle"]["model_tag_number"], # tag number use to generate checkpoint
                                                  number_of_worker_selfplay = config["learning_cycle"]["number_of_worker_selfplay"],
                                                  temperature_type = config["learning_cycle"]["temperature_type"], # "static_temperature" ,"linear_decrease_temperature" ,  "extreme_temperature" and "reversal_tanh_temperature"
                                                  verbose = config["learning_cycle"]["verbose"], # if you want to print the epoch|reward|loss during train
                                                  muzero_model = muzero,
                                                  gameplay = gameplay,
                                                  monte_carlo_tree_search = mcts,
                                                  replay_buffer = replay_buffer)
        print("Training end.")
        
        
################## REPORT ##################
    if len(config_mode_train) > 0 and len(config_mode_report) > 0:
        print("Creating report...")
        report( muzero, replay_buffer, epoch_pr, loss, reward)
        print("Report created")

################## INFERENCE_FROM_CHECKPOINT ##################
    if len(config_mode_play) > 0:
        print("Start play...")
        play_game_from_checkpoint(game_to_play = config["game"]["env"],
                                    
                                  model_tag = config["play_game_from_checkpoint"]["model_tag"],
                                  model_device = config["play_game_from_checkpoint"]["model_device"],
                                  model_type = torch.float32,
                                    
                                  mcts_pb_c_base = config["monte_carlo_tree_search"]["pb_c_base"] , 
                                  mcts_pb_c_init = config["monte_carlo_tree_search"]["pb_c_init"], 
                                  mcts_discount = config["monte_carlo_tree_search"]["discount"], 
                                  mcts_root_dirichlet_alpha = config["monte_carlo_tree_search"]["root_dirichlet_alpha"], 
                                  mcts_root_exploration_fraction = config["monte_carlo_tree_search"]["root_exploration_fraction"],
                                  mcts_with_or_without_dirichlet_noise = config["play_game_from_checkpoint"]["mcts_with_or_without_dirichlet_noise"],
                                  number_of_monte_carlo_tree_search_simulation = config["monte_carlo_tree_search"]["maxium_action_sample"],
                                  maxium_action_sample = config["monte_carlo_tree_search"]["maxium_action_sample"],# number of node per level ( width of the tree )
                                  number_of_player = config["monte_carlo_tree_search"]["number_of_player"], 
                                  custom_loop = config["monte_carlo_tree_search"]["custom_loop"],
                                    
                                  temperature = config["play_game_from_checkpoint"]["temperature"],
                                  game_iter = config["play_game_from_checkpoint"]["game_iter"],
                                    
                                  slow_mo_in_second = config["play_game_from_checkpoint"]["slow_mo_in_second"],
                                  render = config["play_game_from_checkpoint"]["render"],
                                  verbose = config["play_game_from_checkpoint"]["verbose"])
        print("End play")
        
    
################## BENCHMARK_FROM_CHECKPOINT ##################
    if len(config_mode_benchmark) > 0:
        print("Start benchmark...")
        number_of_trial = 100
        cache_t,cache_r,cache_a,cache_p = [],[],[],[]
        for _ in range(number_of_trial):
            tag , reward , action, policy = play_game_from_checkpoint(game_to_play = config["game"]["env"],
            
                                  model_tag = config["play_game_from_checkpoint"]["model_tag"],
                                  model_device = config["play_game_from_checkpoint"]["model_device"],
                                  model_type = torch.float32,
                
                                  mcts_pb_c_base = config["monte_carlo_tree_search"]["pb_c_base"] , 
                                  mcts_pb_c_init = config["monte_carlo_tree_search"]["pb_c_init"], 
                                  mcts_discount = config["monte_carlo_tree_search"]["discount"], 
                                  mcts_root_dirichlet_alpha = config["monte_carlo_tree_search"]["root_dirichlet_alpha"], 
                                  mcts_root_exploration_fraction = config["monte_carlo_tree_search"]["root_exploration_fraction"],
                                  mcts_with_or_without_dirichlet_noise = config["play_game_from_checkpoint"]["mcts_with_or_without_dirichlet_noise"],
                                  number_of_monte_carlo_tree_search_simulation = config["monte_carlo_tree_search"]["maxium_action_sample"],
                                  maxium_action_sample = config["monte_carlo_tree_search"]["maxium_action_sample"],# number of node per level ( width of the tree )
                                  number_of_player = config["monte_carlo_tree_search"]["number_of_player"], 
                                  custom_loop = config["monte_carlo_tree_search"]["custom_loop"],
                                                                      
                                  temperature = config["play_game_from_checkpoint"]["temperature"],
                                  game_iter = config["play_game_from_checkpoint"]["game_iter"],
                                    
                                  slow_mo_in_second = 0,
                                  render = False,
                                  verbose = False,
                                  benchmark = True) # Need benchmark True to return output
            #could do it in one list or even wrap the play_game with benchmark but it reduce clarity
            cache_t.append(tag)
            cache_r.append(reward)
            cache_a.append(action)
            cache_p.append(policy)


        benchmark(cache_t,
                cache_r,
                cache_a,
                cache_p,
                folder = "report",
                verbose = True)
        print("End benchmark")


if __name__ == "__main__":
    main(sys.argv[:])
