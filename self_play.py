import copy
import gc
import random
import time
import pickle

import matplotlib.pyplot as plt
import ray

from game import *
from monte_carlo_tree_search import *
from muzero_model import *
from replay_buffer import *

import gym


##########################################################################################################################

# Create two function because the compute performance in sequential mode using
# local_mode of ray.remote are significantly slower.
@ray.remote
def play_game_ray(environment=None,
                  model=None,
                  monte_carlo_tree_search=None,
                  temperature=1,
                  replay_buffer = None):

    
    environment = copy.deepcopy(environment)
    should_reanalyze = replay_buffer.should_reanalyse()
    if should_reanalyze:
        reanalyze_observation = replay_buffer.reanalyse_buffer_sample_game()
    else:
        if environment.env.metadata['render_fps'] is None:
            environment.env.metadata['render_fps'] = 30

    counter = 0
    observation_reward_done_info = None
    while not environment.terminal and counter < environment.limit_of_game_play:
        
        feedback = reanalyze_observation if should_reanalyze else observation_reward_done_info

        state = environment.observation(iteration=counter,
                                        feedback=feedback)
        
        tree = monte_carlo_tree_search.run(observation=state,
                                            model=model,
                                            train=True)
        observation_reward_done_info = environment.policy_step(root = tree,
                                                               temperature = temperature,
                                                               feedback = feedback,
                                                               iteration = counter)


        environment.store_search_statistics(tree)
        counter += 1
    monte_carlo_tree_search.cycle.global_reset()
    environment.close()
    return environment


def play_game(environment=None,
                  model=None,
                  monte_carlo_tree_search=None,
                  temperature=1,
                  replay_buffer = None):

    environment = copy.deepcopy(environment)
    should_reanalyze = replay_buffer.should_reanalyse()
    if should_reanalyze:
        reanalyze_observation = replay_buffer.reanalyse_buffer_sample_game()
    else:
        if environment.env.metadata['render_fps'] is None:
            environment.env.metadata['render_fps'] = 30

    counter = 0
    observation_reward_done_info = None
    while not environment.terminal and counter < environment.limit_of_game_play:
        
        feedback = reanalyze_observation if should_reanalyze else observation_reward_done_info
        state = environment.observation(iteration=counter,
                                        feedback=feedback)
        
        tree = monte_carlo_tree_search.run(observation=state,
                                            model=model,
                                            train=True)
        observation_reward_done_info = environment.policy_step(root = tree,
                                                               temperature = temperature,
                                                               feedback = feedback,
                                                               iteration = counter)

        monte_carlo_tree_search.cycle.global_reset()
        environment.store_search_statistics(tree)
        counter += 1
        
    monte_carlo_tree_search.cycle.global_reset()
    environment.close()
    return environment

##########################################################################################################################


def scaler(x, newmin=0, newmax=1):
    # bound a serie between new value
    oldmin, oldmax = min(x), max(x)
    oldrange = oldmax - oldmin
    newrange = newmax - newmin
    if oldrange == 0:        # Deal with the case where rvalue is constant:
        if oldmin < newmin:      # If rvalue < newmin, set all rvalue values to newmin
            newval = newmin
        elif oldmin > newmax:    # If rvalue > newmax, set all rvalue values to newmax
            newval = newmax
        else:                    # If newmin <= rvalue <= newmax, keep rvalue the same
            newval = oldmin
        normal = [newval for _ in x]
    else:
        scale = newrange / oldrange
        normal = [(v - oldmin) * scale + newmin for v in x]
    return np.array(normal)

##########################################################################################################################


def temperature_scheduler(epoch=1, actual_epoch=1, mode = "static_temperature"):
    # # # personal add
    # # # will scale the remperature to an opposite tanh distribution ( 1 - tanh )
    # # # of chosen bound ( look like cosineannealing for reference)
    if mode == "reversal_tanh_temperature":
        array = np.array(list(range(1,epoch+1)))
        index = np.where(array == actual_epoch)
        range_scale_array = np.tanh(scaler(array,newmin=0.001,newmax=0.75))[index]
        return (1 - range_scale_array) * 1.1
    
    if mode == "extreme_temperature":
        if actual_epoch < epoch * (100/700): return 3
        elif actual_epoch < epoch * (200/700) : return 2
        elif actual_epoch < epoch * (300/700) : return 1
        elif actual_epoch < epoch * (400/700) : return .7
        elif actual_epoch < epoch * (500/700) : return .5
        elif actual_epoch < epoch * (600/700) : return .4
        elif actual_epoch < epoch * 1: return .0625
        
    # # # https://arxiv.org/pdf/1911.08265.pdf [page: 13]
    # # # original temperature distrubtion of muzero
    # # # Temperature is find for choicing an action such as:
    # # # policy**1/T/sum(policy**1/T)
    # # # using the policy output by the mcts
    # # # | under 50%  T=1 | under 75% T=0.5 | over 75% T=0.25
    if mode == "linear_decrease_temperature":
        if epoch * 0.5 > actual_epoch:
            return 1
        elif epoch * 0.75 > actual_epoch:
            return 0.5
        else:
            return 0.2

    if mode == "static_temperature":
        return 0.0
    
    if mode == "static_one_temperature":
        return 1

##########################################################################################################################


def learning_cycle(number_of_iteration=10000,
                   number_of_self_play_before_training=1,
                   number_of_training_before_self_play=1,
                   model_tag_number=124,
                   number_of_worker_selfplay=1,
                   temperature_type = "static_temperature",
                   verbose=True,
                   muzero_model=None,
                   gameplay=None,
                   monte_carlo_tree_search=None,
                   replay_buffer=None):
    """
        Start learning cycle using Muzero, MCTS, Gameplay and Replay buffer
        
        Parameters
        ----------
            number_of_iteration (int): 
            Number of loop of self-play and training to run
            Defaults to 10000.
            
            number_of_self_play_before_training (int): 
            Number of self-play to run per loop.
            Defaults to 1.

            number_of_training_before_self_play (int): 
            Number of training to run per loop.
            Defaults to 1.


            model_tag_number (int): 
            The tag number of the model
            Defaults to 124.

            number_of_worker_selfplay (int): 
            How many self-play should be run in parallele
            Defaults to 1.

            temperature_type (str): 
            choice between "static_temperature" ,"linear_decrease_temperature" ,  "extreme_temperature" and "reversal_tanh_temperature"
            "static_temperature" : will always choice the argmax of the predicted policy
            "linear_decrease_temperature" : Training [0% -> 50, 50% -> 75%, 75% -> 100%] map to temperature [1,0.5,0.25]
            "extreme_temperature" : Training [0% -> 14.2%, 14.2% -> 28.4%, 28.4% -> 42.8%, etc..] map to temperature [3,2,1,0.7,0.5,0.4,0.0625]
            "reversal_tanh_temperature" : smooth temperature between 1 to 0 following cos annealing like.
            Defaults to "static_temperature".

            verbose (bool): 
            show the print of the iteration number, reward and loss during trainong
            Defaults to True.

            muzero_model : (muzero.class).
            
            gameplay : (gameplay.class)
            
            monte_carlo_tree_search : (mcts.class)
            
            replay_buffer : (replay_buffer.class)
    """

    assert isinstance(number_of_iteration,int) and number_of_iteration >= 1 , "number_of_iterationt ∈ int | {1 < number_of_iteration < +inf)"
    assert isinstance(number_of_self_play_before_training,int) and number_of_self_play_before_training >= 0, "number_of_self_play_before_training ∈ int | {0 < number_of_self_play_before_training < +inf)"
    assert isinstance(number_of_training_before_self_play,int) and number_of_training_before_self_play >= 0, "number_of_training_before_self_play ∈ int | {0 < number_of_training_before_self_play < +inf)"
    assert isinstance(model_tag_number,int) and model_tag_number >= 0, "model_tag_number ∈ int | {0 < model_tag_number < +inf)"
    assert isinstance(number_of_worker_selfplay,int) and number_of_worker_selfplay >= 0, "number_of_worker_selfplay ∈ float | {0 < discount < +inf)"
    assert isinstance(temperature_type,str) and temperature_type in ["reversal_tanh_temperature","extreme_temperature","linear_decrease_temperature","static_temperature","static_one_temperature"], "temperature_type ∈ {reversal_tanh_temperature,extreme_temperature,linear_decrease_temperature,static_temperature,static_one_temperature} ⊆ str "
    assert isinstance(verbose,bool) , "verbose ∈ bool"

    # try:
    # # # Training
    reward, cache_reward, epoch_pr, loss, cache_loss = [-float("inf")], [], [], [], []
    if number_of_worker_selfplay in ["max", "all"] or number_of_worker_selfplay >= int(torch.multiprocessing.cpu_count()):
        number_of_worker_selfplay = int(torch.multiprocessing.cpu_count())

    if number_of_worker_selfplay >= 2:
        ray.init(num_cpus=number_of_worker_selfplay,
                    num_gpus=torch.cuda.device_count(),
                    include_dashboard=False)

    for ep in range(1, number_of_iteration+1):
        # # # reset the cache reward for every iteration
        cache_reward, cache_loss = [], []
        if number_of_worker_selfplay >= 2 :
            game = ray.get([
                    play_game_ray.remote(
                    environment=gameplay,
                    model=muzero_model,
                    monte_carlo_tree_search=monte_carlo_tree_search,
                    temperature=temperature_scheduler(number_of_iteration+1, ep, mode = temperature_type),
                    replay_buffer = replay_buffer)
                for _ in range(number_of_self_play_before_training)])
        
        else:
            game = [play_game(
                environment=gameplay,
                model=muzero_model,
                monte_carlo_tree_search=monte_carlo_tree_search,
                temperature=temperature_scheduler(number_of_iteration+1, ep, mode = temperature_type),
                replay_buffer = replay_buffer)
            for _ in range(number_of_self_play_before_training)]
                                        
        for g in game:
            replay_buffer.save_game(g),             
            cache_reward.append(sum(g.rewards))
        # # # sum the average reward of all self_play
        reward.append(sum(cache_reward)/len(cache_reward))

        # # # save best model. self_play serve as dataset and performace test
        did_better = None if reward[-1] == max(reward) and not all(g.reanalyzed for g in game ) else "do not save"
        if did_better is None: 
            print(" "*1000,end='\r')
            print("save model with : ", reward[-1]," reward")
        muzero_model.save_model(
            directory="model_checkpoint", 
            tag=model_tag_number, 
            model_update_or_backtrack = did_better )
            
            
        # # # train model from all game accumulate in the replay_buffer
        for _ in range(number_of_training_before_self_play):
            new_priority , batch_game_position = muzero_model.train(replay_buffer.sample_batch())
            replay_buffer.update_value(new_priority , batch_game_position)
            cache_loss.append(muzero_model.store_loss[-1][0])
            
        loss.append(sum(cache_loss)/len(cache_loss))

        prompt_feedback = f'EPOCH {ep} || selfplay reward: {reward[-1]} || training loss: { loss[-1] }||'
        epoch_pr.append(prompt_feedback)
        if verbose: 
            print(" "*1000,end='\r')
            print(prompt_feedback,end='\r')
    
    configuration = {'number_of_iteration' : number_of_iteration,
                     'number_of_self_play_before_training' : number_of_self_play_before_training,
                     'number_of_training_before_self_play' : number_of_training_before_self_play,
                     'model_tag_number' : model_tag_number,
                     'number_of_worker_selfplay' : number_of_worker_selfplay,
                     'temperature_type' : temperature_type,
                     "verbose" : verbose}

    return epoch_pr, loss, reward, configuration

##########################################################################################################################

def play_game_from_checkpoint(game_to_play='CartPole-v1',
                              model_tag=124,
                              model_device="cuda:0",
                              model_type=torch.float32,
                              mcts_pb_c_base=19652,
                              mcts_pb_c_init=1.25,
                              mcts_discount=0.997,
                              mcts_root_dirichlet_alpha=0.25,
                              mcts_root_exploration_fraction=0.25,
                              mcts_with_or_without_dirichlet_noise=True,
                              number_of_monte_carlo_tree_search_simulation=11,
                              maxium_action_sample = 2,# number of node per level ( width of the tree )
                              number_of_player = 1, 
                              custom_loop = None,
                              temperature=0,
                              game_iter=2000,
                              slow_mo_in_second=0.0,
                              render=True,
                              verbose=True,
                              benchmark=False):
    """
    Env/Game inference
    
    Parameters
    ----------
        game_to_play (str): Defaults to 'CartPole-v1'.
        model_tag (int): Defaults to 124.
        model_device (str): Defaults to "cuda:0".
        model_type (torch.type): Defaults to torch.float32.
        mcts_pb_c_base (int): Defaults to 19652.
        mcts_pb_c_init (float): Defaults to 1.25.
        mcts_discount (float): Defaults to 0.95.
        mcts_root_dirichlet_alpha (float): Defaults to 0.25.
        mcts_root_exploration_fraction (float: Defaults to 0.25.
        mcts_with_or_without_dirichlet_noise (bool): Defaults to True.
        number_of_monte_carlo_tree_search_simulation (int): Defaults to 11.
        temperature (int): Defaults to 0.
        game_iter (int): Defaults to 2000.
        slow_mo_in_second (float): Defaults to 0.0.
        render (bool): Defaults to True.
        verbose (bool): Defaults to True.
        benchmark (bool: Defaults to False.

    """    

    import random
    import time

    import gym

    from game import Game
    from monte_carlo_tree_search import (MinMaxStats, Monte_carlo_tree_search,
                                         Node)
    from muzero_model import Gym_space_transform, Muzero

    # play with model of choice (will repeat variable for explanatory purpose)
    # # # choice game env
    if render:
        try: env = gym.make(game_to_play, render_mode = 'human')
        except: env = gym.make(game_to_play, render_mode = "rgb_array")
    else: env = gym.make(game_to_play)
    
    # # # initialize model class without initializing a neural network
    muzero = Muzero(load=True,
                    type_format=model_type)
    
    # # # load save model with tag number
    muzero.load_model(tag=model_tag,
                      observation_space_dimensions=env.observation_space,
                      device=model_device)  # set device for model compute
    
    # # # init the mcts class
    monte_carlo_tree_search = Monte_carlo_tree_search(pb_c_base=mcts_pb_c_base,
                                                      pb_c_init=mcts_pb_c_init,
                                                      discount=mcts_discount,
                                                      root_dirichlet_alpha=mcts_root_dirichlet_alpha,
                                                      root_exploration_fraction=mcts_root_exploration_fraction,
                                                      num_simulations = number_of_monte_carlo_tree_search_simulation,# number of level (length of the tree ) 
                                                      maxium_action_sample = maxium_action_sample,# number of node per level ( width of the tree )
                                                      number_of_player = number_of_player, 
                                                      custom_loop = custom_loop)
    # # # create the game class with gameplay/record function
    gameplay = Game(env,
                    discount=monte_carlo_tree_search.discount,
                    observation_dimension=muzero.observation_dimension,
                    action_dimension=muzero.action_dimension,
                    rgb_observation=muzero.is_RGB,
                    action_map=muzero.action_dictionnary,
                    priority_scale=muzero.priority_scale)
    # # # slow animation of the render ( in second )
    sleep = slow_mo_in_second
    # # # temperature set to 0 will use argmax as policy (highest probability action)
    # # # over a temperature of 0.0035 it will sample with the propability associate to the mouve , picking uniformly
    # # # number of iteration (mouve play during the game)
    game_iter = game_iter

    observation_reward_done_info = None
    reward_ls, action_ls, policy_ls = [], [], []
    # # # or while not environment.terminal: # for loop to bypass env terminal limit, else use while loop and add a counter variable incrementing
    for counter in range(game_iter):
        # # #laps time to see a slow motion of the env
        time.sleep(sleep)
        # # # start the game and get game initial observation / game return observation after action
        state = gameplay.observation(iteration=counter,
                                     feedback=observation_reward_done_info)
        
        # render the env
        if render:
            gameplay.vision()
        # # # run monte carlos tree search inference
        # # Train [False or True] mean with or without dirichlet at the root
        mcts = monte_carlo_tree_search
        tree= mcts.run(observation=state,
                       model=muzero,
                       train=mcts_with_or_without_dirichlet_noise)

        # # # select the best action from policy and inject the action into the game (.step())
        observation_reward_done_info = gameplay.policy_step(root = tree,
                                                            temperature = temperature,
                                                            feedback = None,
                                                            iteration = counter)


        # # # print the number of mouve, action and policy
        
        action, policy, _ = gameplay.policy_action_reward_from_tree(tree)
        if verbose:
            print(
                f"Mouve number: {counter+1} ,\ Action: {muzero.action_dictionnary[action[np.argmax(policy/policy.sum())]]}, Policy: {policy/policy.sum()}")

        # that is ugly need to fix it
        if benchmark:
            reward_ls.append(sum(gameplay.rewards))
            action_ls.append(
                muzero.action_dictionnary[action[np.argmax(policy/policy.sum())]])
            policy_ls.append(policy/policy.sum())
        if gameplay.terminal or game_iter == counter:
            break
    mcts.cycle.global_reset()
    gameplay.close()
    if benchmark:
        return muzero.random_tag, reward_ls, action_ls, policy_ls


##########################################################################################################################


def benchmark(model_tag, reward, action, policy, folder="report", verbose=False):
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, hspace=None)
    axs = gs.subplots(sharex=True, sharey=False)

    trial = [f"Trial {i}" for i in range(len(reward))]
    rewa = [i[-1] for i in reward]
    axs[0].bar(trial, rewa)
    axs[0].set_ylabel('Accumulated Reward')
    axs[0].set_title(f'Model: {model_tag[0]} | Reward benchmark |')

    trial = [f"Trial {i}" for i in range(len(reward))]
    rewa = [len(i) for i in reward]
    axs[1].bar(trial, rewa)
    axs[1].set_ylabel('N mouve')
    axs[1].set_title(f'Model: {model_tag[0]} | Mouve benchmark |')
    plt.savefig(f'{folder}/model_{model_tag[0]}_reward_benchmark.png')
    if verbose:
        plt.figure()

    with open(f'{folder}/model_{model_tag[0]}_action_and_policy_benchmark.txt', "a+") as f:
        for trial, (tag, act, poli) in enumerate(zip(model_tag, action, policy)):
            print(f"| Model Tag: {tag} | Trial number: {trial} |", file=f)
            for a, b, c in zip(act, poli, range(len(act))):
                print(f"|Action: {a} |Policy: {b} | Mouve number: {c} |", file=f)

##########################################################################################################################


def report(muzero, replay_buffer, epoch_pr, loss, reward, folder="report", verbose=False):
    # TODO: build interactive html report with plotly
    if not os.path.exists(folder):
        os.makedirs(folder)

    t = time.localtime()
    q = muzero.random_tag
    print(f"creating report at : | directory: {folder}/ | model tag: {q} |")

    with open(f'{folder}/model_{q}_data_of_parameter_weight_and_epoch.txt', "a+") as f:
        
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||| Preview MODEL WEIGHT OF <representation_function> ||||||||||||||", file=f)
        for i in muzero.representation_function.parameters():
            print(i, i.size(), file=f)
        print("|||||||||||||| END MODEL WEIGHT OF <representation_function> ||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||| Preview MODEL WEIGHT OF <dynamics_function> ||||||||||||||||||||", file=f)
        for i in muzero.dynamics_function.parameters():
            print(i, i.size(), file=f)
        print("|||||||||||||| END MODEL WEIGHT OF <dynamics_function> ||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||| Preview MODEL WEIGHT OF <prediction_function> ||||||||||||||||||", file=f)
        for i in muzero.prediction_function.parameters():
            print(i, i.size(), file=f)
        print("|||||||||||||| END MODEL WEIGHT OF <prediction_function> ||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", file=f)
        print("|||||||||||||||||||||||||||Epoch History||||||||||||||||||||||||||", file=f)
        if len(epoch_pr) > 0 :
            for i in epoch_pr:
                print(i, file=f)

    from matplotlib.ticker import NullFormatter, StrMethodFormatter

    fig, ax = plt.subplots()
    plt.plot(reward)
    plt.title("Average Reward")
    plt.xlabel('Number of iteration a.k.a. batch of step')
    plt.ylabel('Avg. Reward')
    plt.savefig(f'{folder}/model_{q}_data_of_the_average_reward.png')
    if verbose:
        plt.figure()

    fig, ax = plt.subplots()
    plt.plot(loss)
    plt.title("Average Loss")
    plt.xlabel('Number of iteration a.k.a. batch of step')
    plt.ylabel('Avg. Loss')
    plt.savefig(f'{folder}/model_{q}_data_of_the_average_loss.png')
    if verbose:
        plt.figure()

    all_loss = np.array([[a.cpu().detach().numpy() for a in x[:]]
                        for x in muzero.store_loss], dtype=np.float64)
    fig, ax = plt.subplots()
    plt.plot(all_loss)
    plt.yscale('log')
    plt.title("Complet Loss Stack")
    plt.xlabel('Step a.k.a. epoch')
    plt.ylabel('Loss')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    plt.savefig(f'{folder}/model_{q}_data_of_all_the loss.png')
    if verbose:
        plt.figure()

##########################################################################################################################


def generate_config_file(env = None,
                         seed = None,
                         muzero = None,
                         replay_buffer = None,
                         mcts = None,
                         gameplay = None,
                         learning_configuration = None,
                         save_codebase = True):
    
    import json
    import zipfile
    import os
    
    list_holder = []
    if env != None and muzero != None:
        try: buffer_path = replay_buffer.reanalyze_stack[1].load_path
        except: buffer_path = None
        dict_env = {"human_demonstration_buffer_builder" : 
                            {"keyboard_map_filename" : None,
                            "set_default_noop" : None,
                            "path_to_store_game" : buffer_path if buffer_path is not None else f"config/{learning_configuration['model_tag_number']}_hbuffer.pickle"}
                    }
        list_holder.append(dict_env)
    
    if env != None:
        try: rendermode = env.spec.kwargs['render_mode'] if env.spec.kwargs['render_mode'] != None else None
        except: rendermode = None
        dict_env = {"game" : {"env" : env.spec.id,
                              "render" : rendermode}
                    }
        list_holder.append(dict_env)

    if seed != None:
        dict_seed = {"random_seed" : {"np_random_seed" : seed , 
                                      "torch_manual_seed" : seed ,
                                      "env_seed" : seed}
                     }
        list_holder.append(dict_seed)

    if muzero != None:
        dict_model = {"muzero" : {"model_structure" : muzero.model_structure ,
                                  "state_space_dimensions" : muzero.state_dimension,
                                  "hidden_layer_dimensions" : muzero.hidden_layer_dimension,
                                  "number_of_hidden_layer" : muzero.number_of_hidden_layer,
                                  "k_hypothetical_steps": muzero.k_hypothetical_steps,
                                  "optimizer" : muzero.opt,
                                  "lr_scheduler" : muzero.sch,
                                  "learning_rate": muzero.lr,
                                  "loss_type": muzero.loss_type,
                                  "num_of_epoch" : muzero.epoch,
                                  "device" : muzero.device,
                                  "load" : False,
                                  "use_amp" : muzero.use_amp,
                                  "scaler_on": False,
                                  "bin_method" : muzero.bin_method,
                                  "bin_decomposition_number" : muzero.bin_decomposition_number,
                                  "priority_scale" : muzero.priority_scale,
                                  "rescale_value_loss" : muzero.rescale_value_loss }
                      }
        list_holder.append(dict_model)

    if replay_buffer != None:
        dict_buffer = {"replaybuffer" : {"window_size" : replay_buffer.window_size,
                                         "batch_size" : replay_buffer.batch_size,
                                         "td_steps" : replay_buffer.td_steps,
                                         "game_sampling" : replay_buffer.game_sampling,
                                         "position_sampling" : replay_buffer.position_sampling,
                                         "reanalyse_fraction" : replay_buffer.reanalyse_fraction,
                                         "reanalyse_fraction_mode" : replay_buffer.reanalyse_fraction_mode
                                         }}
        list_holder.append(dict_buffer)

    if mcts != None:
        dict_mcts = {"monte_carlo_tree_search" : {"pb_c_base" : mcts.pb_c_base , 
                                                  "pb_c_init" : mcts.pb_c_init, 
                                                  "discount" : mcts.discount, 
                                                  "root_dirichlet_alpha" : mcts.root_dirichlet_alpha, 
                                                  "root_exploration_fraction" : mcts.root_exploration_fraction,
                                                  "num_simulations" : mcts.num_simulations,
                                                  "maxium_action_sample" : mcts.maxium_action_sample,
                                                  "number_of_player" : mcts.number_of_player, 
                                                  "custom_loop" : mcts.custom_loop}
                    }
        list_holder.append(dict_mcts)
    
    if gameplay != None:
        dict_gameplay = {"gameplay" : {"limit_of_game_play" : gameplay.limit_of_game_play}}
        list_holder.append(dict_gameplay)

    if learning_configuration != None:
        dict_lc = {"learning_cycle" : {"number_of_iteration" : learning_configuration['number_of_iteration'],
                                       "number_of_self_play_before_training" : learning_configuration['number_of_self_play_before_training'],
                                       "number_of_training_before_self_play" : learning_configuration['number_of_training_before_self_play'],
                                       "temperature_type" : learning_configuration['temperature_type'], 
                                       "model_tag_number" : learning_configuration['model_tag_number'],
                                       "verbose" : learning_configuration["verbose"],
                                       "number_of_worker_selfplay": learning_configuration['number_of_worker_selfplay']}
                   }
        list_holder.append(dict_lc)

    if not None in [muzero , mcts , gameplay , env , learning_configuration]:
        try: rendermode = env.spec.kwargs['render_mode'] if env.spec.kwargs['render_mode'] != None else None
        except: rendermode = None
        dict_playgame = {"play_game_from_checkpoint":{"model_tag" : learning_configuration['model_tag_number'],
                                                      "model_device" : muzero.device,
                                                      "mcts_with_or_without_dirichlet_noise" : True,
                                                      "number_of_monte_carlo_tree_search_simulation" : mcts.num_simulations,       
                                                      "temperature" : 0,
                                                      "game_iter" : gameplay.limit_of_game_play,
                                                      "slow_mo_in_second" : 0.0,
                                                      "render" : rendermode,
                                                      "verbose" : True}
                         }
        list_holder.append(dict_playgame)

    if len(list_holder) != 0:
        json_config = {k:v for d in tuple(list_holder) for k,v in d.items()}

        if learning_configuration != None:
            with open(f"config/experiment_{learning_configuration['model_tag_number']}_config.json", "w") as f:
                json.dump(json_config, f, indent=4)

    if save_codebase:
        directory = os.getcwd()
        zip_file = zipfile.ZipFile(f"config/experiment_{learning_configuration['model_tag_number']}_codebase.zip", 'w')
        for filename in os.listdir(directory):
            if filename.endswith('.py'):
                zip_file.write(os.path.join(directory, filename), arcname=filename)
        zip_file.close()


##########################################################################################################################


def model_obs(rgb_obs=False,observation_space_dimensions=None):
    if rgb_obs:
        observation_dimension_per_model = (98, 98, 3)
    else:
        observation_dimension_per_model = obs_space(observation_space_dimensions)
    return observation_dimension_per_model

def obs_space(obs):
    def checker(container):
        if type(container) == gym.spaces.Discrete:
            return torch.tensor(1)
        if type(container) == gym.spaces.box.Box:
            return torch.prod(torch.tensor(list(container.shape)))

    if type(obs) in [gym.spaces.tuple.Tuple, tuple]:
        return int(sum(checker(i) for i in obs))
    else:
        return int(checker(obs))

def load_back_up_buffer(path):
    if isinstance(path,str):
        with open(path, 'rb') as handle:
            store = pickle.load(handle)
    elif isinstance(path,list):
        store=[]
        for i in path:
            with open(i, 'rb') as handle:
                store += pickle.load(handle)


def human_demonstration_buffer_builder(
         gym_game = "CartPole-v1", 
         render_mode = "human", 
         number_of_bin_action = 10, 
         mode_of_bin_action = "linear_bin",
         discount = 0.997,
         limit_of_game_play = 500,
         rgb_observation = False,
         keyboard_map_filename = None,
         set_default_noop = None,
         path_to_store_game = 'config/filename.pickle'
         ):

    try:
        rendermode = gym.make("CartPole-v1", render_mode=None).metadata['render_modes'][0]
        env = gym.make(gym_game, render_mode=rendermode) 
    except:
        env = gym.make(gym_game, render_mode=render_mode)

    observation_dimension = model_obs(False,env.observation_space )
    action_space = Gym_space_transform(bin=number_of_bin_action, mode=mode_of_bin_action)
    action_space.design_observation_space(env.action_space)
    action_dictionnary = action_space.dictionary
    action_dimension = action_space.dict_shape[0]

    gameplay = Game(gym_env = env, 
                    discount = discount,
                    limit_of_game_play = limit_of_game_play, 
                    observation_dimension =observation_dimension, 
                    action_dimension = action_dimension,
                    rgb_observation = rgb_observation,
                    action_map = action_dictionnary,
                    priority_scale=1)

    if keyboard_map_filename == None:
        gameplay.create_keyboard_to_map()

    store_game = []
    condition_to_continue = True
    while condition_to_continue:
        game = copy.deepcopy(gameplay)
        game.load_keymap(filename_keyboard_map = keyboard_map_filename)
        game.play_record(set_default_noop = set_default_noop)
        game.env = None
        store_game.append(game)
        keyboard = input("Do you want to play an other game ( Y / N ): ")
        if "y" in keyboard.lower():
            print("Start another game...")
        else:
            condition_to_continue = False
            print("Stop self play recording.")
    with open(path_to_store_game, 'wb') as handle:
        pickle.dump(store_game, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Save all game to: {path_to_store_game}")

##########################################################################################################################
def Show_all_gym_env():
    for h in [ i[0] for i in list(gym.envs.registry.items())]:
        print(h)
        
# # # to see all Gym env available:
def Show_specific_gym_env_structure(env):
    env = gym.make(env)
    print(f'{env} :| observation space: {env.observation_space} | action space : {env.action_space} |')

##########################################################################################################################


# # # benchmark speed
# # import cProfile, pstats
# # profiler = cProfile.Profile()
# # profiler.enable()
# # <<<< Function >>>>
# # profiler.disable()
# # stats = pstats.Stats(profiler).sort_stats('cumtime')
# # stats.print_stats()
# # raise Exception("stop test")




# # # hyperparameters tuning pseudo-code
# import ray
# ray.init()
# @ray.remote
# def optimize_hyperparameters(data, num_episodes, learning_rate, hidden_size,
#                              num_simulations, discount_factor):
#   # Train the MuZero algorithm using the provided
#   # hyperparameters.
#   model = MuZero(env, learning_rate, hidden_size, num_simulations, discount_factor)
#   model.train(data, num_episodes)
#   # Evaluate the performance of the trained model
#   # on a validation set.
#   score = evaluate(model, validation_data)
#   return score
# # Define the range of possible values for each
# # hyperparameter.
# hyperparameters = {
#     "learning_rate": [0.001, 0.01, 0.1],
#     "hidden_size": [32, 64, 128],
#     "num_simulations": [10, 20, 30],
#     "discount_factor": [0.9, 0.95, 0.99]}
# # Use grid search to evaluate the performance of
# # the MuZero algorithm for each combination of
# # hyperparameters.
# best_hyperparameters = {}
# best_score = -float("inf")
# for hp in itertools.product(*hyperparameters.values()):
#   # Set the current hyperparameters.
#   learning_rate, hidden_size, num_simulations, discount_factor = hp
#   # Use ray.put to transfer the data needed to
#   # train and evaluate the model to the remote
#   # function.
#   score = ray.get(optimize_hyperparameters.remote(
#       data, num_episodes, learning_rate, hidden_size,
#       num_simulations, discount_factor))
#   # If the current hyperparameters give the best
#   # performance so far, save the hyperparameters
#   # and the score.
#   if score > best_score:
#     best_hyperparameters = hp
#     best_score = score
# # Return the hyperparameters that give the best
# # performance on the validation set.
# return best_hyperparameters
