import numpy as np
import torchvision.transforms as transforms
import torch

import random
import time
import json
import gymnasium as gym
# # # for more details on the Game class
# # # refere to the pseudocode available at https://arxiv.org/src/1911.08265v2/anc/pseudocode.py

class Game():
    def __init__(self, 
                 gym_env = None, discount = 0.95, limit_of_game_play = float("inf"), 
                 observation_dimension = None, action_dimension = None, 
                 rgb_observation = None, action_map = None , priority_scale=1):
        """
        Init game
        
        Parameters
        ----------
            gym_env (gym_class): 
                The gym env (game) use for the learning and inference.
                Defaults to None.
            
            discount (float): 
                The discount factor for the calcul of the value
                Defaults to 0.95.
            
            limit_of_game_play (int): 
                Maximum number of game allow per selfplay
                Defaults to float("inf").
            
            observation_dimension (int): 
                The dimension of the observation space.
                Defaults to None.
            
            action_dimension (int): 
                The dimension of the action space .
                Defaults to None.
            
            rgb_observation (bool): 
                Bool value True or False that tell you to use the rgb render as observation
                Defaults to None.
            
            action_map (dict): 
                Dict containing the map between integer and possible mouve of the game
                Defaults to None.
            
            priority_scale (float):
                scale the new priority value ( beta for priority in the paper)
                Defaults to 1.
        """     
        self.env = gym_env
        self.action_map = action_map

        self.discount = discount
        assert isinstance(discount,float) and discount >= 0 , "discount ∈ float | {0 < discount < +inf)" 
        self.limit_of_game_play = limit_of_game_play
        assert isinstance(limit_of_game_play,(float,int)) and limit_of_game_play >= 0, "limit_of_game_play ∈ int || float | {1 < limit_of_game_play < +inf)" 
        self.action_space_size = action_dimension
        assert isinstance(action_dimension,int) and action_dimension >= 1, "action_dimension ∈ float | {1 < action_dimension < +inf)" 
        self.rgb_observation = rgb_observation
        assert isinstance(rgb_observation,bool), "rgb_observation ∈ bool "
        self.done = False
        assert isinstance(self.done,bool) , "self.done ∈ bool"
        self.priority_scale = priority_scale
        assert isinstance(priority_scale,(float,int)) and 0 <= priority_scale <= 1, "priority_scale ∈ float | {0 < priority_scale < 1)" 

        
        #game storage
        self.action_history = []
        self.rewards = []
        self.policies = []
        self.root_values = []
        self.child_visits = []
        self.observations = []
        
        #Status to know if the game was already reanalyze
        self.reanalyzed = False

        shape = observation_dimension[:-1] if type(observation_dimension) == tuple else None #(24,24)
        if shape != None:
            self.transform_rgb = transforms.Compose([lambda x : x.copy().astype(np.uint8), #make a copy of the array and change type to uint8(allow the next transform to rescale)
                                                    transforms.ToTensor(),       #will permute dimension to the appropiate channel for image and rescale between 0 and 1
                                                    transforms.Resize(shape),  #resize the image
                                                    lambda x : x[None,...] ])     #add an extra dimension at the beginning for batch
        else: 
            self.transform_rgb = None
    
    def tuple_test_obs(self,x):
        if isinstance(x,tuple):
            x = x[0]
        return x
    
    def observation(self,observation_shape=None,
                        iteration=0,
                        feedback=None):
        
        #manage initial observation
        if iteration == 0 and feedback == None: 
            state = self.env.reset(seed=random.randint(0, 100000))
            if self.rgb_observation:
                try:
                    state =  self.tuple_test_obs(self.render())
                except:
                    state = self.transform_rgb(self.tuple_test_obs(state))
            else:
                state = self.flatten_state(self.tuple_test_obs(state))
                
        #manage initial and feedback observation of reanalyze:
        elif not isinstance(feedback,(tuple,type(None))):
            state = feedback.observations[iteration]
            if iteration == 0:
                self.reanalyzed = True
            
        #manage feedback observation
        else:
            state = feedback[0]
        self.feedback_state = state
        return state
    
    def step(self,action):
        try: 
            next_step = (self.env.step(action))
        except:
            obs = self.feedback_state
            reward = min(-len(self.rewards),-self.limit_of_game_play,-1)
            done = self.done
            next_step = (obs,reward,done)
        return next_step
    
    def close(self):
        return self.env.close()
    
    def reset(self):
        self.env.reset()
        
    def vision(self):
        return self.env.render()
 
    def render(self):
        return self.transform_rgb(self.env.render())

    def flatten_state(self, state):
        if isinstance(state,tuple):
            state = torch.tensor([i.tolist() for i in state if isinstance(i,np.ndarray)] , 
                                 dtype=torch.float
                                 ).flatten()[None,...]
        elif isinstance(state,list):
            state = torch.tensor(state , 
                                 dtype=torch.float
                                 ).flatten()[None,...]
        elif isinstance(state,np.ndarray):
            state = torch.tensor(state.tolist() ,
                                 dtype=torch.float
                                 ).flatten()[None,...]
        else:
            try: 
                state =  torch.tensor([float(i) for i in state] ,
                                       dtype=torch.float
                                       ).flatten()[None,...]
            except: 
                state = torch.tensor([float(state)] ,
                                         dtype=torch.float
                                         ).flatten()[None,...]
        return state
    
    @property
    def terminal(self):
        #tell you if the game continue or stop with bool value
        return self.done
    
    @property
    def game_length(self):
        #return the lenght of the game
        return len(self.action_history)
    
    def store_search_statistics(self, root):
        # store policy without temperature rescale using mcts root first children
        visit_count = np.array([child.visit_count 
                                for child in root.children.values()],
                                dtype=np.float64)
        if visit_count.sum() >= 3:
            policy = visit_count/visit_count.sum()
        else:
            policy = np.array([root.children[u].prior 
                              for u in list(root.children.keys())],
                              dtype=np.float64)
            policy = self.softmax_stable(policy , temperature = 0)
        
        #provide policy without temperature
        self.child_visits.append(policy)
        #provide mcts value_sum
        self.root_values.append(root.value())
        
    def policy_action_reward_from_tree(self,root):
        action = np.array(list(root.children.keys()))
        policy = np.array([root.children[u].visit_count for u in list(root.children.keys())], dtype=np.float64)
        if policy.sum() <= 1 :
            policy = np.array([root.children[u].prior for u in list(root.children.keys())], dtype=np.float64)
        reward = np.array([root.children[u].reward for u in list(root.children.keys())], dtype=np.float64)

        return action, policy, reward
        
    def softmax_stable(self, tensor , temperature = 1):
        if temperature >= 0.3:
            tensor = tensor**(1/temperature)
        return tensor/tensor.sum()

    def select_action(self,action,policy,temperature):
        if temperature > 0.1 or len(set(policy)) == 1:
            selected_action = np.random.choice(action, p=policy)
        else:
            selected_action = action[np.argmax(policy)]
        return selected_action
    
    def onehot_action_encode(self,selected_action):
        action_onehot_encoded = np.zeros(self.action_space_size)
        action_onehot_encoded[selected_action] = 1
        return action_onehot_encoded
    
    def policy_step(self, root = None , temperature = 0 , feedback = None, iteration = 0):
        
        #generate action and policy
        action, policy, reward = self.policy_action_reward_from_tree(root)

        # if temperature over the treshhold of 0.3 select 
        # the select an action base on policy distribution
        # and make sure the policy sum to 1 (can glitch with big number rounding)
        policy = self.softmax_stable(policy , temperature = temperature)
        selected_action = self.select_action(action,policy,temperature)

        # # # return one hot encoded action from the discrete action
        action_onehot_encoded = self.onehot_action_encode(selected_action)

        #run env step or next reanalyze observavation
        if isinstance(feedback,(tuple,type(None))):
            # # # apply mouve and return variable of the env
            # # # save game variable to a list to return them 
            #contain [observation, reward, done, info] + [meta_data for som gym env]
            step_output = self.step(self.action_map[selected_action])

            #Get the new observation generate by step 
            if self.rgb_observation : 
                try: observation = self.render()
                except : observation = self.transform_rgb(step_output[0])
            else:
                observation = self.flatten_state(step_output[0])

            # # # save game variable to a list to return them 
            #contain [observation, reward, done, info] + [meta_data for som gym env]
            step_val = (observation,)+step_output[1:]
        else:
            step_val = [feedback.observations[iteration+1],
                        feedback.rewards[selected_action+1],
                        iteration+2 >= len(feedback.observations)-1]
            
        # save/record the policy during self_play
        # with open(f'report/softmax_model_policy_printed.txt', "a+") as f:
        #     print(selected_action,policy, file=f)
        
        # # # save game variable to class storage
        self.observations.append(step_val[0])
        self.rewards.append(step_val[1])
        self.policies.append(policy)
        self.action_history.append(action_onehot_encoded)
        
        # # # done is the parameter of end game [False or True]
        c_max_limit = self.limit_of_game_play != len(self.observations)
        self.done = step_val[2] if c_max_limit  else False
        
        return step_val


    def make_image(self, index):
        # # # select observation AKA state at specific index
        return self.observations[index]

    def make_extended_image(self, index,num_unroll):
        # # # select observation AKA state at specific index
        store_obs = []
        for i in range(index,index+num_unroll):
            try:
                store_obs.append(self.observations[i])
            except:
                store_obs.append(store_obs[-1]*0)
        return store_obs
    
    #NEED TO EXPLAIN EACH STEP
    def make_target(self, state_index, num_unroll, td_steps):
        
        targets = []
        
        for current_index in range(state_index, state_index + num_unroll):
            
            bootstrap_index = current_index + td_steps
            
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else: value = 0.0
            
            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i 

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else: last_reward = 0.0

            if current_index < len(self.root_values):
                targets.append([value, last_reward,self.child_visits[current_index]])
            else: targets.append([0.0, last_reward, np.zeros(self.action_space_size,dtype=np.float64)]) # absorbing state
            
        return targets

    def make_priority(self, td_steps):
        
        target_value = []
    
        for current_index in range(len(self.root_values)):
            
            bootstrap_index = current_index + td_steps

            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else: value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i 
                
            if current_index < len(self.root_values):
                target_value.append(value)
            else: target_value.append(0) # absorbing state
        #priority_scale determine the size of value, if you attempt do use game with huge reward it will renorm them to a more computable unit
        priority_position = np.abs(np.array(self.root_values) - np.array(target_value))**self.priority_scale 
        priority_game = np.max(priority_position)
        return priority_position , priority_game
    
    ########################################################
    ### add for demonstration buffer of muzero reanalyze ###
    ########################################################
    def create_keyboard_to_map(self):
        dict_map = self.action_map
        
        try: 
            action_meaning_map = self.env.unwrapped.get_action_meanings()
            print("Meaning of action: (action : meaning , ...")
            print({i:action_meaning_map[i]for i in range(len(action_meaning_map))})
        except: pass
        
        lenght_dict = len(dict_map)
        dict_keyboard_map = {}
        print(f"Number of input to map to keyboard: {lenght_dict} ")
        print("You can stop the process at any moment if you write STOP")
        
        cond_user_permission = True
        while cond_user_permission:
            
            for i in range(lenght_dict):
                keyboard = input(f" The keyboard you want to set with {dict_map[i]} : (Write keyboard key and press ENTER) ")
                
                if "stop" in keyboard.lower():
                    cond_user_permission = False
                    print(" Stop process...")
                    break
                dict_keyboard_map[keyboard] = dict_map[i]
                
            if "stop" not in keyboard.lower():
                
                keyboard = input(f"Are you satify with this set up ( Y / N ): {dict_keyboard_map} ")
                if "y" in keyboard.lower():
                    cond_user_permission = False
                    
                    path_name = input("Povide a name for the saved keyboard map at path: config/NAME_keyboard_map.json : ")
                    save_path = f"config/{path_name}.json"
                    with open(save_path, "w") as f:
                        json.dump(dict_keyboard_map, f)
                    self.keyboard_map_path = save_path
                    print(f"End keyboard map and save at : {save_path}")
                    
                else:
                    print("Restart map from the beginning")

    def load_keymap(self, filename_keyboard_map = None):
            #open keyboardmap
            if filename_keyboard_map is None: filename_keyboard_map = self.keyboard_map_path

            filename_keyboard_map = filename_keyboard_map[:-5] if ".json" in filename_keyboard_map else filename_keyboard_map
            with open(f"{filename_keyboard_map}.json", 'r') as openfile:
                self.keyboard_map = json.load(openfile)

            self.keyboard_keys = list(self.keyboard_map.keys())
            self.keyboard_values = list(self.keyboard_map.values())
            self.keyboard_len = len(self.keyboard_values)


    def play_record(self,set_default_noop = None):
        import keyboard

        if not self.keyboard_len:
            print("You need to run gameplay.load_keymap( filename_keyboard_map = ? )")
            return 
        
        #test for availability of noop
        try: noop_available = "NOOP" in self.env.unwrapped.get_action_meanings()
        except: noop_available = False
        
        
        #initial observation
        print("Start simulation...")
        self.env.reset(seed=random.randint(0, 100000))
        try: self.vision()
        except : print(self.flatten_state(step_output[0]))
        
        # loop
        self.done = False
        while not self.done:
            # detect key of the action
            if noop_available : 
                # record key of the action for 1/30sec
                keyboard.start_recording(recorded_events_queue=None)
                #30|hz|fps
                time.sleep(1/30)
                event = keyboard.stop_recording()
                event = [ i.name for i in event ] if len(event) > 0 else [None]
                if event[0] in self.keyboard_keys:  
                    action = self.keyboard_map[event[0]]
                else:
                    action = self.env.unwrapped.get_action_meanings().index("NOOP")
            elif not set_default_noop is None:
                keyboard.start_recording(recorded_events_queue=None)
                #30hz/fps
                time.sleep(1/30)
                event = keyboard.stop_recording()
                event = [ i.name for i in event ] if len(event) > 0 else [None]
                if event[0] in self.keyboard_keys:  
                    action = self.keyboard_map[event[0]]
                else:
                    if set_default_noop == "random": 
                        set_default_noop = self.keyboard_map[random.randint(0, self.keyboard_len)]
                    else:
                        action = set_default_noop
            else:
                valid_input = True
                while valid_input:
                    # detect key of the action
                    event = keyboard.read_event(suppress=False)
                    if event.name in self.keyboard_keys:  
                        action = self.keyboard_map[event.name]
                        valid_input = False
                    else:
                        print(f"Key | {event.name} | isn't a valid key")  
                        print(f"Valide key are: {self.keyboard_map.keys()}") 
                        
            #add random action
            #make policy
            index_policy = list(self.keyboard_values).index(action)
            policy = np.zeros(self.keyboard_len)
            policy[index_policy] = 1
            
            #action step
            step_output = (self.env.step(action))

            #render
            
            try: 
                if self.env.render_mode is not None:
                    self.vision()
                else: 
                    raise Exception()
            except : 
                print(self.flatten_state(step_output[0]))

            #generate obs 
            if self.rgb_observation : 
                try: observation = self.render()
                except : observation = self.transform_rgb(step_output[0])
            else:
                observation = self.flatten_state(step_output[0])
            
            #reformate data with wanted obs
            step_val = (observation,)+step_output[1:]
            # # # save game variable to class storage
            self.observations.append(step_val[0])
            self.rewards.append(step_val[1])
            self.policies.append(policy)
            self.action_history.append(policy)
            self.child_visits.append(policy)
            self.root_values.append(step_val[1] * self.discount**(len(self.rewards)-1) )
            
            # # # done is the parameter of end game [False or True]
            c_max_limit = self.limit_of_game_play != len(self.observations)
            self.done = step_val[2] if c_max_limit  else False
        self.env.close()
        print(f"| End simulation | score: {sum(self.rewards)} , number of action : {len(self.rewards)}")





