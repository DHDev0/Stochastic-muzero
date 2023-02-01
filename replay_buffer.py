import numpy as np
import torch

# # # https://arxiv.org/pdf/1911.08265.pdf [page: 3, 13]
class ReplayBuffer():
    def __init__(self, 
                 window_size,
                 batch_size, 
                 num_unroll, 
                 td_steps,
                 game_sampling = "uniform",
                 position_sampling = "uniform",
                 reanalyze_stack = [],
                 reanalyse_fraction = 0.5,
                 reanalyse_fraction_mode = "chance"
                 ):
        """
        Init replay buffer
        
        Parameters
        ----------
            window_size (int): Maximum number of game store in the replay buffer
            (each self_play add one game and take at one if the replay buffer is
            full)
            
            batch_size (int): Number of game sample in the batch
            
            num_unroll (int): number of mouve in the batch for each game in the
            batch 
            
            td_steps (int): The td_step in the MuZero algorithm is a learning
            step that compares expected and observed rewards and transitions in
            the environment to update and improve the prediction model.
            
            game_sampling (str): choice between "uniform" and "priority".
            "uniform": will pick game randomly in the buffer "priority": will
            pick game according to a priority ration in the buffer Defaults to
            "uniform".
            
            position_sampling (str): choice between "uniform" and "priority".
            "uniform": will pick a mouve inside a game randomly in the buffer
            "priority": will pick a mouve inside a game according to a priority
            ration in the buffer . Defaults to "uniform".
            
            reanalyze_stac(replay_buffer_class): Defaults to []
            
            reanalyse_fraction (float): Defaults to 0.5
            
            reanalyse_fraction_mode (str): choice between "chance" and "ratio".
            "chance": pourcentage of chance to reanalyze base on bernoulli
            distribution. need less compute. 
            "ratio": decide to reanalyze looking a the proportion of the buffer
            from replaybuffer and buffer from reanalyze buffer ration in the
            buffer. Defaults to "chance".
        """        
        
        self.window_size = window_size
        assert (isinstance(window_size, int) and window_size >= 1), "window_size ∈ int | {1 < window_size < +inf)"

        self.batch_size = batch_size
        assert (isinstance(batch_size,int) and batch_size >= 1) , "batch_size ∈ int | {1 < batch_size < +inf)"

        self.num_unroll = num_unroll
        assert (isinstance(num_unroll,int) and num_unroll >= 0), "num_unroll ∈ int | {0 < num_unroll < +inf)"

        self.td_steps = td_steps
        assert (isinstance(td_steps,int) and td_steps >=0), "td_steps ∈ int | {0 < td_steps < +inf)"

        self.game_sampling = game_sampling
        assert isinstance(game_sampling,str) and game_sampling in ["priority","uniform"] , "game_sampling ∈ {priority,uniform) ⊆ str"

        self.position_sampling = position_sampling
        assert isinstance(position_sampling,str) and position_sampling in ["priority","uniform"] , "position_sampling ∈ {priority,uniform) ⊆ str"

        self.reanalyze_stack = reanalyze_stack
        assert isinstance(reanalyze_stack,list) , "reanalyze_stack ∈ list"

        self.reanalyse_fraction = reanalyse_fraction
        assert (isinstance(reanalyse_fraction,float) and 0 <= reanalyse_fraction <= 1), "reanalyse_fraction ∈ float | {0 < reanalyse_fraction < 1)"

        self.reanalyse_fraction_mode = reanalyse_fraction_mode
        assert isinstance(reanalyse_fraction_mode,str) and reanalyse_fraction_mode in ["ratio","chance"] , "reanalyse_fraction_mode ∈ {ratio,chance) ⊆ str"

        self.buffer = []
        self.prio = []
        self.prio_position = []
        self.prio_game = []
        self.big_n_of_importance_sampling_ratio = 0

    def load_back_up_buffer(self,path):
        self.load_path = path
        import pickle
        if isinstance(path,str):
            with open(path, 'rb') as handle:
               self.buffer = pickle.load(handle)
        elif isinstance(path,list):
            for i in path:
                with open(i, 'rb') as handle:
                    self.buffer += pickle.load(handle)

    def save_buffer(self,path):
        self.path_save = path
        import pickle
        for i in self.buffer:
            i.env = None
        with open(path, 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_game(self, game):
        
        if len(self.buffer) > self.window_size:
            self.big_n_of_importance_sampling_ratio -= self.buffer[0].game_length
            self.buffer.pop(0)
            if self.game_sampling == "priority":
                self.prio_game.pop(0)
            if self.position_sampling == "priority":
                self.prio_position.pop(0)
            
                
        if "priority" in [self.game_sampling,self.position_sampling]:
            # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
            # # # ν is the mcts.root_value (search value) and z the generated target_value with td_step(observed n-step return)
            p_i_position, p_i_game = game.make_priority( self.td_steps )
            
            # # # individual p_i value for each position
            self.prio_position.append(p_i_position)
            
            # # # average p_i value for each game
            self.prio_game.append(p_i_game)
            self.soft_prio_game = np.array(self.prio_game) / np.sum(np.array(self.prio_game))
            
        # # # save the game into the buffer(storage)self.buffer[0].game_length
        self.buffer.append(game)
        self.big_n_of_importance_sampling_ratio += game.game_length
        
        if not game.reanalyzed:
            self.reanalyse_buffer_save_game(game)
            
        
    def sample_game(self):
        # # # # Sample game from buffer either uniformly or according to some priority.
        # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
        
        if self.game_sampling == "priority":
        # # # priority sampling
            position =  np.random.choice(range(self.soft_prio_game.size), p=self.soft_prio_game)
                        
        elif self.game_sampling == "uniform":
        # uniform sampling
            position = np.random.choice(range(len(self.buffer)))
                        
        return position
    
    def sample_position(self, game):
 
        if game.game_length == 0:
            raise Exception("Game need to return at least one reward")
        
        elif self.position_sampling == "priority":
            tag = self.buffer.index(game)
            soft_prio_position = self.prio_position[tag]/self.prio_position[tag].sum()
            self.buffer[tag].mouve_prio = soft_prio_position
            # # priority sampling
            position =  np.random.choice(list(range(len(soft_prio_position))), p=soft_prio_position)
            

        elif self.position_sampling == "uniform":
            # # uniform sampling
            position =  np.random.randint(0, game.game_length-1)

        
        return position
    
    def fill_gap_empty_action(self, actions):
        # # # Add [0,0] to keep uniform len inside the batch 
        # # # if the num_unroll is too big for the sample
        # # # The zero sequence will be mask later on in the loss
        # # # They are absorbing state
        actions = actions[:self.num_unroll]
        lenght_action_against_num_unroll = (self.num_unroll - len(actions))
        if lenght_action_against_num_unroll > 0:
            actions += [np.zeros(actions[0].shape)] * lenght_action_against_num_unroll
        return actions
    
    def sample_batch(self):
         # # # contain: [<GameLib.Game object at 0x0000000000>,.....]
         # # # return a game choice uniformly(random) or according to some priority
        games_pos = [(self.buffer[i],i) for i in [self.sample_game() for _ in range(self.batch_size)]]
        # # # contain: [(<GameLib.Game object at 000000000000>, 5).....]
        # # # return a game and position inside this game choice uniformly(random)
        # # # or according to some priority
        game_pos_and_mouve_pos = [(g, g_p, self.sample_position(g)) for g,g_p in games_pos] 
                
        # # # return [([state(the observation)], [action array(onehot encoded)], [value, reward, policy]), ... *batch_size]
        # # # They are your X: [[state(the observation)], [action array(onehot encoded)],...] and Y: [[value, reward, policy],...]
        bacth = [(
                g.make_extended_image(m_p,self.num_unroll),
                self.fill_gap_empty_action(g.action_history[m_p:]),
                g.make_target(m_p, self.num_unroll, self.td_steps)
                ) for (g, g_p, m_p) in game_pos_and_mouve_pos]

        #game and mouve position
        game_pos = np.array([(i[1],i[2]) for i in game_pos_and_mouve_pos])
        
        if "priority" in [self.game_sampling,self.position_sampling] :
            #P(i)
            priority = np.array([self.soft_prio_game[i[1]] * self.buffer[i[1]].mouve_prio[ i[2] ] for i in game_pos_and_mouve_pos ])

            # 1/n * 1/P(i)
            importance_sampling_ratio = 1 / ( self.big_n_of_importance_sampling_ratio * priority ) 
            return (bacth , importance_sampling_ratio , game_pos)

        else:
            return (bacth , np.array([0]) , game_pos)
            
        
    def update_value(self,new_value,position):
        if "priority" in [self.position_sampling ,self.game_sampling]:
            for count,i in enumerate(position):
                lenght_game = self.buffer[i[0]].game_length - 1
                for remainder, h in enumerate(range(i[1],min(self.num_unroll + i[1] , lenght_game))):
                    self.prio_position[i[0]][h] = new_value[remainder][count][0]
                self.prio_game[i[0]] = np.max(self.prio_position[i[0]])
            
    ###############################
    ### add of muzero reanalyze ###
    ###############################
    
    def reanalyse_buffer_save_game(self,game):
        for reanalyze_buffer in self.reanalyze_stack:
            reanalyze_buffer.save_game(game)
            
    def reanalyse_buffer_sample_game(self):
        reanalyze_buffer_with_game = [i for i in self.reanalyze_stack if len(i.buffer) > 0 ]
        selected_buffer = np.random.choice(reanalyze_buffer_with_game)
        game_to_reanalyse = selected_buffer.sample_game()
        return game_to_reanalyse
            
    def should_reanalyse(self):
        
        reanalyze_stack  = [i for i in self.reanalyze_stack if len(i.buffer) > 0 ]
        if len(reanalyze_stack) >= 1:
            if self.reanalyse_fraction_mode == "ratio":
                buffer = np.array([len(i.observations) for i in self.buffer])
                reanalyzer = np.array([len(h.observations) for i in reanalyze_stack for h in i.buffer ])
                
                buffer_total_amount_of_obs = buffer.sum()
                reanalyze_total_amount_of_obs = reanalyzer.sum()
                
                buffer_mean_episode_length = buffer.mean()
                reanalysed_mean_episode_length = reanalyzer.mean()
                
                actual = buffer_total_amount_of_obs  / (buffer_total_amount_of_obs + reanalyze_total_amount_of_obs)

                target = self.reanalyse_fraction + (self.reanalyse_fraction - actual) / 2
                target = max(0, min(1, target))

                # Correct for reanalysing only part of full episodes.
                fresh_fraction = 1 - target
                parts_per_episode = max(1,buffer_mean_episode_length / reanalysed_mean_episode_length)
                fresh_fraction /= parts_per_episode
                return torch.bernoulli(torch.tensor(1 - fresh_fraction)).bool()
            if self.reanalyse_fraction_mode == "chance":
                return torch.bernoulli(torch.tensor(self.reanalyse_fraction)).bool()
        else:
            return False

class ReanalyseBuffer:
    def __init__(self,max_buffer_size = float("inf") , keep_or_delete_buffer_after_reanalyze = True):
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.keep_or_delete_buffer_after_reanalyze = keep_or_delete_buffer_after_reanalyze
        
    def load_back_up_buffer(self,path):
        self.load_path = path
        import pickle
        if isinstance(path,str):
            with open(path, 'rb') as handle:
               self.buffer = pickle.load(handle)
        elif isinstance(path,list):
            for i in path:
                with open(i, 'rb') as handle:
                    self.buffer += pickle.load(handle)

    def save_buffer(self,path):
        self.path_save = path
        import pickle
        for i in self.buffer:
            i.env = None
        with open(path, 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                         
    def save_game(self, game):
        self.buffer.append(game)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)

    def sample_game(self):
        sampled_game = np.random.choice(self.buffer)
        if not self.keep_or_delete_buffer_after_reanalyze:
            self.buffer.pop(self.buffer.index(sampled_game))
        return sampled_game

# could use super() instead to get equivalent function
class DemonstrationBuffer:
    """A reanlayse buffer of a fixed set of demonstrations.

    Can be used to learn from existing policies, human demonstrations or for
    Offline RL.
    """
    def __init__(self, max_buffer_size = float("inf") , keep_or_delete_buffer_after_reanalyze = True):
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.keep_or_delete_buffer_after_reanalyze = keep_or_delete_buffer_after_reanalyze

    def load_back_up_buffer(self,path):
        self.load_path = path
        import pickle
        if isinstance(path,str):
            with open(path, 'rb') as handle:
               self.buffer = pickle.load(handle)
        elif isinstance(path,list):
            for i in path:
                with open(i, 'rb') as handle:
                    self.buffer += pickle.load(handle)

    def save_buffer(self,path):
        self.path_save = path
        import pickle
        for i in self.buffer:
            i.env = None
        with open(path, 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_game(self, game):
        # self.buffer.append(game)
        pass

    def sample_game(self):
        sampled_game = np.random.choice(self.buffer)
        if not self.keep_or_delete_buffer_after_reanalyze:
            self.buffer.pop(self.buffer.index(sampled_game))
        return sampled_game

class MostRecentBuffer:
    """A reanalyse buffer that keeps the most recent games to reanalyse."""
    def __init__(self,max_buffer_size = float("inf") , keep_or_delete_buffer_after_reanalyze = True):
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.keep_or_delete_buffer_after_reanalyze = keep_or_delete_buffer_after_reanalyze
        
    def load_back_up_buffer(self,path):
        self.load_path = path
        import pickle
        if isinstance(path,str):
            with open(path, 'rb') as handle:
               self.buffer = pickle.load(handle)
        elif isinstance(path,list):
            for i in path:
                with open(i, 'rb') as handle:
                    self.buffer += pickle.load(handle)

    def save_buffer(self,path):
        self.path_save = path
        import pickle
        for i in self.buffer:
            i.env = None
        with open(path, 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def save_game(self, game):
        self.buffer.append(game)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)

    def sample_game(self):
        sampled_game = np.random.choice(self.buffer)
        if not self.keep_or_delete_buffer_after_reanalyze:
            self.buffer.pop(self.buffer.index(sampled_game))
        return sampled_game


class HighestRewardBuffer:
    """A reanalyse buffer that keeps games with highest rewards to reanalyse."""
    def __init__(self,max_buffer_size = float("inf") , keep_or_delete_buffer_after_reanalyze = True):
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.keep_or_delete_buffer_after_reanalyze = keep_or_delete_buffer_after_reanalyze
        
    def load_back_up_buffer(self,path):
        self.load_path = path
        import pickle
        if isinstance(path,str):
            with open(path, 'rb') as handle:
               self.buffer = pickle.load(handle)
        elif isinstance(path,list):
            for i in path:
                with open(i, 'rb') as handle:
                    self.buffer += pickle.load(handle)

    def save_buffer(self,path):
        self.path_save = path
        import pickle
        for i in self.buffer:
            i.env = None
        with open(path, 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def biggest_rewards(self):
        return max(sum(i.rewards) for i in self.buffer)
    
    def save_game(self, game):
        if len(self.buffer) == 0:
            self.buffer.append(game)
        elif sum(game.rewards) > self.biggest_rewards():
            self.buffer.append(game)
            if len(self.buffer) > self.max_buffer_size:
                self.buffer.pop(0)

    def sample_game(self):
        sampled_game = np.random.choice(self.buffer)
        if not self.keep_or_delete_buffer_after_reanalyze:
            self.buffer.pop(self.buffer.index(sampled_game))
        return sampled_game

