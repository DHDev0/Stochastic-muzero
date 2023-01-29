import numpy as np
import torch

# # # refere to the pseudocode available at https://arxiv.org/src/1911.08265v2/anc/pseudocode.py

class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = 0
        self.reward = 0
        self.to_play = -1
        self.is_chance = False

    def expanded(self):
        return len(self.children) > 0

    def value(self) -> float:
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MinMaxStats(object):
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Player_cycle:
    def __init__(self,number_of_player: int = None , custom_loop : str = None):
        self.number_of_player = number_of_player
        self.custom_loop = custom_loop

        if self.custom_loop is not None and isinstance(self.custom_loop, str):
            self.loop_cycle = self.custom_cycle()
        elif self.number_of_player is not None and self.number_of_player >= 1:
            self.loop_cycle = self.modular_cycle()
        else:
            raise Exception("You have to provide a number of player >= 1 or a custom loop like : \"1>2>3\" ")
        
    def modular_cycle(self):
        self.cycle_map = torch.arange(0,self.number_of_player)
        self.global_origin = self.cycle_map[0]
        self.global_count = 0
        
    def custom_cycle(self):  
        self.cycle_map = torch.tensor([ float(i) for i in self.custom_loop.split(">")])
        self.global_origin = self.cycle_map[0]
        self.global_count = 0
        
    def proximate_player_step(self,player_index):
        return (player_index + 1) % self.cycle_map.size()[0]
    
    def global_step(self):
        player_in_play = self.global_count % self.cycle_map.size()[0]
        self.global_count = (1+self.global_count) % self.cycle_map.size()[0]
        return player_in_play
        
    def global_reset(self):
        self.global_count = 0
        
    def player_in_play(self,player_index):    
        return self.cycle_map[player_index % self.cycle_map.size()[0]]
  
    
class Monte_carlo_tree_search():
    def __init__(self,
                 pb_c_base=19652, 
                 pb_c_init=1.25,
                 discount=0.95, 
                 root_dirichlet_alpha=0.25, 
                 root_exploration_fraction=0.25,
                 num_simulations=10,
                 maxium_action_sample=2,
                 number_of_player = 1 , 
                 custom_loop = None):
              
        """
        Init the monte carlos tree search of muzero
        
        Parameters
        ----------
            pb_c_base (int): This is the base value used in the formula for
            calculating the exploration parameter (known as "Pb") in the MCTS
            algorithm. Pb determines the level of exploration that the algorithm
            should perform at each step, with a higher value resulting in more
            expl- oration and a lower value resulting in more exploitation.
            Defaults to 19652.
            
            pb_c_init (float): This is the initial value of the exploration
            parameter Pb. It determines the level of exp- loration that the
            algorithm should perform at the beginning of the search. Defaults to
            1.25.
            
            discount (float): This is the discount factor used in the MCTS al-
            gorithm. It determines the importance of future rewards relative to
            immediate rewards, with a hi- gher discount factor leading to a
            greater emphasis on long-term rewards. Defaults to 0.95.
            
            root_dirichlet_alpha (float): This is the alpha parameter of the
            Dirichlet distr- ibution used in the MCTS algorithm. The Dirichlet
            distribution is used to sample initial move probab- ilities at the
            root node of the search tree, with the alpha parameter controlling
            the level of explo- ration vs exploitation in the sampling process.
            Defaults to 0.25.
            
            root_exploration_fraction (float): This is the exploration fraction
            used in the MCTS algorithm. It determines the proportion of the
            sear- ch time that should be spent exploring the search tree, with a
            higher value resulting in more explora- tion and a lower value
            resulting in more exploitation. Defaults to 0.25.
            
            maxium_action_sample (int): provide the number of action sample
            during the mcts search. maxium_action_sample provide the width of
            the tree and num_simulations provide the length of the tree.
            Defaults to 2.
            
            num_simulationsn (int):
            Depth of the monte carlos tree search, how many future node tree you want to simulate 
            Defaults to 11.
        """        

        self.reset(pb_c_base, pb_c_init, discount,
                   root_dirichlet_alpha, root_exploration_fraction,
                   num_simulations,maxium_action_sample,
                   number_of_player, custom_loop)

    def reset(self, pb_c_base=19652, 
              pb_c_init=1.25,
              discount=0.95, 
              root_dirichlet_alpha=0.25, 
              root_exploration_fraction=0.25,
              num_simulations=10,
              maxium_action_sample=2,
              number_of_player = 1 , 
              custom_loop = None
              ):
        
        self.pb_c_base = pb_c_base
        assert isinstance(pb_c_base,int) and pb_c_base >= 1, "pb_c_base ∈ int | {1 < pb_c_base < +inf)"

        self.pb_c_init = pb_c_init
        assert isinstance(pb_c_init,float) and pb_c_init >= 0 , "pb_c_init ∈ float | {0 < pb_c_init < +inf)"
        
        self.discount = discount
        assert isinstance(discount,(int,float)) and discount >= 0, "discount ∈ float | {0 < discount < +inf)"
        
        self.root_dirichlet_alpha = root_dirichlet_alpha
        assert isinstance(root_dirichlet_alpha,float) and 0 <= root_dirichlet_alpha <= 1, "root_dirichlet_alpha ∈ float | {0< root_dirichlet_alpha < 1)"
        
        self.root_exploration_fraction = root_exploration_fraction
        assert isinstance(root_exploration_fraction,float) and 0 <= root_exploration_fraction <= 1 , "root_exploration_fraction ∈ float | {0 < root_exploration_fraction < 1)"
        
        self.maxium_action_sample = maxium_action_sample
        assert isinstance(maxium_action_sample,int) and maxium_action_sample >= 1, "maxium_action_sample ∈ int | {1 < maxium_action_sample < +inf)"
        
        self.num_simulations = num_simulations
        assert isinstance(num_simulations,int) and num_simulations >= 0, "num_simulations ∈ int | {0 < num_simulations < +inf)"
        
        self.number_of_player = number_of_player
        assert isinstance(number_of_player,int) and number_of_player >= 1, "number_of_player ∈ int | {1 < number_of_player < +inf)"
        
        self.custom_loop = custom_loop
        assert isinstance(custom_loop,str) or custom_loop is None , "custom_loop ∈ str | 1>2>3>3 "
        
        self.node = None
        self.model = None
        self.cycle = Player_cycle(number_of_player = number_of_player, custom_loop = custom_loop)

    def generate_root_hidden_state(self, observation):
        self.root = Node(0)
        self.min_max_stats = MinMaxStats()
        self.root.hidden_state = self.model.representation_function_inference(
            observation)


    def set_root_to_play_with_the_play_number(self, observation):
        # Monte Carlo Tree Search (MCTS), the to_play variable represents the player
        # whose turn it is to make a move in the current position being considered. This
        # information is used to determine which player's score to update in the MCTS
        # tree, as well as which player's actions to consider when selecting the next
        # move to explore.

        #This configuration always assume the same player is in play.
        self.root.to_play = self.cycle.global_step()


    def generate_policy_and_value(self):
        policy, value = self.model.prediction_function_inference(
            self.root.hidden_state)
        return policy, value


    def expand_the_children_of_the_root_node(self, policy):
        policy = policy[0]
        policy_reshape = (policy + 1e-12) 
        policy = policy_reshape / policy_reshape.sum()
        bound = policy.shape[0]
        for i in np.sort(np.random.choice(policy.shape[0],bound, p=policy, replace=False)):
            self.root.children[i] = Node(prior=policy[i])
            self.root.children[i].to_play = self.cycle.proximate_player_step(self.root.to_play)
            self.root.children[i].is_chance = False


    def add_exploration_noise_at_the_root(self, train):
        if self.num_simulations == 0 :
            train = False
            
        if train:
            actions = list(self.root.children.keys())
            noise = np.random.dirichlet(
                [self.root_dirichlet_alpha] * len(actions))
            frac = self.root_exploration_fraction
            for a, n in zip(actions, noise):
                self.root.children[a].prior = self.root.children[a].prior * \
                    (1 - frac) + n * frac


    def initialize_history_node_searchpath_variable(self):
        history = []
        self.node = self.root
        search_path = [self.root]
        return history, search_path


    def ucb_score(self, parent, child):
        pb_c = np.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        prior_score = (np.sqrt(parent.visit_count) *  pb_c * child.prior) / (child.visit_count + 1)
        if child.visit_count > 0:
            value_score = self.min_max_stats.normalize( child.reward + self.discount * child.value())
        else:
            value_score = 0

        return prior_score + value_score + np.random.uniform(low=1e-7, high=2e-7, size=1)[0]


    def select_child(self):
        if self.node.is_chance:
            # If the node is chance we sample from the prior.
            outcomes, probs = zip(*[(o, n.prior) for o, n in self.node.children.items()])
            outcomes, probs = list(outcomes), np.array(list(probs))
            remainder = np.abs((1 - probs + 1e-12).mean())
            probs = (probs + remainder)/(probs + remainder).sum()    
            outcome = np.random.choice(outcomes, p=probs)
            return outcome, self.node.children[outcome]
        
        _, action, child = max((self.ucb_score(self.node, child), action, child)
                               for action, child in self.node.children.items())
        return action, child


    def choice_node_to_expand_using_max_ucb_score(self, history, search_path):
        while self.node.expanded():
            action, self.node = self.select_child()
            history.append(action)
            search_path.append(self.node)
        return search_path[-2]


    def generate_reward_and_hidden_state(self, parent, history):
        reward, hidden_state = self.model.dynamics_function_inference(parent.hidden_state, history[-1])
        return reward, hidden_state
    def update_reward_and_hidden_state_for_the_chosen_node(self, reward, hidden_state):
        self.node.reward, self.node.hidden_state = reward, hidden_state
    def generate_policy_and_value_for_the_chosen_node(self, hidden_state):
        policy, value = self.model.prediction_function_inference(hidden_state)
        return policy, value
    
    def generate_hidden_state_chance_node(self, parent, history):
        hidden_state = self.model.afterstate_dynamics_function_inference(parent.hidden_state, history[-1])
        return hidden_state
    def update_hidden_state_for_the_chosen_chance_node(self, hidden_state):
        self.node.hidden_state = hidden_state
    def generate_policy_and_value_for_the_chosen_chance_node(self, hidden_state):
        policy, value = self.model.afterstate_prediction_function_inference(hidden_state)
        return policy, value


    def create_new_node_in_the_chosen_node_with_action_and_policy(self, policy, is_child_chance):
        policy = policy[0]
        policy_reshape = (policy + 1e-12) 
        policy = policy_reshape / policy_reshape.sum()
        bound = min(self.maxium_action_sample,policy.shape[0])
        for i in np.sort(np.random.choice(policy.shape[0],bound, p=policy, replace=False)):
            self.node.children[i] = Node(prior=policy[i])
            self.node.children[i].to_play = self.node.to_play if is_child_chance else self.cycle.proximate_player_step(self.node.to_play)
            self.node.children[i].is_chance = is_child_chance

    def back_propagate_and_update_min_max_bound(self, search_path, value):

        for bnode in reversed(search_path):
            bnode.value_sum += value if torch.equal(
                                                    self.cycle.player_in_play(self.root.to_play),
                                                    self.cycle.player_in_play(bnode.to_play)
                                                    ) else -value
            bnode.visit_count += 1
            self.min_max_stats.update(bnode.value())
            value = bnode.reward + self.discount * value


    def run(self, observation=None, model=None, train=True):

        self.model = model
        
        self.generate_root_hidden_state(observation)

        self.set_root_to_play_with_the_play_number(observation)

        policy, value = self.generate_policy_and_value()

        self.expand_the_children_of_the_root_node(policy)

        self.add_exploration_noise_at_the_root(train)

        for _ in range(self.num_simulations):

            history, search_path = self.initialize_history_node_searchpath_variable()

            parent = self.choice_node_to_expand_using_max_ucb_score(
                history, search_path)


            if parent.is_chance:
                reward, hidden_state = self.generate_reward_and_hidden_state(parent, history)
                self.update_reward_and_hidden_state_for_the_chosen_node(reward, hidden_state)
                policy, value = self.generate_policy_and_value_for_the_chosen_node(hidden_state)
                is_child_chance = False
            else:
                hidden_state = self.generate_hidden_state_chance_node(parent, history)
                self.update_hidden_state_for_the_chosen_chance_node(hidden_state)
                policy, value = self.generate_policy_and_value_for_the_chosen_chance_node(hidden_state)
                is_child_chance = True
                

            self.create_new_node_in_the_chosen_node_with_action_and_policy(policy,is_child_chance)

            self.back_propagate_and_update_min_max_bound(search_path, value)

        return self.root
    