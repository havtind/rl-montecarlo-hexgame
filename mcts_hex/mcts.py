from copy import deepcopy
import numpy as np
import random
import collections
import math
from .anet import ANET
from .hex import GameWorld
from typing import Callable


class Agent:
    """Combining all the parts of the RL system."""
    def __init__(self, board_size: int, episodes: int, save_episodes: list = [], \
                simulations: int=10, rbuf_ratio: float = 0.7) -> None:
        print('Agent instance is created')
        self.board_size = board_size
        self.episodes = episodes
        self.simulations = simulations
        self.RBUF = [] # Replay buffer: container with training cases for neural net
        self.RBUF_usage = rbuf_ratio # Fraction of RBUF used for training
        anet_input_dim = board_size**2+1
        anet_output_dim = board_size**2
        self.anet = ANET(input_length=anet_input_dim, output_length=anet_output_dim, epochs=10, lr=0.05)
        self.save_episodes = save_episodes
        
    def train(self, save_folder=1) -> None:
        """
        Use Monte Carlo Tree Search to explore the game of Hex.
        Use neural network during tree search to exploit acquired knowledge.
        Train neural network with statistics from tree search.
        """
        game_world = GameWorld(board_size=self.board_size)
        epsilon = 1
        epsilon_decrement = 0.95/self.episodes
        for episode in range(self.episodes):
            mcts = MCTS(game_world=game_world, predictor=self.anet.get_predictions, \
                simulations=self.simulations, epsilon=epsilon)
            self.RBUF = [] # Empty Replay BUFfer
            print('Episode: ', episode)
            print('    ', end='')
            state, gameover = mcts.init_game()
            while not gameover:
                print('. ', end='')
                new_state, distribution, gameover, winner = mcts.run(state) 
                self.RBUF.append((state, distribution)) # Training case: stats from the simulations
                state = new_state
            self._update_anet()
            epsilon = epsilon - epsilon_decrement
            print('')
            if episode in self.save_episodes:
                self._save_anet(episode, save_folder)

    def _update_anet(self):
        """Give data and labels to 'anet' and train network."""
        data, labels = self._get_training_cases()
        self.anet.update(data, labels)

    def _get_training_cases(self):
        """Return a fraction of 'RBUF' and divide into data and labels."""
        RBUF_size = len(self.RBUF)
        minibatch_size = round(self.RBUF_usage * RBUF_size)
        minibatch = random.sample(self.RBUF, minibatch_size)
        data = np.array([case[0] for case in minibatch])
        labels = np.array([case[1] for case in minibatch])
        return data, labels

    def _save_anet(self, episode, save_folder):
        """Call on 'anet' to save it's topology, weights and optimizer to file."""
        id_str = f'models/set{save_folder}/hex_size{self.board_size}_episode{episode}'
        self.anet.save(id_str)
    
    
class MCTS():
    """Enables the 'Agent' to explore and interact with the 'GameWorld'."""
    def __init__(self, game_world, predictor=None, simulations=10, epsilon=1) -> None:
        print('MCTS instance is created')
        self.game: GameWorld = game_world # Providing all methods related to Hex
        self.N = collections.defaultdict(int) # Visit count for a node
        self.Q = collections.defaultdict(int) # Accumulated rewards for a node
        self.children: dict = {} # Tree structure, providing lookup for node children
        self.simulations: int = simulations # Number of simulated games per move
        self.epsilon = epsilon # Exploration vs exploitation in rollouts 
                         # Fraction of moves being random is 'epsilon'
        self.predictor: Callable = predictor # Predictor method from ANET

    def init_game(self) -> tuple:
        """Initialize a game and return the initial conditions."""
        return self.game.get_initial_state(), False

    def run(self, state: np.ndarray) -> tuple:
        """ Run a specified number of simulations.
        Return the most visited child of current node and if game over."""
        for _ in range(self.simulations):
            self._run_simulation(state)
        child, action_distribution = self._select_most_visited_child(state)
        gameover, winner, _ = self.game.check_if_win(child)
        return child, action_distribution, gameover, winner

    def _select_most_visited_child(self, node: np.ndarray) -> tuple:
        """Return child node with highest visit count.
        Also return the action distribution, for training neural net"""
        node_str = np.array_str(node)
        child_distribution = [self.N[np.array_str(n)] for n in self.children[node_str]]
        child_distribution = np.array(child_distribution)
        child_index = np.argmax(child_distribution)
        child = self.children[node_str][child_index]

        actions = self.game.get_legal_actions(node)
        action_distribution = np.zeros(self.game.get_board_size()**2)
        action_distribution[actions] = child_distribution
        return child, action_distribution

    def _run_simulation(self, root: np.ndarray) -> None:
        """Perform the four stages of a Monte Carlo Tree Search."""
        leaf, path = self._tree_search(root)
        new_leaf, new_path = self._expand(leaf, path)
        reward, _ = self._rollout(new_leaf)
        self._backpropagate(new_path, reward)

    def _tree_search(self, root: np.ndarray) -> tuple:
        """Traverse tree and select a leaf node.
        Return path from root to leaf."""
        node = deepcopy(root)
        node_str = np.array_str(node)
        path = [node_str]
        while node_str in self.children: # while 'node' has children
            node = self._select_uct_child(node) # Select child and move one layer down the tree
            node_str = np.array_str(node)
            path.append(node_str)
        return node, path

    def _select_uct_child(self, parent: np.ndarray) -> np.ndarray:
        """Return child with best UCT score.
        Ensures a good combination of exploitation and exploration."""
        player_one = None
        def uct(child: np.ndarray) -> float: 
            # Upper Confidence bound for Trees
            child = np.array_str(child)
            if self.N[child] == 0: # Avoid division by zero
                score = 1000 if player_one else -1000 # Make sure all nodes are explored at least once
            else:
                estimate = self.Q[child]/self.N[child] 
                bonus = math.sqrt(math.log(self.N[parent])/self.N[child])
                score = estimate + bonus if player_one else estimate - bonus
            return score
        player_one = True if parent[-1]==1 else False
        parent = np.array_str(parent)
        if player_one:
            return max(self.children[parent], key=uct)
        else:
            return min(self.children[parent], key=uct)
        
    def _expand(self, leaf: np.ndarray, path:list) -> tuple:
        """Expand node with all legal children.
        Return new leaf and extended path."""
        gameover, _ , _ = self.game.check_if_win(leaf)
        if not gameover:
            successors = self.game.get_all_successor_states(leaf)
            self.children[np.array_str(leaf)] = successors 
            rand_index = random.randint(0,len(successors)-1)
            leaf = successors[rand_index]
            path.append(np.array_str(leaf))
        return leaf, path

    def _rollout(self, state: np.ndarray) -> tuple:
        """Use an epsilon greedy strategy to play from leaf node until gameover.
        Return the reward. State is returned for testing purposes."""
        gameover, winner, _ = self.game.check_if_win(state)
        while not gameover:
            legal_actions = self.game.get_legal_actions(state)
            if random.uniform(0, 1) < self.epsilon:
                # Fraction of moves being random is 'epsilon' 
                action = legal_actions[random.randint(0,len(legal_actions)-1)]
            else: 
                # Fraction of moves using neural net is 1-'epsilon' 
                action = self._predict_action(state, legal_actions)
            state = self.game.get_successor_state(state, action)
            gameover, winner, _ = self.game.check_if_win(state)
        return self.game.get_reward(winner), state
    
    def _predict_action(self, state, legal_actions):
        predictions = self.predictor(state)
        mask = np.zeros(predictions.shape[0])
        mask[legal_actions] = 1
        predictions = np.multiply(predictions, mask)
        action = np.argmax(predictions)
        if action not in legal_actions: # If distribution is zero for all legal moves
            action = legal_actions[random.randint(0,len(legal_actions)-1)]
        return action

    def _backpropagate(self, path: list, reward: int) -> None:
        """Update visit count and propagate reward for all nodes in path."""
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward


def print_board(state, size=4):
    print('')
    for r in range(size):
        for c in range(size):
            print(f'{int(state[r*size+c]):3} ', end='')
        print('')


if __name__ == '__main__':
    print('***********************************')
    random.seed(a=42)
    board_size= 5
    agent = Agent(board_size=board_size, episodes=201, save_episodes=[0, 50, 100, 150, 200], simulations=500, rbuf_ratio=0.9)
    agent.train(save_folder=2)

    