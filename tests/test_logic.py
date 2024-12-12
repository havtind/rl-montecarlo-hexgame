import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import random
import tensorflow as tf


from mcts_hex.mcts import GameWorld, MCTS, Agent
from mcts_hex.anet import ANET

  
def print_board(state, size=4):
    print('')
    for r in range(size):
        for c in range(size):
            if state[r*size+c]==0:
                print('0 ', end='')
            elif state[r*size+c]==1:
                print('1 ', end='')
            elif state[r*size+c]==2:
                print('2 ', end='')
        print('')




class TestGameWorld(unittest.TestCase):
    def test_initial_state(self):
        game = GameWorld(2)
        np.testing.assert_array_equal(game.get_initial_state(),np.array([0,0,0,0,1]))

    def test_successor_state(self):
        game = GameWorld(2)
        state = np.array([0,0,0,0,1])
        new_state = game.get_successor_state(state, 0)
        np.testing.assert_array_equal(state,np.array([0,0,0,0,1]))
        np.testing.assert_array_equal(new_state,np.array([1,0,0,0,2]))

    def test_get_all_successors(self):
        game = GameWorld(2)
        state = np.array([0,0,0,0,1])
        successors_true = np.array([[1,0,0,0,2],[0,1,0,0,2],[0,0,1,0,2],[0,0,0,1,2]])
        successors = game.get_all_successor_states(state)
        np.testing.assert_array_equal(successors_true, successors)

        state = np.array([1,1,2,2,1])
        succ = game.get_all_successor_states(state)
        self.assertEqual(succ.size, 0)

    def test_check_if_win(self):
        game = GameWorld(2)
        state = np.array([1,1,2,2,1])
        self.assertEqual(game.check_if_win(state)[0], True)
        self.assertEqual(game.check_if_win(state)[1], 2)

        state = np.array([1,2,2,1,1]) 
        self.assertEqual(game.check_if_win(state)[0], True)
        self.assertEqual(game.check_if_win(state)[1], 2)


        state = np.array([1,2,1,0,2])
        self.assertEqual(game.check_if_win(state)[0], True)
        self.assertEqual(game.check_if_win(state)[1], 1)
    
        
        game = GameWorld(4)
        state= np.array([1,1,1,2,1,2,1,2,1,2,2,2,1,2,2,1,2])
        self.assertEqual(game.check_if_win(state)[0], True)
        self.assertEqual(game.check_if_win(state)[1], 1)

        state= np.array([1,1,1,1,2,2,2,0,2,2,2,1,1,1,2,1,1])
        self.assertEqual(game.check_if_win(state)[0], False)

        state= np.array([1,1,1,1,2,2,2,2,2,2,2,1,1,1,2,1,1])
        self.assertEqual(game.check_if_win(state)[0], True)
        self.assertEqual(game.check_if_win(state)[1], 2)

class TestMCTS(unittest.TestCase):
    def test_select_most_visited_child(self):
        game = GameWorld(board_size=2)
        mcts = MCTS(game_world=game)
        state = np.array([0,0,0,0,1])
        state_str = np.array_str(state)
        successors = np.array([[1,0,0,0,2],[0,1,0,0,2],[0,0,1,0,2],[0,0,0,1,2]])
        mcts.children = {state_str: successors} 

        child = np.array([1,0,0,0,2])
        child_str = np.array_str(child)
        mcts.N[child_str] = 4
        selected_child, _ = mcts._select_most_visited_child(state)
        np.testing.assert_array_equal(child, selected_child)


    def test_expand_initial_state(self):
        game = GameWorld(board_size=2)
        mcts = MCTS(game_world=game)
        state = np.array([0,0,0,0,1])
        state_str = np.array_str(state)
        successors = np.array([[1,0,0,0,2],[0,1,0,0,2],[0,0,1,0,2],[0,0,0,1,2]])    
        path = [state_str]

        new_state, new_path = mcts._expand(state, path)
        np.testing.assert_array_equal(successors, mcts.children[state_str])

        new_state_str = np.array_str(new_state)
        self.assertEqual(new_state_str, path[-1])
    
    def test_expand_terminal_state(self):
        game = GameWorld(board_size=2)
        mcts = MCTS(game_world=game)

        state = np.array([1,2,2,1,1])
        state_str = np.array_str(state)
        path = [state_str]
        new_state, new_path = mcts._expand(state, path)
        np.testing.assert_array_equal(state, new_state)
        self.assertEqual(path, new_path)

        state = np.array([1,2,1,0,2])
        state_str = np.array_str(state)
        path = [state_str]
        new_state, new_path = mcts._expand(state, path)
        np.testing.assert_array_equal(state, new_state)
        self.assertEqual(path, new_path)
 
    def test_rollout_terminal_state(self):
        game = GameWorld(board_size=2)
        mcts = MCTS(game_world=game)

        state = np.array([1,2,2,1,1]) # winner is 2
        reward, new_state = mcts._rollout(state)
        self.assertEqual(reward, -1)
        np.testing.assert_array_equal(state, new_state)

        state = np.array([1,2,1,0,2]) # winner is 1
        reward, new_state = mcts._rollout(state)
        self.assertEqual(reward, 1)
        np.testing.assert_array_equal(state, new_state)
    
    def test_rollout_initial_state(self):
        size = 5
        game = GameWorld(board_size=size)
        anet = ANET(input_length=size**2+1, output_length=size**2, epochs=10, lr=0.08) 
        mcts = MCTS(game_world=game, predictor=anet.get_predictions)

        state = game.get_initial_state()
        reward, state = mcts._rollout(state)
        
        self.assertTrue(reward == 1 or reward == -1)
        self.assertEqual(game.check_if_win(state)[0], True)
       
    def test_backpropagate(self):
        game = GameWorld(board_size=5)
        mcts = MCTS(game_world=game)

        state = np.array([0,0,0,0,1])
        state_str = np.array_str(state)
        path = [state_str]
        reward = -1
        mcts._backpropagate(path, reward)
        self.assertEqual(mcts.Q[state_str], -1)
        self.assertEqual(mcts.N[state_str], 1)
       
    def test_run(self):
        board_size= 4
        simulations = 10
        game = GameWorld(board_size=board_size)

        mcts = MCTS(game_world=game, simulations=simulations, epsilon=1)
        state, gameover = mcts.init_game()
        while not gameover: 
            state, distribution, gameover, winner = mcts.run(state)
        self.assertEqual(len(distribution), board_size**2)
        self.assertEqual(game.check_if_win(state)[0], gameover)
        self.assertEqual(game.check_if_win(state)[1], winner)

        mcts = MCTS(game_world=game, simulations=simulations, epsilon=1)
        state, gameover = mcts.init_game()
        state, distribution, gameover, winner = mcts.run(state)
        state_str = np.array_str(state)
        self.assertEqual(sum(distribution), simulations)
        self.assertEqual(mcts.N[state_str],max(distribution))


class TestAgent(unittest.TestCase):
    def test_random_agent(self):
        board_size= 4
        agent = Agent(board_size=board_size, episodes=2, simulations=10, rbuf_ratio=1)
        agent.train()

        data, labels = agent._get_training_cases()
        self.assertEqual(data.shape[1], board_size**2+1)
        self.assertEqual(labels.shape[1], board_size**2)
        
class TestANET(unittest.TestCase):
    def test_anet(self):
        board_size = 3
        input_size = board_size**2+1
        output_size = board_size**2

        #data = np.random.randint(0,2,(samples, input_size))
        #labels = np.random.randint(0,10,(samples, output_size))
        state = np.array([0,1,2,0,1,2,0,0,1,2])
        actions = np.array([0,4,1,5,2,0,0,2,0])
        data = np.array([state,state,state,state])
        labels = np.array([actions, actions, actions, actions])

        anet = ANET(input_length=input_size, output_length=output_size, epochs=1, lr=0.8)
        anet.update(data, labels)

        state = np.array([0,1,2,0,1,2,0,0,0,2])
        pred = anet.get_predictions(state)
        self.assertEqual(pred.shape[0],output_size)



if __name__ == '__main__':
    print('***********************************')
    random.seed(a=42)
    unittest.main()
    
    
    
   


