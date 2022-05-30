from array import array
from locale import normalize
import torch
import numpy as np



load_maze()
Maze=Environment()


def train(num_episodes):
  
  ##### Training
  while Maze.episode<num_episodes:

      Maze.action(is_train=True, render=False)
      
      epsilon = epsilon * epsilon_decay
  print( Maze.reward_history)
  # Plot total rewards for each episode
  fig_2 = plt.figure(10)
  ax_2 = fig_2.gca()
  ax_2.plot(np.arange(1, num_episodes), Maze.reward_history,  color='green')
  ax_2.set_title('Total rewards plot', fontsize=14)
  ax_2.set_xlabel('episode')
  ax_2.set_ylabel('Total reward')
  ax_2.grid()
  ax_2.set_xticks(range(1, num_episodes, 1))
  fig_2.savefig('total_rewards_plot.png')
  plt.clf() 

  # Plot walls hit for each episode
  print(Maze.wall_count_history)
  fig_3 = plt.figure(10)
  ax_3 = fig_3.gca()
  ax_3.plot(np.arange(1, num_episodes), Maze.wall_count_history,  color='red')
  ax_3.set_title('Total walls hit plot', fontsize=14)
  ax_3.set_xlabel('episode')
  ax_3.set_ylabel('Total Walls Hit')
  ax_3.grid()
  ax_3.set_xticks(range(1, num_episodes, 1))
  fig_3.savefig('total_walls_hit_plot.png')
  plt.clf() 
  
  print(Maze.visited_count_history)

  # Plot visited states for each episode
  fig_4 = plt.figure(10)
  ax_4 = fig_4.gca()
  ax_4.plot(np.arange(1, num_episodes), Maze.visited_count_history,  color='orange')
  ax_4.set_title('Total Visited States plot', fontsize=14)
  ax_4.set_xlabel('episode')
  ax_4.set_ylabel('Total Visited States')
  ax_4.grid()
  ax_4.set_xticks(range(1, num_episodes, 1))
  fig_4.savefig('total_visited_states_plot.png')
  plt.clf() 


  # Plot fire states for each episode
  fig_4 = plt.figure(10)
  ax_4 = fig_4.gca()
  ax_4.plot(np.arange(1, num_episodes), Maze.fire_count_history,  color='red')
  ax_4.set_title('Total Fire States plot', fontsize=14)
  ax_4.set_xlabel('episode')
  ax_4.set_ylabel('Total Fire States')
  ax_4.grid()
  ax_4.set_xticks(range(1, num_episodes, 1))
  fig_4.savefig('total_fire_states_plot.png')
  plt.clf() 
  
  torch.save(Maze.qlearning.q_table, 'q_table.pt')

def test(num_episodes):
  if(os.path.exists('q_table.pt')):
    Maze.qlearning.q_table = torch.load('q_table.pt')
    #Maze.q_values = torch.load('q_table.pt')
    # Evaluate
    print("Evaluating Q Agent.")
    # Create .txt output file
    f = open('output_eval.txt', 'w')
    f.write('Dynamic maze solving algorithm - output file \n')
    f.close()
    while Maze.episode<num_episodes:
        Maze.action(is_train=False, render=False)
   
  

train(6)
##### Evaluation

