import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_custom_taxi_env import SimpleTaxiEnv
ACTION_SIZE = 6
# state[NS_relative_dir,WE_relative_dir,picked_up,N_obstacle,S_obstacle,E_obstacle,W_obstacle,current_station,destination_station]
# relative_dir = 0 if coordination of the object is larger than the taxi's coordination
# relative_dir = 1 if coordination of the object is equal to the taxi's coordination
# relative_dir = 2 if coordination of the object is smaller than the taxi's coordination
# picked_up = 1 if the passenger is picked up
# picked_up = 0 if the passenger is not picked up
# we visit station by order, current_station is my current goal
# destionation_station is the index of destination
q_table = np.zeros((3,3,2,2,2,2,2,4,5,ACTION_SIZE))
def choose_action(state_indices, epsilon) -> int:
    """
    Epsilon-greedy action selection: with probability epsilon choose a random action,
    otherwise choose the action with the highest Q-value for the current state.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(ACTION_SIZE)
    else:
        return int(np.argmax(q_table[state_indices]))
def get_state_indices(obs, old_state, pre_action, pas_pos) -> tuple:
    """
    Convert the current environment state into a 9-tuple of indices for the Q-table.
    
    Expected state components:
      - NS_relative_dir: relative North-South direction to target
      - WE_relative_dir: relative West-East direction to target
      - picked_up: 0 if passenger not picked up, 1 if picked up
      - N_obstacle, S_obstacle, E_obstacle, W_obstacle: 0/1 obstacles (or boundaries)
      - current_station: index (0-3) of the current target station. If not picked up, it's the passenger's station.
      - destination_station: index (0-3) of the destination station (we use a 5-sized dimension to allow an extra value, if needed)
    """
    # update the current station
    NS_relative_dir = old_state[0]
    WE_relative_dir = old_state[1]
    destionation_station = old_state[7]
    current_station = old_state[6]
    picked_up = old_state[2]
    if obs[2+2*old_state[6]] == obs[0] and obs[3+2*old_state[6]] == obs[1]:
        if obs[15] == 1:# this station is the destination
            destionation_station = old_state[6]

        if not (current_station == destionation_station and old_state[2] == 1):
            current_station = (current_station + 1)%4
    if destionation_station != 4 and picked_up == 1:
        current_station = destionation_station 
    # update the picked_up status
    condition = ((obs[0] == obs[2] and obs[1] == obs[3]) or (obs[0] == obs[4] and obs[1] == obs[5]) or(obs[0] == obs[6] and obs[1] == obs[7]) or 
    (obs[0] == obs[8] and obs[1] == obs[9]))
    if picked_up == 0 and obs[14] == 1 and condition and pre_action == 4 and pas_pos == (-1,-1):
        picked_up = 1
    elif (((obs[0] == pas_pos[0] and obs[1] == pas_pos[1])) and pre_action == 4):  
        picked_up = 1

    if picked_up == 1 and pre_action == 5:
        picked_up = 0
    # update the relative direction
    if obs[0] > obs[2+2*current_station]:
        NS_relative_dir = 2
    elif obs[0] == obs[2+2*current_station]:
        NS_relative_dir = 1
    else:
        NS_relative_dir = 0
    if obs[1] > obs[3+2*current_station]:
        WE_relative_dir = 2
    elif obs[1] == obs[3+2*current_station]:
        WE_relative_dir = 1
    else:
        WE_relative_dir = 0
    if picked_up == 0 and pas_pos != (-1,-1) and pas_pos != (-2,-2):
        NS_relative_dir = 2 if obs[0] > pas_pos[0] else 1 if obs[0] == pas_pos[0] else 0
        WE_relative_dir = 2 if obs[1] > pas_pos[1] else 1 if obs[1] == pas_pos[1] else 0
    state = (NS_relative_dir,WE_relative_dir,picked_up,obs[10],obs[11],obs[12],obs[13],current_station,destionation_station)
    return state
def train_q_table(episodes = 100000, alpha = 0.015, gamma = 0.99,
                  epsilon_start = 1, epsilon_end = 0) -> np.ndarray:
    """
    Train the Q-table using Q-learning.
    """
    rewards_per_episode = []
    epsilon = epsilon_start
    sucess = []
    passger = []
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        # Reset the environment and get the initial state.
        env = SimpleTaxiEnv(grid_size=random.randint(5,10), fuel_limit=500)
        obs,_ = env.reset()
        total_reward = 0
        done = False
        pre_action = -1
        old_state = (-1,-1,0,obs[10],obs[11],obs[12],obs[13],0,4)
        pas_pos = (-1,-1) 
        First_time = False
        temp = np.sum(q_table)
        while not done:
            # Get current state indices from the environment.
            temp = np.sum(q_table)
            state_idx = get_state_indices(obs,old_state,pre_action,pas_pos)
            # Select an action using epsilon-greedy.
            action = choose_action(state_idx, epsilon)
            pre_action = action
            # Take the action in the environment.
            obs, reward, done, _ = env.step(action)
            if old_state[2] == 1 and pre_action == 5:
                pas_pos = (obs[0],obs[1])
            if old_state[2] == 0 and state_idx[2] == 1:
                pas_pos = (-2,-2) # the passenger is picked up
            old_state = state_idx
            # Get next state indices.
            next_state_idx = get_state_indices(obs,old_state,pre_action,pas_pos)
            # Q-learning update:
            if env.current_fuel > 0 and done:
                reward += 15000
            best_next = np.max(q_table[next_state_idx])
            Flag = not First_time and next_state_idx[2] == 1
            if not done and action == 5:
                reward -= 100
            if Flag:
                reward += 10000
                First_time = True
            q_table[state_idx+(action,)] += alpha * (reward + gamma * best_next - q_table[state_idx+(action,)])
            if env.current_fuel > 0 and done:
                reward -= 15000
            if Flag:
                reward -= 10000
            if not done and action == 5:
                reward += 100
            total_reward += reward
        rewards_per_episode.append(total_reward)
        if env.current_fuel != 0:
            sucess.append(1)
        else:
            sucess.append(0)
        if First_time:
            passger.append(1)
        else:
            passger.append(0)
        if episode%20 == 0:
            #print the average reward of the last 10000 episodes
            print("")
            print(np.mean(rewards_per_episode[-200:]))
            print(np.mean(passger[-200:]))
            print(np.mean(sucess[-200:]))
            print(epsilon)
        # Linearly decay epsilon over episodes.
        epsilon = max(epsilon - (epsilon_start - epsilon_end) / episodes, epsilon_end)
        
    # Plot the moving average of rewards.
    #plot the average reward of every 200 episodes
    rewards_per_episode = np.array(rewards_per_episode)
    rewards_per_episode = rewards_per_episode.reshape(-1, 200)
    rewards_per_episode = np.mean(rewards_per_episode, axis=1)
    
    passger = np.array(passger)
    passger = passger.reshape(-1, 200)
    passger = np.mean(passger, axis=1)
    plt.figure()
    plt.plot(passger)
    plt.xlabel("Episode Block (200 episodes per block)")
    plt.ylabel("Passenger Success Rate")
    plt.title("Passenger Success Rate")
    #plt.savefig("newmyq_learning_passenger.png")
    plt.show()

    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    #plt.savefig("q_learning_training.png")
    plt.show()
    
    sucess_arr = np.array(sucess).reshape(-1, 200)
    sucess_moving_avg = np.mean(sucess_arr, axis=1)
    plt.figure()
    plt.plot(sucess_moving_avg)
    plt.xlabel("Episode Block (200 episodes per block)")
    plt.ylabel("Success Rate")
    plt.title("newmyTraining Success Rate")
    #plt.savefig("newmyq_learning_success.png")
    plt.show()
    return q_table
if __name__ == "__main__":
    trained_q_table = train_q_table(episodes=8000000)
    # Save the updated Q-table.
    with open("my_q_table.pkl", "wb") as f:
        pickle.dump(trained_q_table, f)