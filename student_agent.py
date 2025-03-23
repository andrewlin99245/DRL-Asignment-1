import numpy as np
import pickle
import random
import gym

last_state = (-1,-1,0,0,0,0,0,0,4)
last_action = -1
pas_pos = (-1,-1)
with open("my_q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

def get_action(obs):
    global Q_table
    global pas_pos
    global last_state
    global last_action
    state_indices = get_state_indices(obs, last_state, last_action, pas_pos)
    action = int(np.argmax(Q_table[state_indices]))
    if state_indices[2] == 1 and action == 5:
        pas_pos = (obs[0],obs[1])
    if state_indices[2] == 0 and pas_pos == (obs[0],obs[1]) and action == 4:
        pas_pos = (-2,-2) # the passenger is picked up
    last_state = state_indices
    last_action
    return action

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
    