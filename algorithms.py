# -*- coding: utf-8 -*-
"""
here we define different algorithm for peer selection among the currently deployed pods 
current_deployed_pods: dictionary of all currently deployde pods

"""

import numpy as np
import pandas as pd

total = 0
count = 0

def get_agent_by_id(agent_id, model):
    for agent in model.schedule.agents:
        if agent.unique_id == agent_id:
            return agent
    return None  # Return None if no agent with the given ID is found

def matching_score(new_pod, peer_pod):
    '''
    

    Parameters
    ----------
    new_pod : Pod Agent
        the new elastic pod that search for a rigid pod
    peer_pod : Pod Agent
        the candidate rigid pod 

    Returns
    -------
    f1 : int 
        distance between the new pod demand vector and peer_pod slack 
    f2 : int
        distance between the new pod demand steps and peer pod remaining steps 
    '''
    M=100 # a large enough number
    if peer_pod is None or new_pod.demand[0] > peer_pod.demand_slack[0] or new_pod.demand[1]>peer_pod.demand_slack[1]:
        return (M,M)
    
    f1 = (peer_pod.demand_slack[0] - new_pod.demand[0]) + (peer_pod.demand_slack[1]-new_pod.demand[1]) 
    # f1 = np.linalg.norm([new_pod.demand[0]-peer_pod.demand_slack[0], \
    #                      new_pod.demand[1]-peer_pod.demand_slack[1]])
    f2 = abs(peer_pod.remain_steps - new_pod.demand_steps)
    return (f1, f2)

def random_peer_selection(model):
    
    '''
    randomly select peer out of peers available in current_deployed_pods dictionary
    input: model 
    '''
    current_pods = list(model.master.current_deployed_pods.keys())
    if len(current_pods) == 0:
        return None, None
    else:
        peer_id = np.random.choice(current_pods, size=1, replace=False)[0]
        peer_agent = get_agent_by_id(peer_id, model)
        return peer_id, peer_agent
    
def best_peer_selection(training_data, model, newPod, ticks=False):
    
    '''
    select the best match assuming full information
    '''
    current_pods = model.master.current_deployed_pods
    proper_peers = current_pods #SH: Just to increase the training dataset
    #proper_peers = [k for k in current_pods.keys() if current_pods[k][0] >= newPod.demand[0] \
    #              and current_pods[k][1]>=newPod.demand[1]]
   
    if len(proper_peers) > 0:
        
        peer_fitness = {}
        for k in proper_peers: 
            peer_agent = get_agent_by_id(k, model)
            peer_fitness[k] = matching_score(newPod, peer_agent) 

            # SH: training data logging
            training_data.append({
                "new_pod_demand_cpu": newPod.demand[0],
                "new_pod_demand_mem": newPod.demand[1],
                "peer_pod_slack_cpu": peer_agent.demand_slack[0],
                "peer_pod_slack_mem": peer_agent.demand_slack[1],
                "new_pod_demand_steps": newPod.demand_steps,
                "peer_pod_remain_steps": peer_agent.remain_steps,
                "f1": peer_fitness[k][0],
                "f2": peer_fitness[k][1]
            }) 
         
        if ticks:
            best_peer_id = min(peer_fitness, key=lambda k: (peer_fitness[k][0], peer_fitness[k][1]))
        else:
            best_peer_id = min(peer_fitness, key=lambda k: peer_fitness[k][0])

        best_peer = get_agent_by_id(best_peer_id, model)
         
        #print('the demand is', newPod.demand, 'for demand steps', newPod.demand_steps)
        #print('the best is..', best_peer_id, peer_fitness[best_peer_id],'the all are...', peer_fitness)
        
        #print('time diff...', 'demand is...', newPod.demand_steps, 'offer is ...',best_peer_agent.remain_steps)
        #print('-------------------------------')
        #print(peer_fitness, 'best', current_deployed_pods, best_peer, peer_fitness[best_peer])
         
        # print('#peers=', len(current_deployed_pods), '#proper peers=',\
        #       len(proper_peers),', new-pod demans = ',newPod.demand,
        #       ', selected peer=(',peer_cpu_resrc,',',peer_mem_resrc,')')

        return best_peer_id, best_peer, training_data

    else:
        return None, None, training_data
    
def bottom_up_peer_seletion(model, newPod):
    key = model.master.generate_key(newPod.demand)
    
    if key in model.master.lookup_table and len(model.master.lookup_table[key])>=1:
        best_peer = np.random.choice(model.master.lookup_table[key], size=1, replace=False)[0]
        return best_peer.unique_id, best_peer, key
    else:
        return None, None, None

def nn_peer_selection(model, new_pod, neural_net_model, scaler, ticks=False):
    """
    Uses a neural network model to select the best peer pod for a new pod.

    Parameters
    ----------
    model : Model instance
        The main model containing all deployed pods and other agents.
    new_pod : Pod Agent
        The new elastic pod searching for a peer pod.
    neural_net_model : trained neural network model
        The trained model to predict the most suitable peer pod, trained before, saved using joblit and passed here
    scaler : StandardScaler here, could also be MinMaxScaler
        Scaler used to normalize input data.
    ticks : bool, optional
        Whether to consider both f1 and f2 in peer selection (default is False).

    Returns
    -------
    best_peer_id : int or None
        The ID of the best peer pod, if one is found.
    best_peer : Agent or None
        The Agent instance of the best peer pod, if one is found.
    """

    global total
    global count
    total = total+1
    # Gather the current peer pods in the model
    current_pods = list(model.master.current_deployed_pods.keys())

    if not current_pods:
        print("No available peers to assign.")
        return None, None

    # Initialize variables to track the best peer
    best_peer_id = None
    peer_fitness = {}

    # Define feature names based on your original DataFrame
    feature_names = [
        'new_pod_demand_cpu', 'new_pod_demand_mem', 
        'peer_pod_slack_cpu', 'peer_pod_slack_mem', 
        'new_pod_demand_steps', 'peer_pod_remain_steps'
    ]

    # Loop through each peer pod and predict compatibility
    for peer_id in current_pods:
        peer_agent = get_agent_by_id(peer_id, model)

        # Prepare the input data for the neural network
        input_data = [[
            new_pod.demand[0], new_pod.demand[1],             # new pod's CPU and memory demand
            peer_agent.demand_slack[0], peer_agent.demand_slack[1],  # peer pod's CPU and memory slack
            new_pod.demand_steps, peer_agent.remain_steps     # new pod's demand steps and peer's remaining steps
        ]]

        # Convert input_data to DataFrame with the appropriate column names
        input_data_df = pd.DataFrame(input_data, columns=feature_names)

        # Normalize the data
        input_data_normalized = scaler.transform(input_data_df)

        # Predict the matching score (f1, f2) using the neural network
        try:
            peer_fitness[peer_id] = neural_net_model.predict(input_data_normalized)[0]
        except Exception as e:
            print(f"Error predicting fitness for peer {peer_id}: {e}")
            continue

    if ticks:
        best_peer_id = min(peer_fitness, key=lambda k: (peer_fitness[k][0], peer_fitness[k][1]))
    else:
        best_peer_id = min(peer_fitness, key=lambda k: peer_fitness[k][0])

    best_peer = get_agent_by_id(best_peer_id, model) if best_peer_id else None
        
    """ if best_peer is not None:
        if best_peer.demand_slack[0] < new_pod.demand[0] or \
           best_peer.demand_slack[1] < new_pod.demand[1]:
            print("Warning: Assigned peer_pod does not have enough resources to satisfy new_pod requirements.")
            count = count + 1
    print ("Total and count are: ", total, "\t", count) """
    return best_peer_id, best_peer
