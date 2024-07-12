# -*- coding: utf-8 -*-
"""
here we define different algorithm for peer selection among the currently deployed pods 
current_deployed_pods: dictionary of all currently deployde pods

"""

import numpy as np

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
    
def best_peer_selection(model, newPod, ticks=False):
    
    '''
    select the best match assuming full information
    '''
    current_pods = model.master.current_deployed_pods
    proper_peers = [k for k in current_pods.keys() if current_pods[k][0] >= newPod.demand[0] \
                  and current_pods[k][1]>=newPod.demand[1]]
   
    if len(proper_peers) > 0:
        
        peer_fitness = {}
        for k in proper_peers: 
            peer_agent = get_agent_by_id(k, model)
            peer_fitness[k] = matching_score(newPod, peer_agent) 
         
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

        return best_peer_id, best_peer

    else:
        return None, None
    
def bottom_up_peer_seletion(model, newPod):
    key = model.master.generate_key(newPod.demand)
    
    if key in model.master.lookup_table and len(model.master.lookup_table[key])>=1:
        best_peer = np.random.choice(model.master.lookup_table[key], size=1, replace=False)[0]
        return best_peer.unique_id, best_peer, key
    else:
        return None, None, None


