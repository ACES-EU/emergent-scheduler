# -*- coding: utf-8 -*-
"""
This module defines the pod profiles and some functions to get a pod with specific profile
"""

import numpy as np
import random

'''
get_small_pod, get_medium_pod, get_large_pod
profiles: define [(cpu_demand, mem_demand), (cpu_slack, mem_slack)]
demand_steps: list of required steps to execute the pod, uniform distribution
'''
def get_small_pod(demand_steps=(20,30)): 
    
    # considered small pod profiles
    small_profiles = [[(1,1),(0,0)], [(1,2),(0,0)], [(2,1),(0,0)], [(2,2),(0,0)]]
    
    prob1 = 1./len(small_profiles)*np.ones(len(small_profiles))
    demand, slack = random.choices(small_profiles, weights=prob1, k=1)[0]
    
    demand_step = random.randint(demand_steps[0], demand_steps[1])
    return demand, demand_step, slack

def get_medium_pod(demand_steps=(50,130)):  

    # considered medium pod profiles
    medium_profiles = [[(4,4),(1,1)], [(4,4),(1,2)], [(4,4),(2,1)],\
                      [(4,6),(1,2)], [(4,6),(2,1)], [(4,6),(2,2)],\
                      [(6,4),(1,2)], [(6,4),(2,1)], [(6,4),(2,2)],\
                      [(6,6),(1,2)], [(6,6),(2,1)], [(6,6),(2,2)]]
        
    # medium_profiles = [[(4,4),(0,0)], [(4,4),(0,0)], [(4,4),(0,0)],\
    #                       [(4,6),(0,0)], [(4,6),(0,0)], [(4,6),(0,0)],\
    #                       [(6,4),(0,0)], [(6,4),(0,0)], [(6,4),(0,0)],\
    #                       [(6,6),(0,0)], [(6,6),(0,0)], [(6,6),(0,0)]]

    
    prob1 = 1./len(medium_profiles)*np.ones(len(medium_profiles))
    
    demand, slack = random.choices(medium_profiles, weights=prob1, k=1)[0]
    
    demand_step = random.randint(demand_steps[0], demand_steps[1])
    return demand, demand_step, slack

def get_large_pod(demand_steps=(200,400)): 
    # considered large pod profiles
    large_profiles = [[(8,8),(4,4)],\
                      [(8,16),(4,4)], [(8,16),(4,6)],\
                      [(16,8),(4,4)], [(16,8),(6,4)],\
                      [(16,16),(4,6)], [(16,16),(6,4)], [(16,16),(6,6)]]
    
   
    # large_profiles = [[(8,8),(0,0)],\
    #                   [(8,16),(0,0)], [(8,16),(0,0)],\
    #                   [(16,8),(0,0)], [(16,8),(0,0)],\
    #                   [(16,16),(0,0)], [(16,16),(0,0)], [(16,16),(0,0)]]

        
    prob1 = 1./len(large_profiles)*np.ones(len(large_profiles))
    demand, slack = random.choices(large_profiles, weights=prob1, k=1)[0]
    
    demand_step = random.randint(demand_steps[0], demand_steps[1])

    return demand, demand_step, slack


def get_pod_profile(categories_prob = (0.4, 0.4, 0.2), prob_elastisity=0.5):
    
    '''
    get the pod profile:
        pod_categories = ('s', 'm', 'l'), small, medium, large
        categories_prob = (0.4, 0.4, 0.2): corresponds to small , medium, large pods
        prob_elasticity: the prob. that pod is elastic
    '''

    rnd = random.random()
    if rnd <categories_prob[0]:
        # small demand
        pod_demand, pod_demand_step, pod_slack = get_small_pod(demand_steps=(60,120)) #(60,120)
        pod_is_elastic = True if random.random() < prob_elastisity else False
        
    elif categories_prob[0] <= rnd < categories_prob[0]+categories_prob[1]:
        # medium pod
        pod_demand, pod_demand_step, pod_slack = get_medium_pod(demand_steps=(100,200)) #(100,200)
        pod_is_elastic = True if random.random() < prob_elastisity else False
        
    else:
        #large pod
        pod_demand, pod_demand_step, pod_slack = get_large_pod(demand_steps=(150,300)) #(150,300)
        pod_is_elastic = False #True if random.random() < prob_elastisity else False
    
    if not pod_is_elastic:
        pod_demand_tolerance = int(0.1*pod_demand_step)
    else:
        pod_demand_tolerance = int(0.3*pod_demand_step) 
        pod_slack = [0,0]
        
    return pod_demand, pod_demand_step, pod_slack, pod_is_elastic, pod_demand_tolerance


  
def get_pod_demand(self, cpu_demand=(1,2), mem_demand=(1,2), demand_steps=(5,10), prob=(0.5,0.5)):
    cpu = random.choices(cpu_demand, weights=prob, k=1)[0]
    mem = random.choices(mem_demand, weights=prob, k=1)[0]
    demand_step = random.choices(demand_steps, weights=prob, k=1)[0]
    
    return (cpu, mem), demand_step
