'''
ACES Project: swarm inspired bottom-up resource allocation 

The edge computing system is modeled as multiagent system consists of
one master agent, one worker agent, and sequentially arriving
pod agents.

Each agent has its own attributes and behave accordingly in each consecutive step.

The system model first creates the master and worker agents and adds them. 
The scheduler calls the step function of all agents and creates and adds a new pod
to the system for a specific number of steps.
The inter-arrival time steps are generated using an exponential random variable with parameter λ.
The model scheduler tracks the satisfied and unsatisfied pods and removes the completed pods from the system.


Original Author: Abdorasoul Ghasemi, arghasemi@gmail.com
Modified by: Atta Ur Rahman, rahman@lakeside-labs.com
''' 

# Mesa is used for multiagent simulation
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid

# used packages
import numpy as np
import random
import matplotlib.pyplot as plt

# added modules  
import pod_profiles
import algorithms
import visualization

import yaml


class Pod(Agent):
    
    def __init__(self, unique_id, model, demand, demand_steps=1,
                 is_elastic=False, demand_tolerate_steps=2, demand_slack=[0,0]):
        
        super().__init__(unique_id, model)
        self.demand = demand
        self.demand_steps = demand_steps
        self.demand_tolerate_steps = demand_tolerate_steps
        self.demand_slack = demand_slack
        self.is_elastic = is_elastic
        
        self.remain_steps = demand_steps
        self.assigned_cpu = 0
        self.assigned_mem = 0 
        self.assigned_worker = None
        self.arrival_step = None
       
    def step(self):

        if self.assigned_worker is None:
            # nothing to do in this step, wait in the corresponding master queue
            pass
        
        elif self.remain_steps > 0:
            # pod is deployed, needs more steps 
            self.remain_steps -= 1
               
        else:
            # pod completed, track some metrics 
            expected_departure = self.arrival_step + self.demand_steps + self.demand_tolerate_steps

            if self.model.schedule.steps <= expected_departure:
                if self.is_elastic:
                    self.model.satisfied_elastic.append((self.demand, self.demand_steps))
                else:
                    self.model.satisfied_rigid.append((self.demand, self.demand_steps))
                    
            else:
                if self.is_elastic:
                    self.model.un_satisfied_elastic.append((self.demand, self.demand_steps))
                else:
                    self.model.un_satisfied_rigid.append((self.demand, self.demand_steps))
                    
            # delete from the current_deployed_pods and release resources 
            del self.model.master.current_deployed_pods[self.unique_id]
            self.model.master.del_lookup_table(self)
            self.model.master.release_resources(self)

class Worker(Agent):
    
   
    def __init__(self, unique_id, model, resource_capacity=(0,0)):
        
        super().__init__(unique_id, model)
        
        # vector of resource capacities, for now (cpu_cap, mem_cap) 
        self.resource_capacity = resource_capacity 
        
        self.current_cpu_assignment = 0
        self.current_mem_assignment = 0
        self.current_cpu_utilization = 0
        self.current_mem_utilization = 0
        # track the utilization of worker over time
        self.cpu_utilization = [] 
        self.mem_utilization = []
        
    def get_cpu_utilization(self):
                
        return self.current_cpu_utilization / self.resource_capacity[0]
    
    def get_mem_utilization(self):
                
        return self.current_mem_utilization / self.resource_capacity[1]

    def accept_as_rigid(self, pod):
        
        '''
        asign worker, update parameters, and return True if pod is accepted,
        otherwise return False
        '''
        
        if self.current_cpu_assignment + pod.demand[0] <= self.resource_capacity[0] and \
            self.current_mem_assignment + pod.demand[1] <= self.resource_capacity[1]:
               
                pod.assigned_worker = self         # set the worker for this pod 
                pod.assigned_cpu = pod.demand[0]
                pod.assigned_mem = pod.demand[1]
                
                # update resource assignment and utilization  
                self.current_cpu_assignment += pod.demand[0] 
                self.current_mem_assignment += pod.demand[1]
                self.current_cpu_utilization += pod.demand[0] - pod.demand_slack[0]
                self.current_mem_utilization += pod.demand[1] - pod.demand_slack[1]
                    
                # placement in grid
                xpos = round(pod.demand_steps*pod.demand_slack[0])
                ypos = round(pod.demand_steps*pod.demand_slack[1])
                while self.model.grid.is_cell_empty((xpos, ypos)) is False:
                        xpos += random.randint(-5,5)
                        ypos += random.randint(-5,5)
                self.model.grid.place_agent(pod, (xpos, ypos))

                # print("rigid pod is hosted by a worker")
                
                return True
        else:
            # print("rigid pod is not hosted by a worker")
            return False
     
    
    def accept_as_elastic(self, pod):
        
        '''
        Try to find a proper rigid pod as a host of this elastic pod 
        If the selected rigid pod has sufficient slack resources to meet the elastic pod's demand return True
        otherwise retuen False
        '''
       
        # see algorithms for implemented methods
        if self.model.method == 'RND':
            peer_id, peer_pod = algorithms.random_peer_selection(self.model)
        elif self.model.method == 'BEST':
            peer_id, peer_pod = algorithms.best_peer_selection(self.model, pod, ticks=True)
        elif self.model.method == 'SWARM':
            peer_id, peer_pod, best_key = algorithms.bottom_up_peer_seletion(self.model, pod)
            
        else:
            print('Method is not implemented')
            return False
        
        if peer_id is None:
            # algorithm does not find any peer
            return False
        
        peer_cpu_slack = peer_pod.demand_slack[0]
        peer_mem_slack = peer_pod.demand_slack[1]
        if pod.demand[0] <= peer_cpu_slack and pod.demand[1] <= peer_mem_slack:
              
            pod.assigned_worker = self  
            pod.assigned_cpu = pod.demand[0] #peer_cpu_slack
            pod.assigned_mem = pod.demand[1] #peer_mem_slack
            
            
            # update parameters: increase utilization 
            self.current_cpu_utilization += pod.demand[0] #peer_cpu  
            self.current_mem_utilization += pod.demand[1] #peer_mem
            
            # For visualization: placement in grid
            xpos = round(peer_pod.demand_steps*peer_pod.demand_slack[0])
            ypos = round(peer_pod.demand_steps*peer_pod.demand_slack[1])
            while self.model.grid.is_cell_empty((xpos, ypos)) is False:
                   
                   # place in a random cell if the selected one is not empty
                   xpos += random.randint(-5,5)
                   ypos += random.randint(-5,5)
            self.model.grid.place_agent(pod, (xpos, ypos))
            
            peer_pod.assigned_cpu -= pod.demand[0] #peer_cpu_slack
            peer_pod.assigned_mem -= pod.demand[1] #peer_mem_slack
            peer_pod.demand_slack[0] -= pod.demand[0]
            peer_pod.demand_slack[1] -= pod.demand[1]
            
            # no more elastic pod can exploit this
            self.model.master.current_deployed_pods[peer_id] = (0,0)

            # print("elastic pod is hosted by a rigid pod")

            return True
        else:

            # print("elastic pod is not hosted by a rigid pod")

            return False
           
    def release_resources(self, pod):
        # release the assigned cpu/mem resources
        self.current_cpu_assignment -= pod.assigned_cpu
        self.current_mem_assignment -= pod.assigned_mem
        
        if not pod.is_elastic:
            self.current_cpu_utilization -= pod.assigned_cpu - pod.demand_slack[0]
            self.current_mem_utilization -= pod.assigned_mem - pod.demand_slack[1]
        else:
            self.current_cpu_utilization -= pod.demand[0]
            self.current_mem_utilization -= pod.demand[1]
            
            
    def step(self):
        # update worker utilization
        self.cpu_utilization.append(self.get_cpu_utilization())
        self.mem_utilization.append(self.get_mem_utilization())

class Master(Agent):
    
    
    def __init__(self, unique_id, model, thresholds, Gamma, slack_estimation_error=0., worker_list=[]):
        
        super().__init__(unique_id, model)
        # print("config file read, the values of threshold and gamma are: ", thresholds, Gamma)
        
        self.worker_list = worker_list
        self.rigid_queue = []
        self.elastic_queue = []

        # track the queue status and current deployde pods 
        self.rigid_queue_status = [] 
        self.elastic_queue_status = [] 

        self.current_deployed_pods = {}
        
        self.lookup_table = {}
        # thresholds used to cluster current deployed rigid pods based on their slacks
        self.thresholds= thresholds  
        # probability that an elastic pod is served as a rigid one  
        self.Gamma = Gamma 
        self.slack_estimation_error = slack_estimation_error


        ### ADDED num_elasticasRigid, num_rigid, num_elastic and pod_ignored variables ##
        self.num_elasticasRigid = 0
        self.num_rigid = 0
        self.num_elastic = 0
        self.pod_ignored = 0
    
        
    def add_to_queue(self, pod_agent):
        if not pod_agent.is_elastic:
            self.rigid_queue.append(pod_agent)
        else:
            self.elastic_queue.append(pod_agent)
    
    def next_rigidPod_please(self):
         
        # fetch next pod from queue
        next_pod = self.rigid_queue[0]
        
        # select worker 
        selected_worker = self.worker_list[0]
        
        if selected_worker.accept_as_rigid(next_pod):
            # remove from queue, add to deployed
            del self.rigid_queue[0]
            self.current_deployed_pods[next_pod.unique_id] = next_pod.demand_slack
            
            # update short lookup table
            self.add_lookup_table(next_pod)
                        
            return True  # rigid pod accepted/resource assigned
        else:
            # avoid locking by considering the next one in queue 
            if len(self.rigid_queue) > 1:
                tmp = self.rigid_queue.pop(0)
                self.rigid_queue.insert(1, tmp)  # try next one first

            return False # rigid pod rejected, perhaps next step
        
    def next_elasticPod_please(self):
        
        # fetch next pod from queue
        next_pod = self.elastic_queue[0]
        
        # select worker
        selected_worker = self.worker_list[0]
      
        if selected_worker.accept_as_elastic(next_pod):
            # accepted by a peer pod
            del self.elastic_queue[0]
            # should we add to state table with zeros slack
            self.current_deployed_pods[next_pod.unique_id] = next_pod.demand_slack
            self.add_lookup_table(next_pod)
            return True
        
        if random.random() < self.Gamma:
                
            if selected_worker.accept_as_rigid(next_pod):
                # accepted as rigid pod
                del self.elastic_queue[0]
                self.current_deployed_pods[next_pod.unique_id] = next_pod.demand_slack
                self.add_lookup_table(next_pod)
                # self.num_elasticasRigid += 1

                return True
            else:
                # avoid locking by considering the next one in queue 
                if len(self.elastic_queue) > 1:
                    tmp = self.elastic_queue.pop(0)
                    self.elastic_queue.insert(1, tmp) # try next one first
                    
                    ## pod_ignored increment in elastic: ATTA ##
                    self.pod_ignored += 1
                    # print("pods ignored (elastic):", self.pod_ignored)

                    return False
        else:
            return False
        
    def get_rigid_queue_status(self):
        # print("rigid queue status:", len(self.rigid_queue))
        return len(self.rigid_queue)
    
    def get_elastic_queue_status(self):
        # print("elastic queue status:", len(self.elastic_queue))
        return len(self.elastic_queue)
    
    def generate_key(self, slack_values):
        key = []
        for value, threshold in zip(slack_values, self.thresholds):
            # print("value, threshold: ", value, threshold)
            if value < threshold:
                key.append('L')
            else:
                key.append('H')
                
                
        # Robustness: randomly assign the bucket for a specific percentage of rigid pods' slacks 
        # print("slack error: ", self.slack_estimation_error)
        if random.random() < self.slack_estimation_error:
            key = random.choices([['L','L'],['L','H'],['H','L'],['H','H']], k=1)[0]
        
        return tuple(key)
    
    def generate_3key(self, slack_values):
        # same as above, in finer granularity
        key = []
        for value, threshold in zip(slack_values, self.thresholds):
            if value < threshold[0]:
                key.append('L')
            elif value < threshold[1]:
                key.append('M')
            else:
                key.append('H')
        return tuple(key)
    
    def add_lookup_table(self, new_pod):
        key = self.generate_key(new_pod.demand_slack)
        if key not in self.lookup_table:
            self.lookup_table[key] = []
        self.lookup_table[key].append(new_pod)
        
    def del_lookup_table(self, pod):
        for key, lst in self.model.master.lookup_table.items():
            if pod in lst:
                self.model.master.lookup_table[key].remove(pod)

    def release_resources(self, pod):
        pod.assigned_worker.release_resources(pod)
        self.model.schedule.remove(pod)  # Remove task from the schedule
 
    def step(self):
        
        self.rigid_queue_status.append(self.get_rigid_queue_status())
        self.elastic_queue_status.append(self.get_elastic_queue_status())


        '''
        next pods please: serve as far as there are pods in queue and we have 
        enough capacity; either as rigid pod or hosting by peer using slack
        ''' 
        while len(self.rigid_queue) > 0  and self.next_rigidPod_please():
            pass
        
        while len(self.elastic_queue) > 0  and  self.next_elasticPod_please():
            #print(len(self.elastic_queue), len(self.current_deployed_pods))
            pass 
       

class SchedulerModel(Model):
    
    def __init__(self, method='RND', worker_capacity=(512,512), 
                 inter_arrival_mu=0.1, prob_pod_profiles=(0.4, 0.4, 0.2),
                 prob_elastisity=0.0, config_file="config.yaml", seed=100):
        
        super().__init__()

        # Load configuration
        config = self.load_config(config_file)
        thresholds = tuple(config["scheduler"]["alpha_beta"])
        Gamma = config["scheduler"]["gamma"]
        
       
        self.inter_arrival_mu = inter_arrival_mu
        self.prob_elastisity = prob_elastisity
        self.method = method  
        self.prob_pod_profiles =  prob_pod_profiles

        self.grid = SingleGrid(2000, 2000, False)
        self.schedule = RandomActivation(self)
        
        random.seed(seed)
        np.random.seed(seed)

        self.satisfied_elastic = []
        self.un_satisfied_elastic = []
        self.satisfied_rigid = []
        self.un_satisfied_rigid = []
        
        # create one worker 
        self.agent_id = 0
        worker = Worker(self.agent_id, self, resource_capacity=worker_capacity)
        self.schedule.add(worker)
        self.worker = worker
        self.agent_id += 1
        
        # create the master agent with one worker
        master = Master(self.agent_id, self, thresholds=thresholds, Gamma=Gamma, worker_list = [worker])
        self.schedule.add(master)
        self.agent_id += 1
        
        self.master = master
        
        self.next_pod_time = random.randint(1,2)  # Random time for the first task
        
        
        self.datacollector = DataCollector(
        agent_reporters={"Queue Status": "get_queue_status"} )
        
    # load cofig file
    @staticmethod
    def load_config(config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
        
         
    def get_new_pod(self, prob_elastisity=0.0):
        next_demand, next_demand_step, next_slack, next_is_elastic, next_demand_tolerance\
            = pod_profiles.get_pod_profile(categories_prob = self.prob_pod_profiles,\
                                       prob_elastisity=prob_elastisity)
         
        next_pod = Pod(self.agent_id, self, next_demand,
                      demand_steps=next_demand_step,
                      is_elastic=next_is_elastic,
                      demand_tolerate_steps=next_demand_tolerance,
                      demand_slack=list(next_slack))
       
        self.agent_id += 1
        
        return next_pod
    
 
    def step(self):
        
        self.schedule.step()
 
        if self.schedule.steps == self.next_pod_time:
            
            new_pod = self.get_new_pod(self.prob_elastisity)
            
            new_pod.arrival_step = self.schedule.steps
            self.schedule.add(new_pod)
            self.master.add_to_queue(new_pod)
            
            #self.next_pod_time += random.randint(1, int(1./self.inter_arrival_mu))  # Set next task creation time
            self.next_pod_time += max(1,round(np.random.exponential(scale=1./self.inter_arrival_mu)))
        
        # change1: Update thresholds based on the current inter_arrival_mu
        self.master.thresholds = determine_thresholds(self.inter_arrival_mu)
        # print(f"Updated thresholds: {self.master.thresholds}")


        # change2: uncomment if you want to adjust thresholds based on the size of the elastic queue

        # # Dynamically adjust thresholds based on the size of the elastic queue
        # elastic_queue_len = len(self.master.elastic_queue)

        # # Example logic: if elastic queue is large, raise thresholds slightly;
        # # if it's small, lower them slightly.
        # if elastic_queue_len > 2:
        #     self.master.thresholds = [t + 0 for t in self.master.thresholds]
        #     print(f"Thresholds increased due to large elastic queue ({elastic_queue_len}): {self.master.thresholds}")
        # elif elastic_queue_len < 2:
        #     self.master.thresholds = [max(0, t - 4) for t in self.master.thresholds]
        #     print(f"Thresholds decreased due to small elastic queue ({elastic_queue_len}): {self.master.thresholds}")

            
            

def determine_thresholds(inter_arrival_mu):
    # adjust thresholds accordingly based on inter_arrival_mu
    if inter_arrival_mu < 0.5:
        return [5, 5]
    elif inter_arrival_mu < 1.0:
        return [5, 5]
    else:
        return [0.7, 0.8]
    

def get_agent_by_id(agent_id, model):
    for agent in model.schedule.agents:
        if agent.unique_id == agent_id:
            return agent
    return None  # Return None if no agent with the given ID is found


     
if __name__ == '__main__':
    # specify config_file
    config_file = "config.yaml"
    the_array = []   
    
    # define the scenario here
    prob_pod_profiles=(0.4, 0.4, 0.2)
    prob_elastisity = 0.5


    worker_capacity = (512,512)
    inter_arrival_mu_list = np.linspace(1.2, 1.2, num=1) #np.linspace(0.32, 1.2, num=12)
    # inter_arrival_mu_list = np.linspace(0.32, 1.2, num=12)
    seed_points = [int(x) for x in list(np.linspace(1, 10000000, num=len(inter_arrival_mu_list)))]
    
    num_steps = 12000
    # num_steps = 20000

    # Initialize lists to store satisfaction data
    satisfaction_elastic_over_steps = []
    satisfaction_rigid_over_steps = []
    
    random_model_cpu_utilization_KPI = []
    random_model_mem_utilization_KPI = []
    random_model_satisfication_elastic = []
    random_model_satisfication_rigid = [] 
    
    
    best_model_cpu_utilization_KPI = []
    best_model_mem_utilization_KPI = []
    best_model_satisfication_elastic = []
    best_model_satisfication_rigid = []

    
    swarm_model_cpu_utilization_KPI = []
    swarm_model_mem_utilization_KPI = []
    swarm_model_satisfication_elastic = []
    swarm_model_satisfication_rigid = []

    
    
    for count, inter_arrival_mu in enumerate(inter_arrival_mu_list):
        
        print(f"--------- Traffic intensity, lambda={inter_arrival_mu:.2f}")
        
        
        swarm_model = SchedulerModel(method='SWARM', 
                                      worker_capacity=worker_capacity, 
                                      inter_arrival_mu=inter_arrival_mu,
                                      prob_elastisity=prob_elastisity,
                                      prob_pod_profiles=prob_pod_profiles,
                                      seed=seed_points[count])

        # for random values of mu
        # inter_arrival_mu_array = []

        # change3: varying traffic intensity
        inter_arrival_mu_array = [
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99,
                0.99
            ]

        # for random values of mu
        # for varying mu values, where mu changese every 1000 steps
        # for i in range(num_steps):
        #     # Dynamically change the inter_arrival_mu value
        #     if i % 1000 == 0:  # Change mu every 1000 steps
        #         inter_arrival_mu = np.random.uniform(0.2, 1.0)
        #         print(f"Step: {i}, New lambda={inter_arrival_mu:.2f}")
        #         swarm_model.inter_arrival_mu = inter_arrival_mu
        #         inter_arrival_mu_array.append(inter_arrival_mu)

        #     swarm_model.step()

        #     # Collect satisfaction data at each step
        #     s31, s32 = visualization.get_satification_rate(swarm_model)
        #     satisfaction_elastic_over_steps.append(s31)
        #     satisfaction_rigid_over_steps.append(s32)
        
        # for pre-defined mu values
        for i in range(num_steps):
            # Dynamically change the inter_arrival_mu value
            if i % 1000 == 0:  # Change mu every 1000 steps
                inter_arrival_mu = inter_arrival_mu_array[0]
                print(f"Step: {i}, New lambda={inter_arrival_mu:.2f}")
                swarm_model.inter_arrival_mu = inter_arrival_mu
                inter_arrival_mu_array.pop(0)

            swarm_model.step()

            # Collect satisfaction data at each step
            s31, s32 = visualization.get_satification_rate(swarm_model)
            satisfaction_elastic_over_steps.append(s31)
            satisfaction_rigid_over_steps.append(s32)
            
        s31, s32 = visualization.get_satification_rate(swarm_model)
        q31, q32, cpu3, mem3 = visualization.get_steady_state_utilization(swarm_model)
        
        swarm_model_cpu_utilization_KPI.append(cpu3)
        swarm_model_mem_utilization_KPI.append(mem3)
        swarm_model_satisfication_elastic.append(s31)
        swarm_model_satisfication_rigid.append(s32)

        # change4: Plot satisfaction data
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_steps), satisfaction_elastic_over_steps, label='Elastic Satisfaction')
        plt.plot(range(num_steps), satisfaction_rigid_over_steps, label='Rigid Satisfaction')
        plt.xlabel('Steps')
        plt.ylabel('Satisfaction Rate')
        # plt.title(f'Satisfaction Rate over Steps for lambda={inter_arrival_mu:.2f}')
        plt.title(f'Satisfaction Rate over Steps for lambda=ARBITRARY')
        plt.legend()
        # plt.savefig(f'Satisfaction_Lambda-{inter_arrival_mu:.1f}.pdf')
        plt.savefig(f'Satisfaction_Lambda-ARBITRARY.pdf')
        # plt.show()


        # plot graphs for the selected inter-arrival rate
        # print(f"--------- plot for traffic intensity, lambda={inter_arrival_mu:.2f}")
        print(f"--------- plot for traffic intensity, lambda=ARBITRARY")

        fig1, (ax11, ax12) = plt.subplots(2,1, figsize=(10,6),sharex=True)
        ax12, ax12 = visualization.plot_dynamics(swarm_model, ax11, ax12, q1_color='g', q2_color='#3CB371', cpu_color='g', mem_color='g', alpha=1.0, label='bottom-up')
        fig1.savefig(f'Lambda-{inter_arrival_mu:.1f}.pdf')

        plt.show()


    print(swarm_model_cpu_utilization_KPI, swarm_model_mem_utilization_KPI, swarm_model_satisfication_elastic, swarm_model_satisfication_rigid, inter_arrival_mu_array)
    # plt.show()

 
    