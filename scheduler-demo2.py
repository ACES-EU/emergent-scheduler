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


Abdorasoul Ghasemi, arghasemi@gmail.com
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
               
               return True
        else:
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
                   xpos += random.randint(-5,5)
                   ypos += random.randint(-5,5)
            self.model.grid.place_agent(pod, (xpos, ypos))
            
            peer_pod.assigned_cpu -= pod.demand[0] #peer_cpu_slack
            peer_pod.assigned_mem -= pod.demand[1] #peer_mem_slack
            peer_pod.demand_slack[0] -= pod.demand[0]
            peer_pod.demand_slack[1] -= pod.demand[1]
            
            # no more elastic pod can exploit this
            self.model.master.current_deployed_pods[peer_id] = (0,0)

            return True
        else:
                               
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
                return True
            else:
                # avoid locking by considering the next one in queue 
                if len(self.elastic_queue) > 1:
                    tmp = self.elastic_queue.pop(0)
                    self.elastic_queue.insert(1, tmp) # try next one first
                    return False
        else:
            return False
        
    def get_rigid_queue_status(self):
        return len(self.rigid_queue) 
    
    def get_elastic_queue_status(self):
        return len(self.elastic_queue)
    
    def generate_key(self, slack_values):
        key = []
        for value, threshold in zip(slack_values, self.thresholds):
            if value < threshold:
                key.append('L')
            else:
                key.append('H')
                
                
        # Robustness: randomly assign the bucket for a specific percentage of rigid pods' slacks 
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
            
            # xpos = round(new_pod.demand_steps*new_pod.demand_slack[0])
            # ypos = round(new_pod.demand_steps*new_pod.demand_slack[1])
            # while self.grid.is_cell_empty((xpos, ypos)) is False:
            #     xpos += random.randint(-20,20)
            #     ypos += random.randint(-20,20)
                
            # self.grid.place_agent(new_pod, (xpos, ypos))
            
            new_pod.arrival_step = self.schedule.steps
            self.schedule.add(new_pod)
            self.master.add_to_queue(new_pod)
            
            #self.next_pod_time += random.randint(1, int(1./self.inter_arrival_mu))  # Set next task creation time
            self.next_pod_time += max(1,round(np.random.exponential(scale=1./self.inter_arrival_mu)))
            
            


def get_agent_by_id(agent_id, model):
    for agent in model.schedule.agents:
        if agent.unique_id == agent_id:
            return agent
    return None  # Return None if no agent with the given ID is found



def plot_grid(model, fig, ax, sc1, sc2, width, height):
    
    pods = [agent for agent in model.schedule.agents if isinstance(agent, Pod)]
    
    rigids = [item for item in pods if not item.is_elastic]
    elastics = [item for item in pods if item.is_elastic]
    rigid_positions = [agent.pos for agent in rigids if agent.pos is not None]
    elastic_positions=[agent.pos for agent in elastics if agent.pos is not None]
    
    if len(rigid_positions)>0:
        x1, y1 = zip(*rigid_positions)
        sc1.set_offsets(np.c_[x1,y1])
        
    if len(elastic_positions)>0:
        x2, y2 = zip(*elastic_positions)
        sc2.set_offsets(np.c_[x2,y2])
   
    fig.canvas.draw_idle()
    plt.pause(0.01)

     
if __name__ == '__main__':
    # specify config_file
    config_file = "config.yaml"   
    
    # define the scenario here
    prob_pod_profiles=(0.4, 0.4, 0.2)
    prob_elastisity = 0.5


    worker_capacity = (512,512)
    inter_arrival_mu_list = np.linspace(0.05, 1.01, num=11) #np.linspace(0.32, 1.2, num=12)
    seed_points = [int(x) for x in list(np.linspace(1, 10000000, num=len(inter_arrival_mu_list)))]
    
    
    num_steps = 12000
    
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
  
    
    #visualize the slack 
    #plt.ion()
    # width=height=5000
    # fig0, ax0 = plt.subplots(figsize=(6,6))
    # sc1 = ax0.scatter([],[], color='green', marker='s', label='rigid pods')
    # sc2 = ax0.scatter([],[], color='red', marker='p', label='elastic pods')

    # plt.xlim(0,width)
    # plt.ylim(0,height)
    # plt.legend(loc='upper left')
    # plt.draw()
    
    
    for count, inter_arrival_mu in enumerate(inter_arrival_mu_list):
        
        print(f"--------- Traffic intensity, lambda={inter_arrival_mu:.2f}")

       
        random_model = SchedulerModel(method='RND', 
                                      worker_capacity=worker_capacity,
                                      prob_elastisity=prob_elastisity,
                                      prob_pod_profiles=prob_pod_profiles,
                                      inter_arrival_mu=inter_arrival_mu,
                                      seed=seed_points[count])
        
        for i in range(num_steps):
            random_model.step()
            #plot_grid(random_model, fig0, ax0, sc1, sc2, width, height)
        
        
        s11, s12 = visualization.get_satification_rate(random_model)
        
        q11, q12, cpu1, mem1 = visualization.get_steady_state_utilization(random_model)
        
       
        random_model_cpu_utilization_KPI.append(cpu1)
        random_model_mem_utilization_KPI.append(mem1)
        random_model_satisfication_elastic.append(s11)
        random_model_satisfication_rigid.append(s12)

 
        best_model = SchedulerModel(method='BEST', 
                                      worker_capacity=worker_capacity, 
                                      inter_arrival_mu=inter_arrival_mu,
                                      prob_elastisity=prob_elastisity,
                                      prob_pod_profiles=prob_pod_profiles,
                                      seed=seed_points[count])
        
        for i in range(num_steps):
              best_model.step()
              #plot_grid(best_model, fig0, ax0, sc1, sc2, width, height)

            
        s21, s22 = visualization.get_satification_rate(best_model)
        q21, q22, cpu2, mem2 = visualization.get_steady_state_utilization(best_model)
        
        best_model_cpu_utilization_KPI.append(cpu2)
        best_model_mem_utilization_KPI.append(mem2)
        best_model_satisfication_elastic.append(s21)
        best_model_satisfication_rigid.append(s22)
        
        
        swarm_model = SchedulerModel(method='SWARM', 
                                      worker_capacity=worker_capacity, 
                                      inter_arrival_mu=inter_arrival_mu,
                                      prob_elastisity=prob_elastisity,
                                      prob_pod_profiles=prob_pod_profiles,
                                      seed=seed_points[count])
        
        for i in range(num_steps):
              swarm_model.step()
              #plot_grid(best_model, fig0, ax0, sc1, sc2, width, height)

            
        s31, s32 = visualization.get_satification_rate(swarm_model)
        q31, q32, cpu3, mem3 = visualization.get_steady_state_utilization(swarm_model)
        
        swarm_model_cpu_utilization_KPI.append(cpu3)
        swarm_model_mem_utilization_KPI.append(mem3)
        swarm_model_satisfication_elastic.append(s31)
        swarm_model_satisfication_rigid.append(s32)
  
            

        if count in [1, 4, 8]:#inter_arrival_mu in inter_arrival_mu_list[::4]:
            print(f"--------- plot for traffic intensity, lambda={inter_arrival_mu:.2f}")

            fig1, (ax11, ax12) = plt.subplots(2,1, figsize=(10,6),sharex=True)
            ax11, ax12 = visualization.plot_dynamics(random_model, ax11, ax12, q1_color='blue',q2_color='#00BFFF', cpu_color='blue', mem_color='blue', alpha=0.3, label='random')
            ax12, ax12 = visualization.plot_dynamics(best_model, ax11, ax12, q1_color='r', q2_color='#FF6347', cpu_color='red', mem_color='red', alpha=1.0, label='best')
            ax12, ax12 = visualization.plot_dynamics(swarm_model, ax11, ax12, q1_color='g', q2_color='#3CB371', cpu_color='g', mem_color='g', alpha=1.0, label='bottom-up')
            fig1.savefig(f'Lambda-{inter_arrival_mu:.1f}.pdf')

        #plt.show()

    
   
    
    # KPI visualization
    fig2, (ax21, ax22) = plt.subplots(1,2, figsize=(12,6))
    ax21.set_ylim([-0.01,1.01])
    ax21.set_xlim([-0.0,max(inter_arrival_mu_list)+0.05])
    for ax in (ax21, ax22):
        ax.set_aspect('equal')

    avg_inter_arrival=[1./item for item in inter_arrival_mu_list]
    ax21.plot(inter_arrival_mu_list, random_model_cpu_utilization_KPI, color='b', label='random')
    #ax21.plot(inter_arrival_mu_list, random_model_mem_utilization_KPI, color='b',linestyle='--' )
    ax21.plot(inter_arrival_mu_list, best_model_cpu_utilization_KPI, color='r', label='best')
    #ax21.plot(inter_arrival_mu_list, best_model_mem_utilization_KPI, color='r', linestyle='--')
   
    ax21.plot(inter_arrival_mu_list, swarm_model_cpu_utilization_KPI, color='g', label='bottom-up')
    
    ax21.axvline(x=0.55, color='k', linestyle='--')
    ax21.text(0.5,0.2, r'$\lambda_0$', fontsize=12, color='k')

    ax21.axhline(y=0.72,  color='k', linestyle='--')
    ax22.axvline(x=0.55, color='k', linestyle='--')
    ax22.text(0.5,0.2, r'$\lambda_0$', fontsize=12, color='k')
    ax21.set_xlabel(r'$\lambda$')
    ax21.set_ylabel('CPU/RAM utilization')
    ax21.legend()
 
   
    ax22.plot(inter_arrival_mu_list, random_model_satisfication_rigid, color='b', label='random-rigid' )
    ax22.plot(inter_arrival_mu_list, best_model_satisfication_rigid, color='r', label='best-rigid')
    ax22.plot(inter_arrival_mu_list, swarm_model_satisfication_rigid, color='g', label='bottom-up-rigid')
    ax22.plot(inter_arrival_mu_list, random_model_satisfication_elastic, color='b', linestyle='--', label='random-elastic' )
    ax22.plot(inter_arrival_mu_list, best_model_satisfication_elastic, color='r', linestyle='--', label='best-elastic')
    ax22.plot(inter_arrival_mu_list, swarm_model_satisfication_elastic, color='g', linestyle='--', label='bottom-up-elastic')

    ax22.set_xlabel(r'$\lambda$')
    ax22.set_ylabel('satisfication rate')
    ax22.set_ylim([-0.01,1.01])
    ax22.set_xlim([-0.0,max(inter_arrival_mu_list)+0.05])
    ax22.legend()
    fig2.savefig('KPI.pdf')


    plt.show()

 
    