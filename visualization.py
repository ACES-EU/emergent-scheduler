# -*- coding: utf-8 -*-
"""
Preprocessing and visualization 

"""

def get_running_avg(vector):
    # return running average of input vector
    avg = []
    num_steps = len(vector)
    for step in range(num_steps):
        tmp = sum(vector[:step + 1]) / (step + 1) if step >= 0 else 0.0
        avg.append(tmp)
        
    return avg
 
    
def get_steady_state_utilization(model):
    
    avg_cpu_utilization = get_running_avg(model.worker.cpu_utilization)
    avg_mem_utilization = get_running_avg(model.worker.mem_utilization)
    avg_rigid_queue_length = get_running_avg(model.master.rigid_queue_status)
    avg_elastic_queue_length = get_running_avg(model.master.elastic_queue_status)


    q1_ss = avg_rigid_queue_length[-1]
    q2_ss = avg_elastic_queue_length[-1]

    cpu_ss = avg_cpu_utilization[-1]
    mem_ss = avg_mem_utilization[-1]
    
    return q1_ss, q2_ss, cpu_ss, mem_ss


def plot_dynamics(model, ax1, ax2, alpha=0.4, q1_color = 'k', q2_color='g', cpu_color='k', mem_color='green',
                  label=''):
    

    cpu_utilization = model.worker.cpu_utilization
    mem_utilization = model.worker.mem_utilization
    rigid_queue_length = model.master.rigid_queue_status
    elastic_queue_length = model.master.elastic_queue_status

    avg_cpu_utilization = get_running_avg(cpu_utilization)
    avg_mem_utilization = get_running_avg(mem_utilization)
    avg_rigid_queue_length = get_running_avg(rigid_queue_length)
    avg_elastic_queue_length = get_running_avg(elastic_queue_length)

    
    # Visualize server utilization
    ax1.plot(range(len(rigid_queue_length)), rigid_queue_length, alpha=1, linewidth=0.8,
              color=q1_color)
    ax1.plot(range(len(avg_rigid_queue_length)), avg_rigid_queue_length, linewidth=3, alpha=1,
              color=q1_color, label=label+'-rigid')
    
    
    ax1.plot(range(len(elastic_queue_length)), elastic_queue_length, linewidth=0.8,
              color=q2_color, alpha=0.5)
    ax1.plot(range(len(avg_elastic_queue_length)), avg_elastic_queue_length, linewidth=3, 
              color=q2_color, label=label+'-elastic', alpha=0.5)

    ax1.set_ylabel('Master Queue Length')
    #ax1.set_title('Baseline mechanism: Master queue length (top panel) and cpu/mem utilization (bottom panel)')
    ax1.legend()
    
    #ax2.plot(range(len(cpu_utilization)), cpu_utilization, color=cpu_color, alpha=alpha, linewidth=0.6)
    ax2.plot(range(len(avg_cpu_utilization)), avg_cpu_utilization, color=cpu_color, linewidth=3, label=label)

    #ax2.plot(range(len(mem_utilization)), mem_utilization, color=mem_color, linestyle='dotted', alpha=alpha, linewidth=0.6)
    ax2.plot(range(len(avg_mem_utilization)), avg_mem_utilization, color=mem_color, linestyle='dotted', linewidth=3)

    
    ax2.set_xlabel('steps')
    ax2.set_ylabel('Worker utilization')
    #ax2.set_title('Baseline CPU/Mem utilization')
    ax2.set_ylim([-0.1,1.1])
    ax2.legend()
    
    return ax1, ax2
    
     

def get_satification_rate(model):
    
    num_satisfied_elastic = len(model.satisfied_elastic)
    num_un_satisfied_elastic = len(model.un_satisfied_elastic)

    num_satisfied_rigid = len(model.satisfied_rigid)
    num_un_satisfied_rigid = len(model.un_satisfied_rigid)

    if num_satisfied_elastic+num_un_satisfied_elastic > 0:
        s1 = num_satisfied_elastic/(num_satisfied_elastic+num_un_satisfied_elastic)
    else:
        s1 = 0.
        
    if num_satisfied_rigid+num_un_satisfied_rigid > 0:
        s2 = num_satisfied_rigid/(num_satisfied_rigid+num_un_satisfied_rigid)
    else:
        s2 = 0.
        
    return s1, s2
    
