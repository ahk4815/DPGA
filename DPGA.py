#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

class Solution():    
    #structure of the solution 
    def __init__(self):
        self.num_features = None
        self.num_agents = None
        self.max_iter = None
        self.obj_function = None
        self.execution_time = None
        self.convergence_curve = {}
        self.best_agent = None
        self.best_fitness = None
        self.best_accuracy = None
        self.final_population = None
        self.final_fitness = None
        self.final_accuracy = None


class Data():
    # structure of the training data
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.val_X = None
        self.val_Y = None



def initialize(num_agents, num_features):
    # define min and max number of features
    min_features = int(0.3 * num_features)
    max_features = int(0.6 * num_features)

    # initialize the agents with zeros
    agents = np.zeros((num_agents, num_features))

    # select random features for each agent
    for agent_no in range(num_agents):

        # find random indices
        cur_count = np.random.randint(min_features, max_features)
        temp_vec = np.random.rand(1, num_features)
        temp_idx = np.argsort(temp_vec)[0][0:cur_count]

        # select the features with the ranom indices
        agents[agent_no][temp_idx] = 1   

    return agents



def sort_agents(agents, obj, data):
    # sort the agents according to fitness
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    (obj_function, weight_acc) = obj

    # if there is only one agent
    if len(agents.shape) == 1:
        num_agents = 1
        fitness = obj_function(agents, train_X, val_X, train_Y, val_Y, weight_acc)
        return agents, fitness

    # for multiple agents
    else:
        num_agents = agents.shape[0]
        fitness = np.zeros(num_agents)
        for id, agent in enumerate(agents):
            fitness[id] = obj_function(agent, train_X, val_X, train_Y, val_Y, weight_acc)
        idx = np.argsort(-fitness)
        sorted_agents = agents[idx].copy()
        sorted_fitness = fitness[idx].copy()
        reduced_agents= sorted_agents[:len(sorted_agents)//2]
        reduced_fitness = sorted_fitness[:len(sorted_fitness)//2]
    return reduced_agents, reduced_fitness



def display(agents, fitness, agent_name='Agent'):
    # display the population
    print('\nNumber of agents: {}'.format(agents.shape[0]))
    print('\n------------- Best Agent ---------------')
    print('Fitness: {}'.format(fitness[0]))
    print('Number of Features: {}'.format(int(np.sum(agents[0]))))
    print('----------------------------------------\n')

    for id, agent in enumerate(agents):
        print('{} {} - Fitness: {}, Number of Features: {}'.format(agent_name, id+1, fitness[id], int(np.sum(agent))))

    print('================================================================================\n')


#score=fitness_score_values(data,target)
#print(score)
def compute_accuracy(agent, train_X, test_X, train_Y, test_Y): 
    # compute classification accuracy of the given agents
    fitness=0
    number=0
    for i in range(len(agent)):
        if agent[i]==1:
            number=number+1
            fitness=fitness+score[i]
    return fitness/number        

def DP(chromosomes,num_agents, num_features):
    opposite_chromosomes=[]
    #print(chromosomes)
    for i in range(len(chromosomes)):
        opp=[0 for i in range(num_features)]
        #print(i)
        for j in range(len(chromosomes[i])):
            #opp.append(int(chromosomes[i][j])^1)
            if chromosomes[i][j]==1:
                opp[independent_features[j]]=1
            else:
                opp[j]=1
                
        opposite_chromosomes.append(opp)
    return opposite_chromosomes    
            
def merge(chromosomes,opposition_chromosomes,num_agents, num_features):
    merge_chromosomes=[]
    for i in range(len(chromosomes)):
        merge_chromosomes.append(chromosomes[i])
    for i in range(len(opposition_chromosomes)):
        merge_chromosomes.append(opposition_chromosomes[i])
    return np.asarray(merge_chromosomes)    
        
        
    

def compute_fitness(agent, train_X, test_X, train_Y, test_Y, weight_acc=0.9):
    # compute a basic fitness measure
    if(weight_acc == None):
        weight_acc = 0.9

    weight_feat = 1 - weight_acc
    num_features = agent.shape[0]
    
    acc = compute_accuracy(agent, train_X, test_X, train_Y, test_Y)
    feat = (num_features - np.sum(agent))/num_features

    fitness = weight_acc * acc + weight_feat * feat
    
    return fitness


# In[54]:


import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets


def GA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, prob_cross=0.4, prob_mut=0.3, save_conv_graph=False):

    # Genetic Algorithm
    ############################### Parameters ####################################
    #                                                                             #
    #   num_agents: number of chromosomes                                         #
    #   max_iter: maximum number of generations                                   #
    #   train_data: training samples of data                                      #
    #   train_label: class labels for the training samples                        #                
    #   obj_function: the function to maximize while doing feature selection      #
    #   prob_cross: probability of crossover                                      #
    #   prob_mut: probability of mutation                                         #
    #   save_conv_graph: boolean value for saving convergence graph               #
    #                                                                             #
    ###############################################################################

    short_name = 'GA'
    agent_name = 'Chromosome'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    cross_limit = 5

    # setting up the objectives
    weight_acc = None
    if(obj_function==compute_fitness):
        weight_acc = float(input('Weight for the classification accuracy [0-1]: '))
    obj = (obj_function, weight_acc)
    compute_accuracy = (compute_fitness, 1) # compute_accuracy is just compute_fitness with accuracy weight as 1

    # initialize chromosomes and Leader (the agent with the max fitness)
    chromosomes = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    accuracy = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")

    # initialize convergence curves
    convergence_curve = {}
    convergence_curve['fitness'] = np.zeros(max_iter)
    convergence_curve['feature_count'] = np.zeros(max_iter)

    # initialize data class
    data = Data()
    val_size = float(input('Enter the percentage of data wanted for valdiation [0, 100]: '))/100
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(train_data, train_label, stratify=train_label, test_size=val_size)

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial population
    chromosomes, fitness = sort_agents(chromosomes, obj, data)

    # start timer
    start_time = time.time()
    
    # main loop
    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')
        
        #opposition based population of chromosomes
        opposition_chromosomes=DP(chromosomes,num_agents, num_features)
        
        # perform crossover, mutation and replacement
        cross_mut(chromosomes, fitness, obj_function, data, prob_cross, cross_limit, prob_mut)

        #merge both the set of chromosomes
        chromosomes= merge(chromosomes,opposition_chromosomes,num_agents, num_features)
        
        # update final information
        chromosomes, fitness = sort_agents(chromosomes, obj, data)
        display(chromosomes, fitness, agent_name)
        if fitness[0]>Leader_fitness:
            Leader_agent = chromosomes[0].copy()
            Leader_fitness = fitness[0].copy()
        convergence_curve['fitness'][iter_no] = Leader_fitness
        convergence_curve['feature_count'][iter_no] = int(np.sum(Leader_agent))

    # compute final accuracy
    Leader_agent, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    chromosomes, accuracy = sort_agents(chromosomes, compute_accuracy, data)

    print('\n================================================================================')
    print('                                    Final Result                                  ')
    print('================================================================================\n')
    print('Leader ' + agent_name + ' Dimension : {}'.format(int(np.sum(Leader_agent))))
    print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
    
    print('\n================================================================================\n')

    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time
    
    

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.convergence_curve = convergence_curve
    solution.final_population = chromosomes
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution


def crossover(parent_1, parent_2, prob_cross):
    # perform crossover with crossover probability prob_cross
    num_features = parent_1.shape[0]
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()

    for i in range(num_features):
        if(np.random.rand()<prob_cross):
            child_1[i] = parent_2[i]
            child_2[i] = parent_1[i]

    return child_1, child_2


def mutation(chromosome, prob_mut):
    # perform mutation with mutation probability prob_mut
    num_features = chromosome.shape[0]
    mut_chromosome = chromosome.copy()

    for i in range(num_features):
        if(np.random.rand()<prob_mut):
            mut_chromosome[i] = 1-mut_chromosome[i]
    
    return mut_chromosome


def roulette_wheel(fitness):
    # Perform roulette wheel selection
    maximum = sum([f for f in fitness])
    selection_probs = [f/maximum for f in fitness]
    return np.random.choice(len(fitness), p=selection_probs)


def cross_mut(chromosomes, fitness, obj_function, data, prob_cross, cross_limit, prob_mut):
    # perform crossover, mutation and replacement
    count = 0
    num_agents = chromosomes.shape[0]
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    print('Crossover-Mutation phase starting....')

    while(count<cross_limit):
        print('\nCrossover no. {}'.format(count+1))
        id_1 = roulette_wheel(fitness)
        id_2 = roulette_wheel(fitness)

        if(id_1 != id_2):
            child_1, child_2 = crossover(chromosomes[id_1], chromosomes[id_2], prob_cross)
            child_1 = mutation(child_1, prob_mut)
            child_2 = mutation(child_2, prob_mut)
            fitness_1 = obj_function(child_1, train_X, val_X, train_Y, val_Y)
            fitness_2 = obj_function(child_2, train_X, val_X, train_Y, val_Y)

            if(fitness_1 < fitness_2):
                temp = child_1, fitness_1
                child_1, fitness_1 = child_2, fitness_2
                child_2, fitness_2 = temp

            for i in range(num_agents):
                if(fitness_1 > fitness[i]):
                    print('1st child replaced with chromosome having id {}'.format(i+1))
                    chromosomes[i] = child_1
                    fitness[i] = fitness_1
                    break

            for i in range(num_agents):
                if(fitness_2 > fitness[i]):
                    print('2nd child replaced with chromosome having id {}'.format(i+1))
                    chromosomes[i] = child_2
                    fitness[i] = fitness_2
                    break

            count = count+1

        else:
            print('Crossover failed....')
            print('Restarting crossover....\n')


# In[20]:


import pandas as pd
df=pd.read_csv('LtrP_U_Smote.csv')
(a,b)=df.shape
print(a,b)
target=df['attr_768']
data=df[df.columns.drop('attr_768')]
for i in range(len(target)):
    target[i]=target[i][1]
for i in range(len(target)):
    target[i]=int(target[i])
import numpy as np
target=np.array(target)
#print(target)
data=np.array(data)
print(data)

target=target.astype('int')


# In[32]:


def fitness_score_values(data,target):
    from Py_FS.filter import PCC as PCC
    from Py_FS.filter import MI as MI
    from sklearn.feature_selection import mutual_info_classif
    from Py_FS.filter import Relief as Relief
    from sklearn.feature_selection import chi2
    
    #print('Pcc scores')
    #PCC_solution = PCC(data, target)
    #print(PCC_solution.scores)
    print('MI scores')
    mi_scores=mutual_info_classif(data, target, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
    print((mi_scores))
    print('Relief scores')
    Relief_solution = Relief(data, target)
    print((Relief_solution.scores))
    print('chi sqaure scores')
    chi_scores = chi2(data,target)
    max_score=max(chi_scores[0])
    for i in range(len(chi_scores[0])):
        chi_scores[0][i]=chi_scores[0][i]/max_score
    print((chi_scores[0]))
    shapley=[]
    for i in range(b-1):
        shapley.append((mi_scores[i]+chi_scores[0][i]+Relief_solution.scores[i])/3)
    return shapley


# In[45]:


def PCC_features():
    from scipy import stats
    independent_features=[]
    for i in range(b-1):
        x=np.asarray(df['attr_'+str(i+1)])
        mmin=10000
        position=-1
        for j in range(b-1):
            y=np.asarray(df['attr_'+str(j+1)])
            corr,_=stats.pearsonr(x,y)
            if corr<mmin:
                mmin=corr
                position=j
        independent_features.append(position)
    return independent_features    


# In[33]:


#fitness_score_values
score=fitness_score_values(data,target)
print(score)


# In[46]:



independent_features=PCC_features()
print(independent_features)


# In[56]:



solution = GA(num_agents=30, max_iter=30, train_data=data,  prob_cross=0.6, prob_mut=0.4,train_label=target,save_conv_graph=True)


# In[43]:



x=np.asarray(df['attr_1'])
y=np.asarray(df['attr_2'])
stats.pearsonr(x,y)


# In[44]:





# In[ ]:





# In[ ]:




