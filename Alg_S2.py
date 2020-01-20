import random as r 
from deap import base, creator, tools
from time import sleep
import sys 
import numpy as np
import matplotlib.pyplot as plt

from Progetto import load
from Multi_2 import MLP_2
from datetime import datetime


X_train, y_train, X_test, y_test = load()

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)

print("Insert the desidered range [c_1,d_1] for the number of neurons in the hidden layer")
c_1, d_1 = int(input("c_1 = ")), int(input("d_1 = "))
print("Insert the desidered range [c_2,d_2] for the minibatch size")
c_2, d_2 = int(input("c_2 = ")), int(input("d_2 = "))
print("Insert the desidered range [c_3,d_3] for the learning rate")
c_3, d_3 = float(input("c_3 = ")), float(input("d_3 = "))

acc = {'acc': [], 'avg': []}

func_seq = [lambda:r.randint(c_1,d_1), lambda:r.randint(c_1,d_1), lambda: r.random()*(d_3-c_3), lambda:r.randint(c_2,d_2)]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initCycle, creator.Individual,
                 func_seq, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, 20)

def evaluate(Individual):
    net = MLP_2(Individual[0], Individual[1], Individual[2], Individual[3])
    return net.fit(X_train, y_train, X_test, y_test, 10),
    
def new_mut(Individual, indpb = 0.15):
    l1= [0,0,0,0]
    if (r.random() < indpb):
        l1[0] = (r.gauss(mu = Individual[0], sigma =2))
    if (r.random() < indpb):
        l1[1] = (r.gauss(mu = Individual[1], sigma =2))
    if (r.random() < indpb):
        l1[2] = (r.gauss(mu = Individual[3], sigma =2))
    if (r.random() < indpb):
        l1[3] = (r.gauss(mu = Individual[2], sigma =0.25))
    return [abs(int(l1[0])), abs(int(l1[1])), abs(l1[3]), abs(int(l1[2]))]
    
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform)
toolbox.register("mutate", new_mut)
toolbox.register("select", tools.selTournament, tournsize = 3, fit_attr='fitness')

def main():
    t1 = datetime.now()
    print(t1)
    pop = toolbox.population()
    CXPB, MUTPB, NGEN = 0.4, 0.2, 3
    
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    for g in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone,offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if(r.random() < CXPB):
                toolbox.mate(child1, child2, indpb = 0.2)
                del child1.fitness.values
                del child2.fitness.values
            
        for mutant in offspring:
            if(r.random() < MUTPB):
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        individui_da_valutare = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, individui_da_valutare))
        for ind, fit in zip(individui_da_valutare, fitnesses):
            ind.fitness.values = fit
        avg = 0.
        fits = []
        for ind in pop:
            avg+=ind.fitness.values[0]
            fits.append(ind.fitness.values[0])
        avg = avg/len(pop)
        acc['avg'].append(avg)
        
        index = np.argmax(fits)
        acc['acc'].append(fits[index])
           
        sys.stdout.write("[%-20s] %d%%" % ('='*int((g+1)*20/NGEN), (100/NGEN)*(g+1)))
        sys.stdout.flush()
        sleep(0.25)
        
        pop[:] = offspring
        
        pop_2 = [list(ind) for ind in pop]
        
        if len(np.unique(pop_2, axis = 0)) <=5:
            print('Last Gen ' +str(g+1))
            plt.plot(range(len(acc['avg'])), acc['avg'])
            plt.xlabel('Generations')
            plt.ylabel('Average Fitness')
            plt.show()
            plt.plot(range(len(acc['acc'])), acc['acc'])
            plt.xlabel('Generations')
            plt.ylabel('Best Fitness')
            plt.title('Best Fitness vs Generations')
            plt.show()
            print(pop)
            t2 = datetime.now()
            print(t2-t1)
            break
        
        print("Generation #" + str(g))
      
    results = []
    for ind in pop:
        results.append((ind, ind.fitness.values))
    
    plt.plot(range(len(acc['avg'])), acc['avg'])
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness vs Generations')
    plt.show()
    plt.plot(range(len(acc['acc'])), acc['acc'])
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness vs Generations')
    plt.show()
    t2 = datetime.now()
    print(t2-t1)
    return results
    
    
    
main()
    
