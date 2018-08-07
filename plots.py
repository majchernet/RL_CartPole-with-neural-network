import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

data = [[[]] * 2]

def plotResults(mean,nr):
    
    global data
    
    data.append([range(0,2000), mean])
    
    plt.figure(figsize=(8, 6), dpi=100)
    plt.subplot(1, 1, 1)
    plt.title("Experiments reults")
    
    # Set x limits
    plt.xlim(0, 2000)
    plt.xlabel('Number of episodes')
    
    # Set y limits
    plt.ylim(0, 200)
    plt.ylabel('Mean score of last 100 episodes')
    
    for line in range (1,nr+1):
        plt.plot(data[line][0],data[line][1],lw=0.5)
    
    plt.show()
    return
