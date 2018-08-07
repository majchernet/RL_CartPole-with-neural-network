import gym, collections
import numpy as np
import plots

from nn import NN

verbose = False

# CartPole environment 
env = gym.make('CartPole-v0')

# Number of runs experiment
NR = 100

# Number of episodes to observe in a single experiment
NE = 2000

# Max time steps in episode
T = 200
env.max_episode_steps = T

# All states reached during experiment
allStates = []

# Calculate mean value from lastNTry  
lastNTry = 100

# Minimal explore vs exploit rate
minimalEve = 0.01

# If is true, minimalEve is taken as fixed value for explore vs exploit probability rate
fixedEve = False

# Discount Rate
discountRate = 0.98

# Minimal update of total reward that matter 
minimalUpdate = 0.1


    
"""
  Determine the probability of environmental exploration base on number of try in experiment and recent results
  1 - explore 
  0 - exploit
"""
def exploreRate(i):
    
    # minimal explore vs exploit rate 
    global minimalEve, exploreResults
    
    if fixedEve:
        return minimalEve    
 
    # Forece explore througt first 100 tries
    if i < 100:
        return 1
    
    # calculate eve value based on recent tries results and number of episode
    rateLastNTry = 1 - (sum(exploreResults[-lastNTry:])/lastNTry)*(1.0/T)
    rateStep = 1 - i * (1/NE)
    rate = (2*rateLastNTry + 8*rateStep) / 10

    return rate if rate > minimalEve else minimalEve


"""
 Run single episode 
"""
def runEpisode(env, T, eve, episodeNumber, render = False):
    global allStates
 
    # All states in single episode
    episodeStates = []
    observation = env.reset()

    totalReward = 0

    for t in range(T):
        if verbose:
            print ("\nstep {0} eve {1}".format(t,eve))
            print (observation)
 
        if eve > np.random.random(): #random explore
            action = env.action_space.sample()
            if verbose:
                print ("random move {0}".format(action))
        else: # get best action for curent state 
            if verbose:
                print ("get best policy action")

            action = np.argmax(nn.ask([observation]))
            
        
        if (render):
            env.render()
            
        prevObservation = observation
        observation, reward, done, info = env.step(action)

        episodeStates.append([t,prevObservation, action, reward])
    
        if done:
            # backward calculate rewards for episode 
            for move in range(len (episodeStates)-1,0,-1):
                update = episodeStates[move][3] * discountRate ** (len (episodeStates)-1-move)

                if (update > minimalUpdate): #minimal update that matter 
                    totalReward += update

                # add totalreward to states table
                episodeStates[move].append(totalReward)
                episodeStates[move].append(episodeNumber)
                
                allStates.append(episodeStates[move])
            
            return {'t':t, 'eps': episodeStates}


"""
 Run NR experiments
"""
for e in range(1,NR):
    # Neural Network to store knowledge 
    nn = NN([4,5,2])

    # Explore result table for calculate mean value
    exploreResults = [0] * 100
   
    mean = [0] * NE
    maxi = 0
    maxt = 0 

    # In one run try NE episodes
    for i in range(1,NE):
        res = runEpisode(env, T, exploreRate(i), i)
        
        # remove oldest result
        exploreResults.pop(0)                
        # and append newest result
        exploreResults.append(res['t'])
        
        # average of recent attempts
        mean[i] = np.mean(exploreResults)
        if mean[i] > maxi:
            maxi = mean[i]

        if res['t'] > maxt:
            maxt = res['t']                

        # don't learn first 100 tries
        if (i<100):
            continue

        # take 10k last sates
        del allStates[:-10000]
        
        # and sort them by total obtained reward
        allStatesSorted = sorted (allStates, key=lambda x: x[4])
        
        # take best 1k resul
        X = [row[1] for row in allStatesSorted[-1000:]]
        Y = [row[2] for row in allStatesSorted[-1000:]]
            
        # Y is a list that contains 0 or 1 for made movements
        # convert Y to vectors list, becouse neural network has two output nodes (for move left or right)
        Y_ = []
        for y in Y:
            if y == 0:
                Y_.append([1,0])
            else: 
                Y_.append([0,1])

        # train neural network model 
        nn.train(X,Y_)
    
    plots.plotResults(mean,e)
    
    print ("Run {} max last 100 explore try mean {}".format(e,maxi))

    res = runEpisode(env, T, 0, 1, True)
    print ("Test score {}".format(res['t']))
    del nn
