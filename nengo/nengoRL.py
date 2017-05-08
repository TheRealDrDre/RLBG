import PSSTask as pt
import nengo
import numpy as np

state = pt.PSS_State(('A','B'))


model = nengo.Network()
with model:
    #think about how to use SPA to represent these
    stim = nengo.Node(state.statevec)

    
    currState = nengo.Ensemble(n_neurons = 600, dimensions = 6, radius = 1.4)
    nengo.Connection(stim, currState)

    bg = nengo.networks.actionselection.BasalGanglia(6)
    
    def utilityA(x):
        return x*0.5 #this is purposely initialized incorrectly in order to demonstrate learning
    
    def utilityC(x):
        return x*0.7
    
    def utilityE(x):
        return x*0.6
    
    def utilityF(x):
        return x*0.4
    
    def utilityD(x):
        return x*0.3
    
    def utilityB(x):
        return x*0.2
    
    connA = nengo.Connection(currState[0], bg.input[0], function = utilityA, learning_rule_type=nengo.PES())
    connC = nengo.Connection(currState[1], bg.input[1], function = utilityC, learning_rule_type=nengo.PES())
    connE = nengo.Connection(currState[2], bg.input[2], function = utilityE, learning_rule_type=nengo.PES())
    connF = nengo.Connection(currState[3], bg.input[3], function = utilityF, learning_rule_type=nengo.PES())
    connD = nengo.Connection(currState[4], bg.input[4], function = utilityD, learning_rule_type=nengo.PES())
    connB = nengo.Connection(currState[5], bg.input[5], function = utilityB, learning_rule_type=nengo.PES())

    thal = nengo.networks.actionselection.Thalamus(6)
    nengo.Connection(bg.output,thal.input)
    
    #define a node "reward" that takes input from thalamus
    #we have to know the choice thal is making to use the right
    #probability of reward
    
    #all of this is so totally wrong
    #needs to be the probability that we return a value of 1 or 0   
    
    #draw six values (0 or 1) with the 6 probabilities, always
    #multiply the results of the draw with the thalamus choice
    
    #doesn't sum result into one value - could apply to each qvalue estimate
    #even when options aren't presented, their estimate (of 0) deviates a bit
    #most likely to internal noise, but could we use these tiny deviations
    #on examples where they aren't presented to learn on them? i.e. the idea
    #that the system always TRIES to learn about all options
    #could conceptualize as learning about non-present options relative to 
    #present options?
    def get_money(t, x):
        probs = [0.8, 0.7, 0.6, 0.4, 0.3, 0.2]
        results = []
        for i in probs:
            result = np.random.binomial(1,i)
            results.append(result)
        return np.multiply(results,x)
    
    #sums errors for ALL SIX OPTIONS (yes, even though only 2 are presented)
    def get_money_sum(t, x):
        probs = [0.8, 0.7, 0.6, 0.4, 0.3, 0.2]
        results = []
        for i in probs:
            result = np.random.binomial(1,i)
            results.append(result)
        return sum(np.multiply(results,x))
    
    
    
    reward = nengo.Node(get_money_sum,size_in=6,size_out=1)
    nengo.Connection(thal.output,reward)
    
    #calculate error between expected utility and reward
    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=6)
    nengo.Connection(reward,errors.input,transform=-np.ones((6,1)))
    
    #connect bg output directly to error ensemble neurons
    #not sure why we're doing this
    #I believe it's supposed to negate the error estimate of all options
    #we didn't choose (since BG is outputs -1 for those, and 0 for the selected)
    #transform replicates singleton bg output value to the # of neurons
    #not sure why it multiples by 4 - play with that
    nengo.Connection(bg.output[0],errors.ensembles[0].neurons, transform = np.ones((50,1))*4)
    nengo.Connection(bg.output[1],errors.ensembles[1].neurons, transform = np.ones((50,1))*4)
    nengo.Connection(bg.output[2],errors.ensembles[2].neurons, transform = np.ones((50,1))*4)
    nengo.Connection(bg.output[3],errors.ensembles[3].neurons, transform = np.ones((50,1))*4)
    nengo.Connection(bg.output[4],errors.ensembles[4].neurons, transform = np.ones((50,1))*4)
    nengo.Connection(bg.output[5],errors.ensembles[5].neurons, transform = np.ones((50,1))*4)
    
    #connect bg input (aka the estimate utility of each option) to the error
    #population
    #identity transform - we want the error as estimated by our initial estimate,
    #with the actual return subtracted (the flip transform between reward and
    #errors)(this seems to be modulated by the bg output, so we only estimate 
    #error for the chosen option)
    nengo.Connection(bg.input, errors.input, transform=1)
    
    #connect the error estimate to the connection between currState and BG
    #that computes the utility of the presented options
    nengo.Connection(errors.ensembles[0], connA.learning_rule)
    nengo.Connection(errors.ensembles[1], connC.learning_rule)
    nengo.Connection(errors.ensembles[2], connE.learning_rule)
    nengo.Connection(errors.ensembles[3], connF.learning_rule)
    nengo.Connection(errors.ensembles[4], connD.learning_rule)
    nengo.Connection(errors.ensembles[5], connB.learning_rule)
    
    
    
    

    
    
