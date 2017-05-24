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
    
    def get_money(t, x):
        probs = [0.8, 0.7, 0.6, 0.4, 0.3, 0.2]
        results = []
        for i in range(len(probs)):
            if x[i]<0.2:
                results.append(0)
            else:
                result = np.random.binomial(1,probs[i])
                results.append(result)
        return sum(results)
    
    reward = nengo.Node(get_money,size_in=6,size_out=1)
    nengo.Connection(thal.output,reward)
    
    #calculate error between expected utility and reward
    #blasts each error pop with the reward - we need to inhibit the unchosen ones
    #can define get_money to return a 6d vector that has probability p of giving
    #reward for the chosen option, and 0 otherwise - then wouldn't have to inhibit
    #essentially giving no reward for "not choosing" the other options
    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=6)
    nengo.Connection(reward,errors.input,transform=-np.ones((6,1)))
    
    #cheat a bit and set some stimulus to keep them active
    inhbStim = nengo.Node([1])
    inhb = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=6)
    nengo.Connection(inhbStim,inhb.input, transform = np.ones((6,1)))
    
    #connect inhb to error pops so they can tonically inhibit them
    nengo.Connection(inhb.output[0],errors.ensembles[0].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb.output[1],errors.ensembles[1].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb.output[2],errors.ensembles[2].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb.output[3],errors.ensembles[3].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb.output[4],errors.ensembles[4].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb.output[5],errors.ensembles[5].neurons, transform = -np.ones((50,1))*4)

    #connect thalamus output to inhb ensembles - the chosen option will inhibit 
    #it's inhibition of learning
    #connecting BG output rather than thal output doesn't work for closely valued options
    nengo.Connection(thal.output[0],inhb.ensembles[0].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[1],inhb.ensembles[1].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[2],inhb.ensembles[2].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[3],inhb.ensembles[3].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[4],inhb.ensembles[4].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[5],inhb.ensembles[5].neurons, transform = -np.ones((50,1))*10)

    #connect bg input (aka the estimate utility of each option) to the error
    #population
    #identity transform - we want the error as estimated by our initial estimate,
    #with the actual return subtracted (the flip transform between reward and
    #errors)(this seems to be modulated by the bg output, so we only estimate 
    #error for the chosen option)
    #so, in Andrea's bgrl code, the update is Q(s,a) = Q(t1) + alpha(rew-Q(t))
    #the errors ensembles are performing the (rew-Q(t)) calculation
    #the input from the error ensemble to the connection between currState and
    #BG is updating the function performed there so that its result will be 
    #closer to Q(t1) + (rew-Q(t)) in the future
    #not sure how alpha plays in - we are connecting to the learning rule, 
    #not necessarily using neurotransmitters?
    nengo.Connection(bg.input, errors.input, transform=1)
    
    #connect the error estimate to the connection between currState and BG
    #that computes the utility of the presented options
    nengo.Connection(errors.ensembles[0], connA.learning_rule)
    nengo.Connection(errors.ensembles[1], connC.learning_rule)
    nengo.Connection(errors.ensembles[2], connE.learning_rule)
    nengo.Connection(errors.ensembles[3], connF.learning_rule)
    nengo.Connection(errors.ensembles[4], connD.learning_rule)
    nengo.Connection(errors.ensembles[5], connB.learning_rule)
    
    
    
    
    
    

    
    