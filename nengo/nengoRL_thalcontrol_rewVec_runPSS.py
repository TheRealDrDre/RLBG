import PSSTask as pt
import nengo
import numpy as np

#state = pt.PSS_State(('A','B'))

task = pt.PSS_Task()

#print task.state

model = nengo.Network()
with model:
    #think about how to use SPA to represent these
    
    def make_statevec(t):
        statevec = [0,0,0,0,0,0]
        statevec[task.ACTIONS.index(task.state.options[0])] = 1
        statevec[task.ACTIONS.index(task.state.options[1])] = 1
        return statevec
    
    stim = nengo.Node(make_statevec)
    
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
        return results
    
    #reward = nengo.Node(get_money,size_in=6,size_out=6)
    #nengo.Connection(thal.output,reward)
    
    def execute_action(t,thalvec):
        #executes actions and returns the new state and reward
        a,c,e,f,d,b = thalvec
        actionvec = [a,c,e,f,d,b]
        
        #convert thalamic input (6d vector) to ...
        #chosen = [actionvec.index(i) for i in actionvec if i >= 0.2])
        
        chosen = actionvec.index(max(actionvec))
        
        action = task.ACTIONS[0]
        
        if task.is_action(action):
            if action.startswith('-'):
                #this handles the cases where agent chooses NOT to pick
                #a specific action
                action = [x for x in task.state.options if x is not action[-1]][0]
                
            r = None
            r = task.get_reward(action)
            
            #currently learns through training and test phases - if it didn't
            #get reward in test phase, would begin to lower estimates of options
            #if task.phase is "Training":
            #    r = task.get_reward(action)
                
            #update history
            #d = PSS_Decision(task.state, action, reward = r)
            #task.history[task.phase].append(d)
            
            task.state = task.next_state()
            print task.state
            rewVec = [0,0,0,0,0,0]
            rewVec[chosen] = r
            return (rewVec) ### here we return just the reward I think - calling next_state in the line above will update the current state.
                                    #but, this update might happen too quickly, in which case the reward would be subtracted from the utility estimates of the new state
    
    
    reward = nengo.Node(execute_action,size_in=6)
    nengo.Connection(thal.output,reward)
    
    #calculate error between expected utility and reward
    #get_money returns a reward of 0 for the unchosen option - if it's never 
    #chosen, estimate of utility of the unchosen quickly goes to 0
    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=6)
    nengo.Connection(reward,errors.input,transform=-1)
    
    
    #don't need any inhibition stuff (kinda)
    #if we wanted to hold our expectations of the unchosen option static (i.e.,
    #assume that not choosing the option returns no information about it, rather
    #than assuming that not choosing it is a choice we make alongside choosing
    #the option we did, and the total return was reward(chosen)+0(unchosen))
    
    #can also assume that there is some maximum reward that is always allocated 
    #across available options - when we receive a reward, subtract it from the
    #maximum reward, and split the difference amongst unchosen options
    
    #cheat a bit and set some stimulus to keep them active
    #inhbStim = nengo.Node([1])
    #inhb = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=6)
    #nengo.Connection(inhbStim,inhb.input, transform = np.ones((6,1)))
    
    #connect inhb to error pops so they can tonically inhibit them
    #nengo.Connection(inhb.output[0],errors.ensembles[0].neurons, transform = -np.ones((50,1))*4)
    #nengo.Connection(inhb.output[1],errors.ensembles[1].neurons, transform = -np.ones((50,1))*4)
    #nengo.Connection(inhb.output[2],errors.ensembles[2].neurons, transform = -np.ones((50,1))*4)
    #nengo.Connection(inhb.output[3],errors.ensembles[3].neurons, transform = -np.ones((50,1))*4)
    #nengo.Connection(inhb.output[4],errors.ensembles[4].neurons, transform = -np.ones((50,1))*4)
    #nengo.Connection(inhb.output[5],errors.ensembles[5].neurons, transform = -np.ones((50,1))*4)

    #connect thalamus output to inhb ensembles - the chosen option will inhibit 
    #it's inhibition of learning
    #connecting BG output rather than thal output doesn't work for closely valued options
    #nengo.Connection(thal.output[0],inhb.ensembles[0].neurons, transform = -np.ones((50,1))*10)
    #nengo.Connection(thal.output[1],inhb.ensembles[1].neurons, transform = -np.ones((50,1))*10)
    #nengo.Connection(thal.output[2],inhb.ensembles[2].neurons, transform = -np.ones((50,1))*10)
    #nengo.Connection(thal.output[3],inhb.ensembles[3].neurons, transform = -np.ones((50,1))*10)
    #nengo.Connection(thal.output[4],inhb.ensembles[4].neurons, transform = -np.ones((50,1))*10)
    #nengo.Connection(thal.output[5],inhb.ensembles[5].neurons, transform = -np.ones((50,1))*10)

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
    
    
    
    
    
    

    
    
