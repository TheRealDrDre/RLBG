import PSSTask as pt
import nengo
import numpy as np

#https://studywolf.wordpress.com/2012/12/09/nengo-model-low-pass-derivative-filter/
#https://studywolf.wordpress.com/2012/11/19/nengo-scripting-absolute-value/

#https://pythonhosted.org/nengo/examples/tuning_curves.html

#state = pt.PSS_State(('A','B'))

task = pt.PSS_Task()

model = nengo.Network()
with model:
    
    tau=0.1

    
    #think about how to use SPA to represent these
    def make_statevec(t):
        statevec = [0,0,0,0,0,0]
        statevec[task.ACTIONS.index(task.state.options[0])] = 1
        statevec[task.ACTIONS.index(task.state.options[1])] = 1
        return statevec
    
    stim = nengo.Node(make_statevec)
    #stim = nengo.Node(state.statevec)
    
    currState = nengo.Ensemble(n_neurons = 600, dimensions = 6, radius = 1.4)
    #currState = nengo.Ensemble(n_neurons = 700, dimensions = 7, radius = 2.6)

    nengo.Connection(stim, currState)
    
    
    #still does not allow state evidence to accumulate fast enough
    #do same trick as relay - only respond when above a certain value
    
    relay1 = nengo.Ensemble(n_neurons=75,
                           dimensions=1,
                           radius=2,
                           intercepts=nengo.dists.Exponential(0.3,0.5,1.),
                           encoders=nengo.dists.Choice([[1]]),
                           eval_points=nengo.dists.Uniform(0.5,1.))
                           
    nengo.Connection(currState,relay1,function=sum,synapse=tau)
    
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
    
    inhb1Stim = nengo.Node([1])
    inhb1 = nengo.Ensemble(n_neurons=50,dimensions=1)
    nengo.Connection(inhb1Stim,inhb1)

    relay2 = nengo.Ensemble(n_neurons=75,
                           dimensions=1,
                           radius=1,
                           intercepts=nengo.dists.Exponential(0.3,0.75,1.),
                           encoders=nengo.dists.Choice([[1]]),
                           eval_points=nengo.dists.Uniform(0.75,1.))
    
    def subone(x):
        return x-1
        
    nengo.Connection(relay1,relay2,function=subone)
    
    nengo.Connection(relay2,inhb1.neurons,transform=-np.ones((50,1))*4)

    nengo.Connection(inhb1,thal.actions.ensembles[0].neurons,transform=-np.ones((50,1)))
    nengo.Connection(inhb1,thal.actions.ensembles[1].neurons,transform=-np.ones((50,1)))
    nengo.Connection(inhb1,thal.actions.ensembles[2].neurons,transform=-np.ones((50,1)))
    nengo.Connection(inhb1,thal.actions.ensembles[3].neurons,transform=-np.ones((50,1)))
    nengo.Connection(inhb1,thal.actions.ensembles[4].neurons,transform=-np.ones((50,1)))
    nengo.Connection(inhb1,thal.actions.ensembles[5].neurons,transform=-np.ones((50,1)))
    
    #define a node "reward" that takes input from thalamus
    #we have to know the choice thal is making to use the right
    #probability of reward

    #I think a solution to the "rapid decision" problem is to have thalamus 
    #"wait" until currState is "sure" of state to return decision
    #so, it always outputs 0 on all dimensions, unless currState has "settled"
    #on a state - then it will output the decision
    
    
    #Okay, so it doesn't make decisions when thalamus is at zero any more
    #but as soon as it makes a decision and thalamus deflects, it will 
    #rapid-fire change states, making the same decision over and over again
    #(because thalamus is still deflected, choosing for the original state
    
    #state change only after thalamus has been brought to zero
    #inhibit relay as part of decision make? i.e. connect inhb2 to relay1
    #still switches states too fast, making same choice for multiple states
    #before network can settle
    
    #so: make choice, remember choice while allowing network to settle,
    #only when network is settled do we allow memory of choice to be executed
    #and state to change
    
    def execute_action(t,thalvec):
        #executes actions and returns the new state and reward
        a,c,e,f,d,b = thalvec
        actionvec = [a,c,e,f,d,b]
        
        #convert thalamic input (6d vector) to ...
        #chosen = [actionvec.index(i) for i in actionvec if i >= 0.2])
        
        chosen = actionvec.index(max(actionvec))
        
        #this isn't the greatest, as it is returning zero reward while not
        #choosing and just not learning from it
        if actionvec[chosen] < 0.5:
            rewVec = [0,0,0,0,0,0]
            return rewVec
        else:
            action = task.ACTIONS[chosen]
        
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
                print action
                rewVec = [0,0,0,0,0,0]
                rewVec[chosen] = r
                return (rewVec) ### here we return just the reward I think - calling next_state in the line above will update the current state.
                                        #but, this update might happen too quickly, in which case the reward would be subtracted from the utility estimates of the new state
    
    reward = nengo.Node(execute_action,size_in=6)
    nengo.Connection(thal.output,reward)
    
    #calculate error between expected utility and reward
    #blasts each error pop with the reward - we need to inhibit the unchosen ones
    #can define get_money to return a 6d vector that has probability p of giving
    #reward for the chosen option, and 0 otherwise - then wouldn't have to inhibit
    #essentially giving no reward for "not choosing" the other options
    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=6)
    nengo.Connection(reward,errors.input,transform=-1)
    
    inhbrelay1 = nengo.Ensemble(n_neurons=75,dimensions=6)
    nengo.Connection(thal.output,inhbrelay1)
    nengo.Connection(inhbrelay1,relay1.neurons,function=sum,transform=-np.ones((75,1)))
    nengo.Connection(inhbrelay1,currState.neurons,function=sum,transform=-np.ones((600,1)))

    
    #cheat a bit and set some stimulus to keep them active
    inhb2Stim = nengo.Node([1])
    inhb2 = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=6)
    nengo.Connection(inhb2Stim,inhb2.input, transform = np.ones((6,1)))
    
    #connect inhb to error pops so they can tonically inhibit them
    nengo.Connection(inhb2.output[0],errors.ensembles[0].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb2.output[1],errors.ensembles[1].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb2.output[2],errors.ensembles[2].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb2.output[3],errors.ensembles[3].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb2.output[4],errors.ensembles[4].neurons, transform = -np.ones((50,1))*4)
    nengo.Connection(inhb2.output[5],errors.ensembles[5].neurons, transform = -np.ones((50,1))*4)
    


    #connect thalamus output to inhb ensembles - the chosen option will inhibit 
    #it's inhibition of learning
    #connecting BG output rather than thal output doesn't work for closely valued options
    nengo.Connection(thal.output[0],inhb2.ensembles[0].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[1],inhb2.ensembles[1].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[2],inhb2.ensembles[2].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[3],inhb2.ensembles[3].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[4],inhb2.ensembles[4].neurons, transform = -np.ones((50,1))*10)
    nengo.Connection(thal.output[5],inhb2.ensembles[5].neurons, transform = -np.ones((50,1))*10)

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
    
    
    
    
    
    

    
    
