import random
import copy
import math
import numpy as np
from collections import deque
import matplotlib

#comment to push

## ------------------------------------------------------------------------ ##
## The PSS Task 
## ------------------------------------------------------------------------ ##

class PSS_Object():
    """A generic object for PSS task components"""
    ACTIONS = ("A", "C", "E", "F", "D", "B")
    NEG_ACTIONS = tuple("-" + x for x in ACTIONS)
    REWARD_TABLE = {"A" : 0.8, "C" : 0.7, "E" : 0.6,
                    "F" : 0.4, "D" : 0.3, "B" : 0.2}

    def is_action(self, action):
        """An action is valid only if it belongs to the list of possible actions"""
        return action[-1] in self.ACTIONS
    
    def prob_reward(self, action):
        """Returns the probability of obtaining a reward given an action"""
        if self.is_action(action):
            return self.REWARD_TABLE[action]
        
    def get_reward(self, action):
        """Return a probabilistic reward associated with an action"""
        i = random.random()
        if i <= self.prob_reward(action):
            return 1.0
        else:
            return -1.0
    
    def complement_action(self, action):
        """Returns the complement action (i.e., -A for A, and A for -A)""" 
        if self.is_action(action):
            if action.startswith("-"):
                return action[-1]
            else:
                return "-" + action
        
class PSS_State(PSS_Object):
    """
A state in the PSS object. A state is consists of two possible options
to choose from, one on the left and one on the right.
    """
    def __init__(self, options = ("A", "B")):
        """Initializes a state, with default options being (A, B)"""
        if self.is_options(options):
            self.options = options
        else:
            self.options = None
            
    @property
    def left(self):
        """The option on the left""" 
        if (self.is_options(self.options)):
            return self.options[0]
        else:
            return None
            
    
    @property
    def right(self):
        """The option on the right"""
        if (self.is_options(self.options)):
            return self.options[1]
        else:
            return None
           

    def is_options(self, options):
        """Checks whether a given tuple is a set of options"""
        if len(options) == 2 and not False in [x in self.ACTIONS for x in options]:
            return True
        else:
            return False
    
    def __eq__(self, other):
        """Equality if the options are the same, independent of order"""
        return (self.left == other.left and self.right == other.right) or \
               (self.left == other.right and self.right == other.left)
    
    def __repr__(self):
        """Represented as a tuple '(O1, O2)'"""
        return "(%s,%s)" % (self.left, self.right)
    
    def __str__(self):
        return self.__repr__()
    
    
class PSS_Decision(PSS_Object):
    """A decision made during the PSS task"""
    def __init__(self, state = None, action = None, reward = 0.0):
        self.state = state
        self.action = action
        self.reward = reward
    
    def is_state(self, state):
        """Checks if something is a valid state"""
        return isinstance(state, PSS_State)
    
    @property
    def successful(self):
        """Success if reward > 0."""
        if self.reward > 0:
            return True
        else:
            return False
    
    @property
    def optimal(self):
        """A an action was optimal if it corresponded to the highest prob option"""
        s = self.state
        apos = s.options.index(self.action)
        probs = [self.prob_reward(x) for x in s.options]
        ppos = probs.index(max(probs))
        return apos == ppos
    
    def includes_option(self, option):
        """Checks if the decision included option 'option'"""
        return option in self.state.options
    
    
    def __repr__(self):
        """The decision as a string"""
        return "<%s, %s, %0.1f>" % (self.state, self.action, self.reward)



class PSS_Task(PSS_Object):
    """An object implementing the PSS task"""
    CRITERION = {"AB" : 0.65, "CD" : 0.60, "EF" : 0.50}
    
    TRAINING_BLOCK = ((("A", "B"),) * 10 +
                      (("B", "A"),) * 10 +
                      (("C", "D"),) * 10 +
                      (("D", "C"),) * 10 +
                      (("E", "F"),) * 10 +
                      (("F", "E"),) * 10)
    
    TEST_BLOCK = ((("A", "B"),) * 2 + (("B", "A"),) * 2 +
                  (("A", "C"),) * 2 + (("C", "A"),) * 2 +
                  (("A", "D"),) * 2 + (("D", "A"),) * 2 +
                  (("A", "E"),) * 2 + (("E", "A"),) * 2 +
                  (("A", "F"),) * 2 + (("F", "A"),) * 2 +

                  (("B", "C"),) * 2 + (("C", "B"),) * 2 +
                  (("B", "D"),) * 2 + (("D", "B"),) * 2 +
                  (("B", "E"),) * 2 + (("E", "B"),) * 2 +
                  (("B", "F"),) * 2 + (("F", "B"),) * 2 +
                  
                  (("C", "D"),) * 2 + (("D", "C"),) * 2 +
                  (("C", "E"),) * 2 + (("E", "C"),) * 2 +
                  (("C", "F"),) * 2 + (("F", "C"),) * 2 +
                  
                  (("D", "E"),) * 2 + (("E", "D"),) * 2 +
                  (("D", "F"),) * 2 + (("F", "D"),) * 2 +
    
                  (("E", "F"),) * 2 + (("F", "E"),) * 2)

                  
    
    PHASES = ("Training", "Test")
    
    def __init__(self):
        """Initializes a PSS task experiment"""
        self.index = 0
        self.phase = "Training"
        
        self.train = self.instantiate_block(self.TRAINING_BLOCK)        
        self.test =  self.instantiate_block(self.TEST_BLOCK)
        self.blocks = dict(zip(self.PHASES, [self.train, self.test]))                
        self.history = dict(zip(self.PHASES, [[], []]))
        
        self.state = self.next_state()
    
    def instantiate_block(self, block):
        """Instantiates and randomizes a block of trials"""
        trials = [PSS_State(x) for x in block]
        random.shuffle(trials)
        return deque(trials)
    
    def criterion_reached(self):
        """Reached criterion for successful learning"""
        training = self.history['Training']
        if len(training) < 60:
            return False
        
        else:
            if len(training) > 60:
                training = training[-60:]
            ab = self.calculate_accuracy(training, "A")
            cd = self.calculate_accuracy(training, "C")
            ef = self.calculate_accuracy(training, "E")
            
            if ab >= self.CRITERION["AB"] and cd >= self.CRITERION["CD"] and ef >= self.CRITERION["EF"]:
                return True
            else:
                return False
    
    def next_state(self):
        """Next state (transitions are independent of actions)"""
        state_next = None
        current_block = self.blocks[self.phase]
        if len(current_block) == 0:
            if self.phase == "Training":
                if self.criterion_reached() or len(self.history["Training"]) >= 360:
                    # Move to the Test phase and recalculate the current block.
                    self.phase = "Test"
                      
                else:
                    self.blocks["Training"] = self.instantiate_block(self.TRAINING_BLOCK)
                    
                current_block = self.blocks[self.phase]
                state_next = current_block.popleft()
            
            else: 
                state_next = None # End of the experiment
        else:
            state_next = current_block.popleft()
        return state_next
                    
    
    def execute_action(self, action):
        """Executes and action and returns the new state and a reward"""
        if self.is_action(action):
            if action.startswith("-"):
                # This handles the cases where an agent chooses NOT
                # to pick a specific action (as in the BG models)
                action = [x for x in self.state.options if x is not action[-1]][0]
            
            r = None
            if self.phase is "Training":
                r = self.get_reward(action)
            
            # Update history
            d = PSS_Decision(self.state, action, reward = r)
            self.history[self.phase].append(d)
            
            self.state = self.next_state()
            return (self.state, r)
    
    def calculate_accuracy(self, decisions, option, exclude = "None"):
        """Calculates accuracy across all decisions that include option 'option' but not option 'exclude'"""
        opt = [x.optimal for x in decisions if x.includes_option(option) and not x.includes_option(exclude)]
        return np.mean(opt)
        
    def accuracies(self):
        """Returns the Choose / Avoid accuracies"""
        test = self.history["Test"]
        if len(test) >= 60:
            return (self.calculate_accuracy(test, option = 'A', exclude = 'B'),
                    self.calculate_accuracy(test, option = 'B', exclude = 'A'))
                    
            
            
class PSS_Agent(PSS_Object):
    """An abstract agent. Can't do anything by itself, and needs to be setup properly"""
    def __init__(self):
        pass
    
    def run(self, task):
        """Runs with a PSS task object until the task is over (state == None)""" 
        while (task.state is not None):
            method = self.method
            s_t1 = task.state
            a_t1 = self.policy(s_t1)
            s_t2, r_t2 = task.execute_action(a_t1[0])
            if (r_t2 is not None and len(a_t1) is 1): #would prefer "and self.policy is not RL_Agent.actor", but don't know how to check bound methods
                # Only learns if a reward signal has been given
                self.learn(s_t1, a_t1, r_t2)
            elif (r_t2 is not None):
                self.criticize(s_t1, a_t1, r_t2, method)
    
    def epsilon_greedy_policy(self, state):
        """The e-greedy policy"""
        qactions = self.identify_actions(state)
        i = random.random()
        if i <= self.epsilon:
            return random.choice(list(qactions.keys()))
        else:
            #print("i = %s, Best option chosen" % i)
            qmax = max(qactions.values())
            qsubset = [x for x in qactions.keys() if qactions[x] == qmax] 
            return random.choice(qsubset)
        
    def gibbs_policy(self, state):
        """A noisy policy based on Gibbs distribution"""
        qactions = self.identify_actions(state)
        actions = qactions.keys()
        qvals = [qactions[x] for x in actions]
        t = self.temperature
        #print([math.exp(x/t) for x in qvals])
        total = sum([math.exp(x/t) for x in qvals])
        pvals = [math.exp(x/t)/total for x in qvals]
        cumulative = [sum(pvals[0:i]) for i in range(1, len(pvals) + 1)]
        pactions = zip(actions, cumulative)
        ## Now, selection.
        j = random.random()
        above_set = [x for x in pactions if x[1] > j]
        return above_set[0][0]
    
    def actor(self,state):
        """A noisy actor (policy) for an actor-critic model
            Returns action and probability of pick as tuple"""
        as_actions = self.id_ass_strs(state)
        actions = as_actions.keys()
        ass_strs = [as_actions[x] for x in actions]
        total = sum([math.exp(x) for x in ass_strs])
        pvals = [math.exp(x)/total for x in ass_strs]
        cumulative = [sum(pvals[0:i]) for i in range(1, len(pvals)+1)]
        pactions = zip(actions,cumulative)
        j = random.random()
        above_set = [x for x in pactions if x[1] > j]
        if len(above_set) is 2:
            return above_set[0]
        else:
            return (above_set[0][0], pvals[1])
    
       
    
class RL_Agent(PSS_Agent):
    """A Q-Agent that uses a single-state representation and an action filter"""
    def __init__(self, alpha = 0.2, epsilon = 0.1, gamma = 0.9, temperature = 0.1): 
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.temperature = temperature
        self.Q = dict(zip([(PSS_State, x) for x in self.ACTIONS], 
                          [0.0] * len(self.ACTIONS)))
        self.ass_str = dict(zip([(PSS_State, x) for x in self.ACTIONS], 
                          [0.0] * len(self.ACTIONS)))
        self.policy = self.epsilon_greedy_policy
        self.method = "simple"
    
    
    def identify_actions(self, state):
        """Returns a list of plausible actions and their Q-values"""
        possible = [(x[1], self.Q[x]) for x in self.Q if x[1] in state.options]
        return dict(possible)
    
    
    def id_ass_strs(self,state):
        """Returns a list of plausible actions and their association strengths"""
        possible = [(x[1], self.ass_str[x]) for x in self.ass_str if x[1] in state.options]
        return dict(possible)
    
    
    def learn(self, s_t1, a_t1, r_t2):
        """Assuming every state is independent, d_t = R_t - Q_(t-1)"""
        state = PSS_State
        q_t1 = self.Q.get((state, a_t1), None)         
        if q_t1 is None:             
            self.Q[(state, a_t1)] = r_t2
        else:             
            self.Q[(state, a_t1)] = q_t1 + self.alpha * (r_t2 - q_t1)
            
            
    def criticize(self, s_t1, a_t1, r_t2, method):
        """Assuming every state is independent, d_t = R_t - Q_(t-1)
        Defaults to using the simple method of updating association
        strength"""
        
        if method is "simple":
            ap = 0
        else:
            ap = a_t1[1]
            
        state = PSS_State
        q_t1 = self.Q.get((state, a_t1[0]), None)
        as_t1 = self.ass_str.get((state, a_t1[0]), None) 
        
        #Use this to update the policy as well as the value function
        dq = r_t2 - q_t1
        
        if q_t1 is None:             
            self.Q[(state, a_t1[0])] = r_t2
        else:             
            self.Q[(state, a_t1[0])] = q_t1 + self.alpha*dq
            
        #can this be condensed into the if statement above? I'm assuming the if
        #statement checks if it is the first  "trial", and just inputs the
        #reward. If so, it can be condensed.
        
        #if ap is 1, it is the simple solution - using dq as an error term for 
        #the associative strength            
        #if ap is the actual probability of the action (returned by 
        #RL_Agent.actor(state) this solution uses dq as a signal to convey how 
        #much learning is needed
        #(how much the policy should be adapted?)
        if as_t1 is None:
            self.ass_str[(state, a_t1)] = 0
        else:
            self.ass_str[(state, a_t1)] = as_t1 + (self.alpha*dq)*(1-ap)


            
        
## ---------------------------------------------------------------- ##
## Biologically-plausible, BG-inspired model-free RL agents
## ---------------------------------------------------------------- ##

class RL_AgentBG(RL_Agent):
    """A BG-inspired agent"""
    
    def identify_actions(self, state):
        """Returns a list of plausible actions and their Q-values"""
        possible = [(x[1], self.Q[x]) for x in self.Q if x[1][-1] in state.options]
        return dict(possible)

    def __init__(self, alpha = 0.2, epsilon = 0.1, gamma = 0.9, temperature = 0.1): 
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.temperature = temperature
        
        self.Q = dict( zip([(PSS_State, x) for x in self.ACTIONS + self.NEG_ACTIONS], 
                           [0.0] * len(self.ACTIONS + self.NEG_ACTIONS)))
        self.policy = self.epsilon_greedy_policy

        
class RL_AgentBG_Anticorrelated(RL_Agent):
    """A BG-inspired agent with anticorrelated pathways"""
    
    def identify_actions(self, state):
        """Returns a list of plausible actions and their Q-values"""
        possible = [(x[1], self.Q[x]) for x in self.Q if x[1][-1] in state.options]
        return dict(possible)

    def __init__(self, alpha = 0.2, epsilon = 0.1, gamma = 0.9, temperature = 0.1): 
        """Inits"""
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.temperature = temperature
        
        self.Q = dict( zip([(PSS_State, x) for x in self.ACTIONS + self.NEG_ACTIONS], 
                           [0.0] * len(self.ACTIONS + self.NEG_ACTIONS)))
        self.policy = self.epsilon_greedy_policy
        

    def learn(self, s_t1, a_t1, r_t2):
        """
Assuming every state is independent, dQ_t = R_t - Q_(t-1).
Then, updates one pathway as Q_t = Q_(t-1) + dQ_t, and the
other as Q_t = Q_(t-1) - dQ_t.
        """
        state = PSS_State
        q_t1 = self.Q.get((state, a_t1), None)
        
        # Create values for the other pathway
        neg_a_t1 = self.complement_action(a_t1)
        neg_q_t1 = self.Q.get((state, neg_a_t1), None)
        
        # Update first pathway
        if q_t1 is None:             
            self.Q[(state, a_t1)] = r_t2
        else:
            self.Q[(state, a_t1)] = q_t1 + self.alpha * (r_t2 - q_t1)
            
        # Update other pathway
        if neg_q_t1 is None:
            self.Q[(state, neg_a_t1)] = -1 * r_t2
        else:             
            self.Q[(state, neg_a_t1)] = neg_q_t1 - self.alpha * (r_t2 - neg_q_t1)

## ---------------------------------------------------------------- ##
## Actor-Critic versions of the agents
## ---------------------------------------------------------------- ##

class ActorCritic_Agent(PSS_Agent):
    """An agent that learns in actor-critic fashion"""
    pass
