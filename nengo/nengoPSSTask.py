#define PSS task to interact with nengo model
import random
import numpy as np
from collections import deque

#PSS_Object should be good
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
        

#good except for statevec maybe
class PSS_State(PSS_Object):
    """
A state in the PSS object. A state is consists of two possible options
to choose from, one on the left and one on the right.
    """
    #good
    def __init__(self, options = ("A", "B")):
        """Initializes a state, with default options being (A, B)"""
        if self.is_options(options):
            self.options = options
        else:
            self.options = None
    
    #good       
    @property
    def left(self):
        """The option on the left""" 
        if (self.is_options(self.options)):
            return self.options[0]
        else:
            return None
            
    #good
    @property
    def right(self):
        """The option on the right"""
        if (self.is_options(self.options)):
            return self.options[1]
        else:
            return None
    
    #this is good, but have to integrate into task.state and task.next_state()
    @property
    def statevec(self):
        #state space represented as vector
        # [A, C, E, F, D, B]
        # [0, 1, 0, 0, 1, 0] for (C,D)
        statevec = [0,0,0,0,0,0]
        statevec[self.ACTIONS.index(self.options[0])] = 1
        statevec[self.ACTIONS.index(self.options[1])] = 1
        return statevec
    
           
    #good
    def is_options(self, options):
        """Checks whether a given tuple is a set of options"""
        if len(options) == 2 and not False in [x in self.ACTIONS for x in options]:
            return True
        else:
            return False
    
    #good
    def __eq__(self, other):
        """Equality if the options are the same, independent of order"""
        return (self.left == other.left and self.right == other.right) or \
               (self.left == other.right and self.right == other.left)
    
    #good
    def __repr__(self):
        """Represented as a tuple '(O1, O2)'"""
        return "(%s,%s)" % (self.left, self.right)
    
    #good
    def __str__(self):
        return self.__repr__()
    
class PSS_Decision(PSS_Object):
    """A decision made during the PSS task"""
    #good
    def __init__(self, state = None, action = None, reward = 0.0):
        self.state = state
        self.action = action
        self.reward = reward
    
    #good
    def is_state(self, state):
        """Checks if something is a valid state"""
        return isinstance(state, PSS_State)
    
    #good
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
    #object implementing PSS task
    
    #good
    CRITERION = {"AB" : 0.65, "CD" : 0.60, "EF" : 0.50}
    
    #good
    TRAINING_BLOCK = ((("A", "B"),) * 10 +
                      (("B", "A"),) * 10 +
                      (("C", "D"),) * 10 +
                      (("D", "C"),) * 10 +
                      (("E", "F"),) * 10 +
                      (("F", "E"),) * 10)
    
    #good
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

                  
    
    #good
    PHASES = ("Training", "Test")
    
    def __init__(self):
        #inits a PSS task experiment
        self.index = 0
        self.phase = "Training"
        
        self.train = self.instantiate_block(self.TRAINING_BLOCK)
        self.test = self.instantiate_block(self.TEST_BLOCK)
        self.blocks = dict(zip(self.PHASES, [self.train, self.test]))
        self.history = dict(zip(self.PHASES, [[], []]))
        
        self.state = self.next_state() ###
        
    #good
    def instantiate_block(self, block):
        #inits and randomizes a block of trials
        trials = [PSS_State(x) for x in block]
        random.shuffle(trials)
        return deque(trials)
    
    def next_state(self):
        """Next state (transitions are independent of actions)"""
        state_next = None
        current_block = self.blocks[self.phase]
        if len(current_block) == 0:
            if self.phase == "Training":
                if self.criterion_reached() or len(self.history["Training"]) >= 360: ### execute_action() is where self.history is update - will have to integrate
                                                                                    #most likely integrate execute_action() in beteween thalamus and reward (aka it is the function the "reward"
                                                                                    #node performs, it accepts thal output as arg and returns reward)
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
        return state_next ###should try to convert this to return statevec
    
    def criterion_reached(self):
        """Reached criterion for successful learning"""
        training = self.history['Training']  ###
        if len(training) < 60:
            return False
        
        else:
            if len(training) > 60:
                training = training[-60:]
            ab = self.calculate_accuracy(training, "A") ###
            cd = self.calculate_accuracy(training, "C")
            ef = self.calculate_accuracy(training, "E")
            
            if ab >= self.CRITERION["AB"] and cd >= self.CRITERION["CD"] and ef >= self.CRITERION["EF"]:
                return True
            else:
                return False
            
            
    def calculate_accuracy(self, decisions, option, exclude = "None"):
        """Calculates accuracy across all decisions that include option 'option' but not option 'exclude'"""
        opt = [x.optimal for x in decisions if x.includes_option(option) and not x.includes_option(exclude)]
        return np.mean(opt)
    
    
    
    
    
    
    
    
    
    
###############################################################################

class PSS_Agent(PSS_Object):
    """An abstract agent. Can't do anything by itself, and needs to be setup properly"""
    def __init__(self):
        pass
    
    def run(self, task):
        """Runs with a PSS task object until the task is over (state == None)""" 
        while (task.state is not None):
            s_t1 = task.state ### - this is where it will have to report state to nengo
            return s_t1 #task.state doesn't return statevec, just the 2d state
            a_t1 = self.policy(s_t1) ### - this is where it will have to accept decision from nengo
            s_t2, r_t2 = task.execute_action(a_t1) #can still execute here maybe, and return reward to nengo?
            
            #most likely not needed
            if (r_t2 is not None):
                # Only learns if a reward signal has been given
                self.learn(s_t1, a_t1, r_t2)
    

  