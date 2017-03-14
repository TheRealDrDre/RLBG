import bgrl as b
import bgplots as bgp
import numpy as np

NUM_SIMS = 5000

#define just to satisfy agent definition
E = range(0, 21, 1)
E = [e/20.0 for e in E]  # Range of epsilon for e-greedy
e_results = {}

#what we actually use
T = range(1, 21, 1)
T = [t/10.0 for t in T]  # Range of temperatures for Gibbs distribution
T = [t**2 for t in T]
t_results = {}

for t in T:
    performance = []
    for i in range( NUM_SIMS ):
        task = b.PSS_Task()
        agent = b.RL_Agent(epsilon = E, alpha = 0.1, temperature = t)
        agent.policy = agent.gibbs_policy
        agent.run(task)
        performance.append(task.accuracies())

    t_results[t] = (np.mean([x[0] for x in performance]), np.mean([x[1] for x in performance]))

bgp.plot_results(t_results, T)