# Notes

### What I know so far
* The benchmark policy is reflects the greedy route, so q-learning performs the best.
* Swarms of size 10 perform best.
* CoFEA works best on larger maps.
* Co-training provides better performance over PSO RL with the same amount of episodes.
* Q-learning and exp sarsa perform well together.
* Co-training not necessarily helpful on low iteration count
    * Co-training and non Co performed equally as well for e=100, i=300
    * Co-training outperforms Co for e=100 and i=3000
    * Co-training **** for e=1000 and i=3000
    * This could be because Co is highly erroneous to start due to pseudo-reward exchange,
    and only increases confidence in pseudo-reward assignment with certain amount of iterations.
* Increasing the particle count helps with non Co:
    * general observation shows that a 4x increase in map size warrants a square of particle size.
    
* Expected SARSA + Co-training benefit more from increased swarm populations.
___
Notes:
* Strasser's formalized definition of the FEA model extends the definition proposed by Haberman and 
Sheppard in [32] and Fortier, et al. in [34] with their model of overlapping swarm intelligence (OSI). 
* Overlapping Swarm Intelligence (OSI) is a version of FEA that uses particle swarm optimization (PSO)
to learn the weights of deep ANNs. See [33]. In this work, each swarm represents a unique path
starting at an input node and ending at an output node. A common vector of weights is also maintained
across all swarms to describe a global view of the network, which is created by combining the weights
of the best particles in each of the swarms. Pillai and Sheppard showed that OSI outperformed several
other PSO-based algorithms as well as standard back propagation in neural networks. (This last point
could be very interesting to study as an outside effort).
    * OSI seems like the best version of FEA to use for DOSPCoT.
    
* OSI was introduced in 2011 by Haberman and Sheppard [32]. It uses PSO as the underlying optimization
algorithm and works by creating multiple swarms that are assigned to overlapping subproblems. It was 
later extended in [33] by showing the effectiveness of OSI in producing energy-efficient routing protocols
for sensor networks.

* Distributed OSI (DOSI) was demonstrated by Fortier, et al. (2012) [35] in which swarms were allowed to 
communicate values to facilitate competition with one another. Butcher et al. (2018) [36] extended this work
by illustrating a concept of information sharing and conflict resolution that could be accomplished
through this communication through Pareto improvements.

* Do provide full statistical credibility to your study, you will need to evaluate SARSA, Q, and Expected
SARSA. However, given the scope of the project, it may not be feasible to do all 3. Start by evaluating 
Expected SARSA and give a good qualitative analysis. If there is time left over, then look at the other
two.

* Two main comparators for t-test: Expected SARSA + OSI and a Q-learner (perhaps DQN?). Still need to
figure out if neural networks should be used. A third comparator could be to assess non-overlapping
subregions (since OSI uses `overlapping` subregions).
___
* PSO + FEA (no co-training) results seem more repeatable than just PSO for learning a policy.