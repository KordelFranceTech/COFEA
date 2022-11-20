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
    
    
    