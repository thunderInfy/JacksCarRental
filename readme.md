This repository is my attempt to reproduce the solutions to Jack's Car Rental problem and its variant as mentioned in Example 4.2 and Exercise 4.3 respectively of the book by Sutton and Barto (Reinforcement Learning: An Introduction, Second Edition).

The original Jack's car rental problem is as follows:

Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars. If Jack has a car available, he rents it out and is credited $10 by the national company. If he is out of cars at that location, then the business is lost. Cars become available for renting the day after they are returned. To help ensure that
cars are available where they are needed, Jack can move them between the two locations overnight, at a cost of $2 per car moved. We assume that the number of cars requested and returned at each location are Poisson random variables. Suppose λ (parameter for poisson process) is 3 and 4 for rental requests at the first and second locations and 3 and 2 for returns. To simplify the problem slightly, we assume that there can be no more than 20 cars at each location (any additional cars are returned to the nationwide company, and thus disappear from the problem) and a maximum of five cars can be moved from one location to the other in one night. We take the discount rate to be γ = 0.9 and formulate this as a continuing finite MDP, where the time steps are days, the state is the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved between the two locations overnight.

Some Solution details:
The initial policy is 0 for each state, i.e. initially no cars are moved between either locations. The resulting policy and value graphs are moved to the folder jcrp_graphs. policy1 is the policy we get after one step of policy evaluation and iteration on the initial policy. 

And its variant is the following:

One of Jack’s employees at the first location rides a bus home each night and lives near the second location. She is happy to shuttle one car to the second location for free. Each additional car still costs $2, as do all cars moved in the other direction. In addition, Jack has limited parking space at each location. If more than 10 cars are kept overnight at a location (after any moving of cars), then an additional cost of $4 must be incurred to use a second parking lot (independent of how many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often occur in real problems and cannot easily be handled by optimization methods other than dynamic programming.

The resulting policy and value graphs are moved to the folder jcrp_variant_graphs.

The policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration. One important special case is when policy evaluation is stopped after just one sweep (one update of each state). This algorithm is called value iteration. Output policy and value graphs for value iteration are moved to the folder value_iteration_graphs.

I wrote a blog explaining my solution to this problem. Its link is:
https://towardsdatascience.com/elucidating-policy-iteration-in-reinforcement-learning-jacks-car-rental-problem-d41b34c8aec7