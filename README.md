# ReinforcementLearning_SARSA
Implementation of ReinforcementLearning_SARSA.

cuteboydot@gmail.com

reference : http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html

example : cliff walking problem


<br>
<img src="https://github.com/cuteboydot/ReinforcementLearning_SARSA/blob/master/statemap.png" />
</br>
s : start state
c : cliff
g : goal
-1 : reward

mu : learning rate
gamma : discount factor
R : reward map
A : possible action path
Q : Q(s,a) value map
action : 0(up), 1(left), 2(down), 3(left)

Q(s,a) update : Q(s,a) <- Q(s,a) + mu(reward + gamma*Q(s',a') - Q(s,a))

