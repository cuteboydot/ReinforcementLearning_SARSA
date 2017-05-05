# ReinforcementLearning_SARSA
Implementation of ReinforcementLearning_SARSA.

cuteboydot@gmail.com

- reference : http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html

- example : cliff walking problem

<br>
<img height="200" src="https://github.com/cuteboydot/ReinforcementLearning_SARSA/blob/master/img/cliff.png" />
</br>
<br>
<img src="https://github.com/cuteboydot/ReinforcementLearning_SARSA/blob/master/img/statemap.png" />
</br>
  
s : start state  
c : cliff  
g : goal   
-1 : reward  
g : goal  
  
len(row) : 4, len(col) : 12  
row_start : 3, col_start : 0     
row_goal : 3, col_goal : 11   
action : 0(up), 1(left), 2(down), 3(left)  
  
mu : learning rate  
gamma : discount factor  
R : reward map  
A : possible action path  
Q : Q(s,a) value map  
action : 0(up), 1(left), 2(down), 3(left)  
  
Q(s,a) update : Q(s,a) <- Q(s,a) + mu(reward + gamma*Q(s',a') - Q(s,a))  
  
- test result
<br>
<img height="700" src="https://github.com/cuteboydot/ReinforcementLearning_SARSA/blob/master/img/test_result.png" />
</br>
- code  

```python  
import numpy as np

row = 4
col = 12
dir = 4
up = 0
right = 1
down = 2
left = 3

# states and policy definition
R = -np.ones((row, col))  # reward map
R[-1, 1:-1] = -1000
print('REWARD MAP')
print(R)
print()

A = np.ones((row, col, dir))  # action map
A[0, :, up] = 0
A[:, 0, left] = 0
A[:, -1, right] = 0
A[-1, :, down] = 0
print('POSSIBLE ACTION MAP')
print(A)
print()

Q = np.ones((row, col, dir))  # Q(s,a) func map
Q = np.random.random_sample(np.shape(Q)) * 0.03
Q[0, :, up] = -np.inf
Q[:, 0, left] = -np.inf
Q[:, -1, right] = -np.inf
Q[-1, :, down] = -np.inf
print('INITIAL Q(s,a) MAP')
print(Q)
print()

# start and goal
row_start = 3
col_start = 0
row_goal = 3
col_goal = 11

# retrieval parameter
epochs = 500
mu = 0.65
gamma = 0.4
epsilon = 0.05

def choose_action(R, A, Q, row, col):
    action_ret = -1

    # print (A)
    while action_ret == -1:
        # select action considering epsilon
        if (np.random.rand() < epsilon):
            action_ret = np.random.randint(dir)
        else:
            action_ret = np.argmax(Q[row][col])

        if (A[row][col][action_ret] == 0):
            action_ret = -1
    return action_ret


def get_next_pos(row, col, action) :
    row_next = 0
    col_next = 0
    
    # cliff case -> go to start
    if (row == 3 and (col > 0 and col < 11)) :
        return row_start, col_start
    
    if (action == 0) :          # move up
        row_next = row-1
        col_next = col
    elif (action == 1) :        # move right
        row_next = row
        col_next = col+1
    elif (action == 2) :        # move down
        row_next = row+1
        col_next = col
    else:                       # move left
        row_next = row
        col_next = col-1
    return row_next, col_next


def print_path_map(path) :

    print()

    steps = 0
    for ts in path :
        print('ts = %d' %(steps))
        steps += 1
        for a in range(row) :
            for b in range(col):
                if (a == ts[0] and b == ts[1]) :
                    print (' s\t' % (R[a][b]), end='')
                elif (R[a][b] == -1000) :
                    print (' c\t' % (R[a][b]), end='')
                elif (a == row_goal and b==col_goal) :
                    print (' g\t' % (R[a][b]), end='')
                else:
                    print (' %d\t' % (R[a][b]), end='')
            print()
        print()
    print()


def train(R, A, Q):
    print('TRAIN START!')

    for epoch in range(epochs):
        # time step
        ts = 0

        # state s
        row_cur = row_start
        col_cur = col_start

        # action a
        action_cur = choose_action(R, A, Q, row_cur, col_cur)

        # repeat episode
        ts += 1
        while ts :
            # reward r
            reward_cur = R[row_cur][col_cur]

            # state s'
            row_next, col_next = get_next_pos(row_cur, col_cur, action_cur)

            # action a'
            action_next = choose_action(R, A, Q, row_next, col_next)

            # reward r'
            reward_next = R[row_next][col_next]

            #print ('ts(%d), s[%d][%d] : a[%d] -> sp[%d][%d]' %(ts, row_cur, col_cur, action_cur, row_next, col_next))

            # update Q(s,a) <- Q'(s,a), s <- s', a <- a'
            Q[row_cur][col_cur][action_cur] += mu * (reward_cur + (gamma * Q[row_next][col_next][action_next]) - Q[row_cur][col_cur][action_cur])
            row_cur = row_next
            col_cur = col_next
            action_cur = action_next
            reward_cur = reward_next

            # find goal
            if (row_cur == row_goal and col_cur == col_goal) :
                #print ('GOAL REACHED!   epoch=%d, time steps=%d' % (epoch, ts))
                break
            ts += 1
        #print('TRAINED Q(s,a) MAP   epoch=%d' % (epoch))
        #print (Q)
    print()


def test(R, A, Q, print_path) :
    print('TEST START!')

    # time step
    ts = 0

    # total reward sum
    reward_tot = 0

    # path to goal
    path = []

    # state s
    row = row_start
    col = col_start
    path.append((row, col))

    while True :
        ts += 1

        # action a
        action = choose_action(R, A, Q, row, col)

        # reward r
        reward_cur = R[row][col]
        reward_tot += reward_cur

        # state s'
        row, col = get_next_pos(row, col, action)
        path.append((row, col))

        # find goal
        if (row == row_goal and col == col_goal) :
            print ('GOAL REACHED!   ts=%d, total reward=%d' % (ts, reward_tot))
            break

    if(print_path == True):
        print_path_map(path)

    print()


train(R, A, Q)
test(R, A, Q, True)
test(R, A, Q, False)
test(R, A, Q, False)
test(R, A, Q, False)

```
