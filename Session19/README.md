# Reinforcement Learning: MDP

### Assignment

You are given a 4x4 GridWorld where an agent starts at the top-left corner (state 0) and tries to reach the bottom-right corner (state 15). The agent can move up, down, left or right with equal probability. The rewards are -1 for each move, and the terminal state (bottom-right) has a reward 0. There are no obstacles. Your task is to:

> 1) initialize V(s) to 0 for all states

> 2) Iteratively apply the Bellman equation until convergence:
image.png
where P(s'|s, a) is the transition probability (equal for all moves)

> 3) Use gamma = 1 (no discounting)

> 4) Stop when maximum change in V(s) across all states is < 1e - 4


## Pseudo Code:

1) Initialize:

> 1) set grid size (NxN)

> 2) Define rewards for each state (-1 per move, 0 for terminal state)

> 3) initialize value function V(s) = 0 for each states

> 4) set discount factor (gamma) and convergence threshold (theta) 

2) Define possible actions: up, down, left, right.

3) Repeat until value converges:

> 1) track maximum changes in values

> 2) create a copy of current value function V_new

> 3) for each state s (excluding the terminal state):

>> 1) compute new value using the Bellman Equation:

>> 2) for each action, calculate:

>>> 1) next state s' (handling grid boundaries)

>>> 2) expected value update: sum over all possible s'

>> 3) update V_new(s)

>> 4) track max change (to see if converged)

> 4) Set V = V_new (update value function)

> 5) if threshold reached, then stop

4) Print the final value function. It would look something like this:

[[-59.42367735 -57.42387125 -54.2813141  -51.71012579]

 [-57.42387125 -54.56699476 -49.71029394 -45.13926711]

 [-54.2813141  -49.71029394 -40.85391609 -29.99766609]
 
 [-51.71012579 -45.13926711 -29.99766609   0.        ]]

5) Upload the Jupyter notebook to Gihub and share the link. Readme must show the final output like above.


## Final Output

```
Final Value Function:
[[-59.423676 -57.423866 -54.281315 -51.710125]
 [-57.423874 -54.566994 -49.710293 -45.139267]
 [-54.28131  -49.710293 -40.853916 -29.997665]
 [-51.71013  -45.139267 -29.997665   0.      ]]
```

## Code

```
import torch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GridWorld parameters
grid_size = 4
theta = 1e-4           # convergence threshold
gamma = 1.0            # discount factor (no discounting)

# Initialize value function V(s) = 0 for all states
V = torch.zeros((grid_size, grid_size), device=device)

# Create a boolean mask for terminal state (bottom-right corner)
terminal_mask = torch.zeros((grid_size, grid_size), dtype=torch.bool, device=device)
terminal_mask[-1, -1] = True  # terminal state at (3, 3)

delta = float('inf')

while delta > theta:
    # Create a new tensor for updated values
    # Instead of a Python loop over each state, we use tensor operations
    # to shift the grid in each direction and handle boundaries.
    
    # For upward movement: for row 0, stay in the same row; for others, use the row above.
    V_up = torch.cat([V[0:1, :], V[:-1, :]], dim=0)
    
    # For downward movement: for last row, stay in the same row; for others, use the row below.
    V_down = torch.cat([V[1:, :], V[-1:, :]], dim=0)
    
    # For left movement: for column 0, stay in the same column; for others, use the column to the left.
    V_left = torch.cat([V[:, 0:1], V[:, :-1]], dim=1)
    
    # For right movement: for last column, stay in the same column; for others, use the column to the right.
    V_right = torch.cat([V[:, 1:], V[:, -1:]], dim=1)
    
    # Each move yields a reward of -1 (except the terminal state which is fixed to 0).
    Q_up    = -1 + gamma * V_up
    Q_down  = -1 + gamma * V_down
    Q_left  = -1 + gamma * V_left
    Q_right = -1 + gamma * V_right

    # The new value is the average over all four actions.
    V_new = (Q_up + Q_down + Q_left + Q_right) / 4.0

    # For the terminal state, enforce V(terminal) = 0.
    V_new[terminal_mask] = 0.0

    # Check convergence: maximum change in value across states.
    delta = torch.max(torch.abs(V_new - V)).item()

    # Update the value function for the next iteration.
    V = V_new

# Print the final value function (move to CPU for numpy printing if needed)
print("Final Value Function:")
print(V.cpu().numpy())

```