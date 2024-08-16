Here is a pseudocode explanation based on the `__init__`, `getQValue`, `computeValueFromQValues`, `computeActionFromQValues`, `getAction`, and `update` functions from the Q-Learning Agent in the provided files:

### 1. **`__init__`** (Initialization)
- **Description**: This is the constructor for the Q-learning agent, responsible for initializing the agent's parameters and structures.
  
```plaintext
FUNCTION __init__(**args):
    CALL parent class constructor to initialize epsilon, alpha, and discount rates
    
    CREATE an empty dictionary (Counter) called stateActionPair to store Q-values
    # This stores Q-values for state-action pairs, initialized to 0.
    
    END FUNCTION
```

### 2. **`getQValue`** (Retrieve Q-value for a State-Action Pair)
- **Description**: This function returns the Q-value for a given state-action pair. If the state-action pair has never been seen, the default value of 0 is returned.
  
```plaintext
FUNCTION getQValue(state, action):
    LOOKUP the Q-value for the given (state, action) pair in the stateActionPair dictionary
    IF the (state, action) pair is not found, return 0 (default behavior of Counter)
    RETURN the Q-value

    END FUNCTION
```

### 3. **`computeValueFromQValues`** (Find Maximum Q-value for a State)
- **Description**: This function computes the value of a state as the maximum Q-value for all possible legal actions from that state. If no legal actions are available (e.g., terminal state), it returns 0.

```plaintext
FUNCTION computeValueFromQValues(state):
    GET the list of legal actions for the given state
    IF there are no legal actions:
        RETURN 0 (value of the terminal state)
    
    INITIALIZE maxValue to negative infinity
    
    FOR each legal action:
        GET the Q-value of the (state, action) pair
        UPDATE maxValue to the maximum Q-value observed so far
    
    RETURN maxValue

    END FUNCTION
```

### 4. **`computeActionFromQValues`** (Determine the Best Action from Q-values)
- **Description**: This function finds the action that has the highest Q-value for a given state. If multiple actions have the same Q-value, it randomly chooses among them. If no legal actions are available, it returns `None`.

```plaintext
FUNCTION computeActionFromQValues(state):
    GET the list of legal actions for the given state
    IF there are no legal actions:
        RETURN None (indicating no action available for terminal state)
    
    FIND the maximum Q-value across all legal actions
    
    CREATE a list of possible actions that have the maximum Q-value
    
    RANDOMLY choose and RETURN one of the actions with the maximum Q-value

    END FUNCTION
```

### 5. **`getAction`** (Choose an Action with Exploration)
- **Description**: This function chooses the next action the agent will take in the current state. With probability `epsilon`, it chooses a random action (exploration). Otherwise, it chooses the best action based on the Q-values (exploitation).

```plaintext
FUNCTION getAction(state):
    GET the list of legal actions for the given state
    
    IF no legal actions are available:
        RETURN None
    
    WITH probability epsilon:
        RETURN a random action from the list of legal actions
    
    ELSE:
        RETURN the best action according to computeActionFromQValues(state)

    END FUNCTION
```

### 6. **`update`** (Update Q-values Based on State Transitions)
- **Description**: This function updates the Q-value for a state-action pair based on the reward received and the estimated value of the next state (using the Q-learning update rule).

```plaintext
FUNCTION update(state, action, nextState, reward):
    GET the current Q-value for the (state, action) pair (qThis)
    GET the value of the nextState using computeValueFromQValues(nextState) (qNext)
    
    COMPUTE the updated Q-value using the formula:
        newQValue = (1 - alpha) * qThis + alpha * (reward + discount * qNext)
    
    UPDATE the Q-value for (state, action) in stateActionPair with the newQValue

    END FUNCTION
```

### Summary of Functionality:
- **`__init__`**: Initializes the agent, including setting up the Q-value storage and parameters.
- **`getQValue`**: Returns the Q-value for a given state-action pair.
- **`computeValueFromQValues`**: Computes the value of a state as the maximum Q-value over all possible actions.
- **`computeActionFromQValues`**: Determines the best action to take in a state based on the Q-values.
- **`getAction`**: Chooses an action, balancing exploration and exploitation based on epsilon.
- **`update`**: Updates the Q-value for a state-action pair using the Q-learning update rule.





### Simplified Overall Flowchart & Pseudocode Explanation

These functions together implement a **Q-Learning Agent** that interacts with an environment (e.g., Pacman game) to learn an optimal policy by updating its estimates of Q-values for state-action pairs. Here's how they work together in a simplified flowchart and pseudocode:

#### **Flowchart Overview**
The agent goes through the following cycle:
1. **Initialize** the Q-learning agent.
2. **Get Current Q-value** for the state-action pair.
3. **Choose Action** based on exploration or exploitation.
4. **Take Action** in the environment.
5. **Observe Reward and Next State**.
6. **Update Q-values** based on the experience.
7. **Repeat** the process for each step in the game or episode.

#### **Flowchart Diagram**

```plaintext
           ┌───────────────────────────┐
           │ Initialize Q-Learning Agent│
           └────────────┬───────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │ Get Current State (state)    │
         └────────────┬────────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │ Choose Action (Explore/Best)│
          └────────────┬───────────────┘
                        │
                        ▼
             ┌────────────────────────┐
             │ Take Action in Game     │
             └────────────┬───────────┘
                          │
                          ▼
        ┌────────────────────────────────┐
        │ Observe Reward and Next State   │
        └─────────────┬──────────────────┘
                          │
                          ▼
          ┌──────────────────────────────────┐
          │ Update Q-Values Using Q-Learning  │
          └───────────────┬──────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Loop Until Episode Ends│
              └───────────────────────┘
```

### Pseudocode Overview

The agent interacts with the environment to learn from its experiences. Below is a high-level pseudocode representation.

```plaintext
INITIALIZE the Q-learning agent
    Initialize Q-value table (state-action pairs) to zero

WHILE game is running (or until the episode ends):
    SET current_state ← get current state from environment
    
    CHOOSE action:
        WITH probability epsilon (exploration):
            action ← random action
        OTHERWISE (exploitation):
            action ← best action based on Q-values (max Q-value)
    
    TAKE action in the environment
    
    OBSERVE reward and next_state from the environment after taking action
    
    UPDATE Q-values:
        q_this ← Q-value for (current_state, action)
        q_next ← max Q-value for next_state (based on next state's best action)
        
        new Q-value for (current_state, action) ← (1 - alpha) * q_this + alpha * (reward + discount * q_next)
        
        STORE the new Q-value for (current_state, action)
    
    UPDATE current_state ← next_state

REPEAT until the end of the game or episode
```

### Simplified Diagram of Q-Learning Process

Here is a visual representation of how the Q-learning process works:

```plaintext
 ┌─────────────────────────────────────────────────────┐
 │                     Environment                     │
 │   ┌─────────────────────────────────────────────┐   │
 │   │           Q-Learning Agent                   │   │
 │   │                                             │   │
 │   │   1. Initialize Q-values (all = 0)          │   │
 │   │                                             │   │
 │   │   2. Choose action (ε-greedy strategy)      │   │
 │   │                                             │   │
 │   │   3. Take action                            │   │
 │   │                                             │   │
 │   │   4. Observe reward & next state            │   │
 │   │                                             │   │
 │   │   5. Update Q-values using formula          │   │
 │   │                                             │   │
 │   └─────────────────────────────────────────────┘   │
 └─────────────────────────────────────────────────────┘
```

### Step-by-Step Explanation of Functions in the Flow:

1. **`__init__` (Initialization)**:
   - Initialize the agent’s learning parameters (e.g., exploration rate, learning rate, discount factor).
   - Initialize a table (or Counter) to store Q-values for state-action pairs.

2. **`getQValue` (Get Current Q-value)**:
   - Return the Q-value for the current (state, action) pair from the Q-value table.
   - If the (state, action) pair hasn’t been encountered yet, the default value is 0.

3. **`computeValueFromQValues` (Compute Best Value for State)**:
   - Determine the maximum Q-value for the current state over all possible actions.
   - This represents the value of the state (i.e., the best expected future reward).

4. **`computeActionFromQValues` (Compute Best Action for State)**:
   - Choose the action that maximizes the Q-value for the current state.
   - If there is a tie, randomly choose one of the best actions.

5. **`getAction` (Choose Action)**:
   - With probability `epsilon`, the agent will choose a random action (exploration).
   - Otherwise, it will choose the best action based on the current Q-values (exploitation).

6. **`update` (Update Q-values)**:
   - After taking an action and observing the result (reward and next state), the agent updates the Q-value for the (state, action) pair.
   - The update is based on the observed reward and the value of the next state, using the Q-learning formula.

### Conclusion:
The agent repeats this cycle, adjusting its Q-values over time based on its experiences in the environment. This allows it to gradually improve its policy, which determines the best action to take in any given state.