# reinforce_dqn_taxi
Deep Q-Learning (DQN) for OpenAI Taxi Domain.

Implementation utilizes a target network to guide learning in the right direction and takes advantage of experience replay to prevent state transition dependencies from interfering with learning. The Markov Decision Process and overall environment are defined/provided by OpenAI. Tensorboard was integrated into this project for training/progress visualizations.

**Notes:** Empircally, running the DQN model with multiple passes (saving weights from previous pass and running model again initialized with those weights) leads to better performance because the exploration/exploitation epsilon constant is allowed to re-decay, effectively helping the agent escape from local "traps" and not get stuck during training. Essentially, the agent gets to pick up from where it ended in the last pass, except with a fresh pair of eyes.

**Description of OpenAI Taxi Domain:**

    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+

There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

    Observations: There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi),     and 4 destination locations. 
    
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
        
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.
    
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
