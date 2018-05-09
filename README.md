# Elevator-MARL
Asynchronous Elevator Control Simulator + Multi-Agent Reinforcement Learning training algorithms

## Simulator
+ **Elevators**: are modeled as independent agents.  
+ **Elevator States**: Motion state: MOVING, IDLING; Intent state: IDLE_INTENT_IDLE, IDLE_INTENT_UP, IDLE_INTENT_DOWN  
+ **Observation Space**: elevator position, carrying weight, hall call status, and in-elevator requests.  
+ **Decision Epoch**: Two events. Elevator Arrival or Loading Finished. Elevator Arrival is triggered whenever an elevator reaches a floor,
or the previous idling event is finished. Loading Finished is triggered when the loading proccess is done and the elevator is ready to 
take the next action.    
+ **Action Space**: 

   If the elevator is moving when Elevator Arrival is triggered, then it needs to decide if it wants to stop on the next floor.
   
   If the elevator is IDLING when Elevator Arrival is triggered, then it needs to choose a intended moving direction and then a passenger
loading event is queued by the environment (intent needs to be declared so that passengers in the waiting area can decide whether they want
to enter the elevator or not). 

   When the Loading Finished event is triggered, the elevator needs to choose an action among actions that correspond to
its' declared intent.(For example, if the intent was MOVING_UP, then it choose between IDLE_UP_IDLE, or IDLE_UP_MOVE, basically whether
it wants to stop at the next floor up or not. This changes how much time it takes to move to the next floor.)  


Error will be thrown if illegal action is passed into the environment. Otherthan that the environment does not impose any constraint on what the elevators can choose to do.