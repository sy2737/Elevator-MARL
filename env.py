import simpy
from simpy.events import AnyOf
import random 
from numpy import Sign
from passenger import Passenger
    from elevator import Elevator

def make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, moveSpeed):

    '''
    nElevator:     Number of elevators
    nFloor:        Number of floors
    spawnRates:    Poisson rate for passenger generation on each floor
    avgWeight:     avg weight of passengers
    weightLimit:   weight limit of the elevators
    loadTime:      avg load time per passenger (offloading, onloading), normal
    moveSpeed:     [move to move, move to stop/stop to move, stop to stop]
    '''

    # initializes a simpy environment
    simenv = simpy.Environment()

    # Returns an Environment instance
    return Environment(simenv, nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, moveSpeed)

    
class Environment():
    '''
    controls the simulation, allows outside controllers (policies)
    to act at decision epochs (certain specific types of events)
    '''
    def __init__(self, simenv, nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, moveSpeed):
        self.simenv = simenv
        self.nElevaotr = nElevator
        self.nFloor = nFloor
        self.spawnRates = spawnRates
        self.avgWeight = avgWeight
        self.weightLimit = weightLimit
        self.loadTime = loadTime
        self.moveSpeed = moveSpeed

        self.elevators = None
        self.epoch_events = None
        self.psngr_by_fl= {floor:set() for floor in range(nFloor)}
        #TODO: sinlge->multi
        #self.decision_elevator = None
        
        pass

    def step(self, action):
        '''Steps through the simulation until the next decision epoch is reached
           at which point the function prepares the state representation and 
           returns it
        
        Decision epoch comes immediately when elevator reaches a floor
        - if elevator happens to stop, then you process passengers, and when that
          is done, you immediately let the elevator move toward the direction that
          it decided early on. 
        '''
        #TODO: sinlge->multi
        #self.elevators[self.decision_elevator].act(action)
        # This schedules an event for the next ElevatorArrival event for that elevator
        

        self.simenv.process(self.elevators[0].act(action))

        while True:
            event_type = self.simenv.run(until=AnyOf(self.epoch_events.values()))
            # Here is where we process the events
            # We calculate total weighting time etc, and assign loading events
            # If the event_type qualifies as a decision epoch then break 
            # out of the while loop and return the appropriate state 
            if event_type == "ElevatorArrival":
                decision = self._process_elevator_arrival()
            elif event_type == "PassengerArrival":
                decision = self._process_passenger_arrival()
            else:
                raise ValueError("Unimplemented event type: {}".format(event_type))
            if decision:
                break

        # TODO: elevator should handle what kind of env state representation it wants to return
        #       It should only return state values in formats that it sees
        print("Decision epoch triggered at time {}, by event type: {}".format(
            self.simenv.now, event_type)
        )
        return self.get_states()
    
    def _process_elevator_arrival(self, idx=0):
        # Every elevator arrival is an decision epoch
        return True

    def _process_passenger_arrival(self):
        # Decision epoch if there is an elevator waiting, otherwise
        # simply update the hallway calls
        raise NotImplementedError


    def generate_loading_event(self, elevator):
        '''Elevator calls this function when it reaches a floor and is ready to load'''
        num_loaded = 0
        for p in elevator.carrying:
            num_loaded += p.leave_if_arrived()
        for p in self.psngr_by_fl[elevator.floor]:
            if Sign(p.destination-elevator.floor) == elevator.state:
                if p.enter(elevator):
                    self.psngr_by_fl.remove(p)
                    num_loaded += 1
            
        yield self.simenv.timeout(2+random.normalvariate(self.num_loaded*self.loadTime, 1))


    def get_states(self):
        '''
        Prepares a state representation that is appropriate for the agents.
        i.e. elevators shouldn't be able to see how many passengers there are on 
             each floor
        '''
        nPsngrs_by_fl = [len(self.psngr_by_fl[i]) for i in range(self.nFloor)]
        return [nPsngrs_by_fl, self.hall_calls_up, self.hall_calls_down]

    def reset(self):
        '''
        Initializes the simulation, which implies that we reset the following items:
          - simluation environment
          - decision_epoch events
        '''
        self.simenv.process(self.passenger_generator())
        self.elevators = [Elevator(1) for _ in range(self.nElevator)]
        # TODO: will need to modify this part to differentiate different elevators
        self.epoch_events = {
            "PassengerArrival": self.simenv.event("PassengerArrival"),
            "ElevatorArrival": self.simenv.event("ElevatorArrival"),
        }
        self.hall_calls_up = [0]*nFloor
        self.hall_calls_down = [0]*nFloor

        return self.get_states()

    def trigger_epoch_event(self, event_type):
        self.epoch_events[event_type].succeed()
        self.epoch_events[event_type] = self.simenv.event(event_type)

    def passenger_generator(self):
        while True:
            # Keeps generating new passengers
            yield self.simenv.timeout(random.expovariate(sum(self.spawnRates)))
            print("generating new passenger!")
            floor = random.choices(range(self.nFloor), self.spawnRates)[0]
            # Weight is normally distributed 
            self.psngr_by_fl[floor].add(Passenger(random.normalvariate(self.avgWeight, 10), floor))
            if len(self.psngr_by_fl[floor])>=1:
                self.trigger_epoch_event("PassengerArrival")



