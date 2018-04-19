import simpy
from simpy.events import AnyOf
import random 
from passenger import Passenger
from elevator import Elevator

def make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, moveSpeed):

    '''
    nElevator
    nFloor
    spawnRates: Poisson rate for passenger generation on each floor
    avgWeight: avg weight of passengers
    weightLimit: weight limit of the elevators
    loadTime: avg load time per passenger (offloading, onloading) TODO: exponential?
    moveSpeed: [move to move, move to stop/stop to move, stop to stop]
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
        self.psngr_by_fl= {floor:[] for floor in range(nFloor)}
        #TODO: sinlge->multi
        #self.decision_elevator = None
        
        pass

    def step(self, action):
        '''Steps through the simulation until the next decision epoch is reached
           at which point the function prepares the state representation and 
           returns it
        '''
        #TODO: sinlge->multi
        #self.elevators[self.decision_elevator].act(action)
        self.elevators[0].act(action)

        while True:
            event_type = self.simenv.run(until=AnyOf(self.epoch_events.values()))

        print("Decision epoch triggered at time {}".format(self.simenv.now))
        return self.get_states()

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
        self.decision_epoch = self.simenv.event()
        self.elevators = [Elevator(1) for _ in range(self.nElevator)]
        # TODO: will need to modify this part to differentiate different elevators
        self.epoch_events = {
            "PassengerArrival": self.simenv.event("PassengerArrival"),
            "ElevatorArrival": self.simenv.event("ElevatorArrival"),
            "LoadingFinished": self.simenv.event("LoadingFinished"),
        }
        self.hall_calls_up = [0]*nFloor
        self.hall_calls_down = [0]*nFloor

        return self.get_states()

    def trigger_decision_epoch(self, event_type):
        self.epoch_events[event_type].succeed()
        self.epoch_events[event_type] = self.simenv.event(event_type)

    def passenger_generator(self):
        while True:
            # Keeps generating new passengers
            yield self.simenv.timeout(random.expovariate(sum(self.spawnRates)))
            print("generating new passenger!")
            floor = random.choices(range(self.nFloor), self.spawnRates)[0]
            # Weight is normally distributed 
            self.psngr_by_fl[floor].append(Passenger(random.normalvariate(self.avgWeight, 10), floor))
            if len(self.psngr_by_fl[floor])>=1:
                self.trigger_decision_epoch("PassengerArrival")

    def passenger_loading_process(self):
        ''' When ElevatorArrival event takes place, we process the loading/off-loading 
            using this function
        ''' 
        raise NotImplementedError



