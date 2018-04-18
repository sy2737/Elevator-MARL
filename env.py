import simpy
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
        self.psngr_by_fl= {floor:[] for floor in range(nFloor)}
        self.elevators = None
        self.nFloor = nFloor
        self.nElevaotr = nElevator
        self.spawnRates = spawnRates
        self.simenv = simenv
        self.avgWeight = avgWeight
        pass

    def step(self, action):
        '''Steps through the simulation until the next decision epoch is reached
           at which point the function prepares the state representation and 
           returns it
        '''
        self.simenv.run(until=self.decision_epoch)
        print("Decision epoch triggered at time {}".format(self.simenv.now))
        return self.get_states()

    def get_states(self):
        '''
        Prepares a state representation that is appropriate for the agents.
        i.e. elevators shouldn't be able to see how many passengers there are on 
             each floor
        '''
        nPsngrs_by_fl = [len(self.psngr_by_fl[i]) for i in range(self.nFloor)]
        return nPsngrs_by_fl

    def reset(self):
        #for _ in range(self.nElevaotr):
        #    self.simenv.process(Elevator(1))
        
        self.simenv.process(self.passenger_generator())
        self.decision_epoch = self.simenv.event()

        return self.get_states()

    def trigger_decision_epoch(self):
        self.decision_epoch.succeed()
        self.decision_epoch = self.simenv.event()

    def passenger_generator(self):
        while True:
            # Keeps generating new passengers
            yield self.simenv.timeout(random.expovariate(sum(self.spawnRates)))
            print("generating new passenger!")
            floor = random.choices(range(self.nFloor), self.spawnRates)[0]
            # Weight is normally distributed 
            self.psngr_by_fl[floor].append(Passenger(random.normalvariate(self.avgWeight, 10), floor))
            if len(self.psngr_by_fl[floor])>=1:
                self.trigger_decision_epoch()
                
            

