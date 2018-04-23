import simpy
import time
from simpy.events import AnyOf
import random 
from math import log as Log
from numpy import sign as Sign
from numpy import exp as Exp
import numpy as np
from .passenger import Passenger
from .elevator import Elevator
from .logger import get_my_logger

def make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime):

    '''
    nElevator:     Number of elevators
    nFloor:        Number of floors
    spawnRates:    Poisson rate for passenger generation on each floor
    avgWeight:     avg weight of passengers
    weightLimit:   weight limit of the elevators
    loadTime:      avg load time per passenger (offloading, onloading), normal
                   total load time = 2+norm(log(1+num_loaded)*loadTime, 1)
    moveSpeed:     [move to move, move to stop/stop to move, stop to stop]
    '''

    # initializes a simpy environment
    simenv = simpy.Environment()

    # Returns an Environment instance
    return Environment(simenv, nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime)

    
class Environment():
    '''
    controls the simulation, allows outside controllers (policies)
    to act at decision epochs (certain specific types of events)
    '''
    def __init__(self, simenv, nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime):
        self.simenv = simenv
        self.nElevator = nElevator
        self.nFloor = nFloor
        self.spawnRates = spawnRates
        self.avgWeight = avgWeight
        self.weightLimit = weightLimit
        self.loadTime = loadTime

        self.elevators = None
        self.epoch_events = None
        self.psngr_by_fl= {floor:set() for floor in range(nFloor)}
        self.decision_elevators = []
        self.action_space = Elevator.action_space
        self.action_space_size = Elevator.action_space_size
        # Need to manually change this...
        # hall_calls_up, hall_calls_down, hall_call_up_times, hall_call_down_times
        # onehot_elevator_positions (nFloor positions)
        # onehot_elevator_states (3 states)
        # time_elapsed
        self.observation_space_size = nFloor*4 + nFloor*nElevator + 3*nElevator + 1


        self.last_reward_time = 0
        self.reward_discount = 0.01
        self._elevator_candidate = 0
        self.logger = get_my_logger(self.__class__.__name__)
        
        pass

    def step(self, actions):
        '''Steps through the simulation until the next decision epoch is reached
           at which point the function prepares the state representation and 
           returns it
        
        Decision epoch comes immediately when elevator reaches a floor
        - if elevator happens to stop, then you process passengers, and when that
          is done, you immediately let the elevator move toward the direction that
          it decided early on. 
        '''
        # This schedules an event for the next ElevatorArrival event for that elevator
        
        for idx, a in enumerate(actions):
            if a == -1:
                continue
            self.simenv.process(self.elevators[self.decision_elevators[idx]].act(a))

        while True:
            self.decision_elevators = []
            finished_events = self.simenv.run(until=AnyOf(self.simenv, self.epoch_events.values())).events
            self.update_all_loss()
            # There can be multiple events finished (when passenger arrives and multiple elevators go into decision mode)

            # TODO: Here is where we process the events
            # We calculate total weighting time etc, and assign loading events
            # If the event_type qualifies as a decision epoch then break 
            # out of the while loop and return the appropriate state 
            if len(finished_events)>1:
                for e in finished_events:
                    assert("ElevatorArrival" in e.value)
            for event in finished_events:
                event_type = event.value
                if "ElevatorArrival" in event_type:
                    decision = self._process_elevator_arrival(event_type)
                elif event_type == "PassengerArrival":
                    decision = self._process_passenger_arrival()
                else:
                    raise ValueError("Unimplemented event type: {}".format(event_type))
            if decision:
                break

        # TODO: elevator should handle what kind of env state representation it wants to return
        #       It should only return state values in formats that it sees
        self.logger.info("Decision epoch triggered at time {:8.3f}, by event type: {}".format(
            self.simenv.now, [event.value for event in finished_events])
        )
        output = {
            "states": self.get_states(self.decision_elevators, True),
            "rewards": self.get_rewards(self.decision_elevators, True),
            "decision agents": self.decision_elevators
        }
        return output
    
    def _process_elevator_arrival(self, event_type):
        # Every elevator arrival is an decision epoch
        self.decision_elevators.append(int(event_type.split('_')[-1]))
        return True

    def _process_passenger_arrival_helper(self):
        # Decision epoch if there is an elevator waiting, otherwise
        # simply update the hallway calls.
        # 
        self._update_hall_calls()
        for idx in range(self._elevator_candidate, self.nElevator):
            e = self.elevators[idx]
            self._elevator_candidate += 1
            if e.state == self.elevators[0].IDLE:
                self.logger.debug("INTERRUPTING ELEVATOR {}, at time {}".format(e.id, self.simenv.now))
                e.interrupt_idling()
        return False

    def _process_passenger_arrival(self):
        # Interrupt the idling process of elevator if it is waiting
        # If there are at least two, then allow both of them to make a decision
        self._process_passenger_arrival_helper()
        if self._elevator_candidate < self.nElevator:
            self.trigger_epoch_event("PassengerArrival")
        else:
            self._elevator_candidate = 0
        return False

    def generate_loading_event(self, elevator):
        '''Elevator calls this function when it reaches a floor and is ready to load'''
        num_loaded = 0
        carrying = [p for p in elevator.carrying]
        for p in carrying:
            num_loaded += p.leave_if_arrived()
        if num_loaded > 0:
            assert(elevator.floor in elevator.requested_fls)
            elevator.requested_fls.remove(elevator.floor)

        waiting = [p for p in self.psngr_by_fl[elevator.floor]]
        for p in waiting:
            if Sign(p.destination-elevator.floor) == elevator.state:
                if p.enter(elevator):
                    num_loaded += 1
            
        self._update_hall_calls()
        return self.simenv.timeout(2+max(0,random.normalvariate(Log(1+num_loaded)*self.loadTime, 1)))

    def now(self):
        return self.simenv.now

    def get_states(self, elevator_idxes, decision_epoch):
        '''
        Input:
            elevator_idxes: the indexes for elevators that we want the states for
            decision_epoch: if this is a decision epoch for the elevators then True
                            If set to True this will tell elevators to update their
                            last deicions_epoch_time

        Prepares a state representation that is appropriate for the agents.
        i.e. elevators shouldn't be able to see how many passengers there are on 
             each floor
        '''
        #elevator_positions = [e.floor for e in self.elevators]
        #elevator_states = [e.state for e in self.elevators]

        #return {
        #    "hall_calls_up": self.hall_calls_up,
        #    "hall_calls_down": self.hall_calls_down,
        #    "elevator_positions": elevator_positions,
        #    "elevator_states": elevator_states,
        #    "decision_elevators": self.decision_elevators,
        #}
        return [self.elevators[idx].get_states(decision_epoch) for idx in elevator_idxes]
    
    def get_rewards(self, elevator_idxes, decision_epoch):
        return [self.elevators[idx].get_loss(decision_epoch) for idx in elevator_idxes]


    def reset(self):
        '''
        Initializes the simulation, which implies that we reset the following items:
          - simluation environment
          - decision_epoch events
        '''
        self.simenv.process(self.passenger_generator())
        self.elevators = [Elevator(self, 0, self.weightLimit, id) for id in range(self.nElevator)]
        # TODO: will need to modify this part to differentiate different elevators
        self.epoch_events = {
            "PassengerArrival": self.simenv.event(),
        }
        for idx in range(self.nElevator):
            self.epoch_events["ElevatorArrival_{}".format(idx)] = self.simenv.event()

        self.hall_calls_up = np.zeros(self.nFloor)
        self.hall_calls_up_pressed_at = np.zeros(self.nFloor)
        self.hall_calls_down = np.zeros(self.nFloor)
        self.hall_calls_down_pressed_at = np.zeros(self.nFloor)

        return self.step([-1])

    def _update_hall_calls(self):
        for fl in range(self.nFloor):
            up = False
            down = False
            for p in self.psngr_by_fl[fl]:
                if p.destination > fl:
                    up = True
                elif p.destination < fl:
                    down = True
                else:
                    raise ValueError("Passenger's floor is equal to the destination?")
            if up and not self.hall_calls_up[fl]:
                self.hall_calls_up[fl] = 1
                self.hall_calls_up_pressed_at[fl] = self.simenv.now
            if not up:
                self.hall_calls_up[fl] = 0
            if down and not self.hall_calls_down[fl]:
                self.hall_calls_down[fl] = 1
                self.hall_calls_down_pressed_at[fl] = self.simenv.now
            if not down:
                self.hall_calls_down[fl] = 0

            

    def trigger_epoch_event(self, event_type):
        self.epoch_events[event_type].succeed(event_type)
        self.epoch_events[event_type] = self.simenv.event()

    def passenger_generator(self):
        while True:
            # Keeps generating new passengers
            yield self.simenv.timeout(random.expovariate(sum(self.spawnRates)))
            self.logger.debug("generating new passenger! at time {}".format(self.simenv.now))
            floor = random.choices(range(self.nFloor), self.spawnRates)[0]
            # Weight is normally distributed 
            self.psngr_by_fl[floor].add(
                Passenger(
                    random.normalvariate(self.avgWeight, 10), floor, self._destination(floor), self.simenv.now
                )
            )
            if len(self.psngr_by_fl[floor])>=1:
                self.trigger_epoch_event("PassengerArrival")


    def _destination(self, starting_floor):
        '''
        Generates destination given starting floor
        '''
        # TODO: this distribution needs to be more sophisticated. ie: first floor
        options = set(range(self.nFloor))
        options.remove(starting_floor)
        return random.sample(options, 1)[0]
    
    def legal_actions(self, idx):
        return self.elevators[idx].legal_actions()

    def render(self):
        '''
        Prints some stone age visualization in stdout...
        '''
        DIR_MAP = {self.elevators[0].IDLE: '-', self.elevators[0].MOVING_UP: '^', self.elevators[0].MOVING_DOWN:'v'}

        for floor in range(self.nFloor-1, -1, -1):
            num_psngr_going_up = len([p for p in self.psngr_by_fl[floor] if p.destination>p.floor])
            num_psngr_going_down = len([p for p in self.psngr_by_fl[floor] if p.destination<p.floor])

            string = ""
            for elevator in self.elevators:
                if elevator.floor == floor:
                    string+="|{}{:>2}|".format(DIR_MAP[elevator.state], len(elevator.carrying))
                else:
                    string+="|   |"
            string+="^"*num_psngr_going_up
            string+="v"*num_psngr_going_down
            print(string)


    def update_all_loss(self):
        # Add incremental loss to all elevators and update the last_reward_time
        for e in self.elevators:
            e.update_loss(self.calculate_loss(e.last_decision_epoch))
        self.last_reward_time = self.simenv.now
        return True


    def calculate_loss(self, d):
        '''
        Input:
            d: last decision time of the elevator
        Output:
            the incremental loss that need incurred between last 
            reward calculation time and now
        '''
        output = 0
        # First calculate for all passengers in hall ways
        for fl in range(self.nFloor):
            for p in self.psngr_by_fl[fl]:
                output += self._calculate_loss(
                              d, self.last_reward_time, self.simenv.now, 
                              self.last_reward_time-p.created_at, self.simenv.now-p.created_at
                          )
        
        # Then calculate for all passengers in elevators
        for e in self.elevators:
            for p in e.carrying:
                output += self._calculate_loss(
                              d, self.last_reward_time, self.simenv.now, 
                              self.last_reward_time-p.created_at, self.simenv.now-p.created_at
                          )

        return output

    def _calculate_loss(self, d, t0, t1, w0, w1):
        return Exp(-self.reward_discount*(t0-d))\
               * (2/self.reward_discount**3 + 2*w0/self.reward_discount**2 + w0**2/self.reward_discount)\
               - Exp(-self.reward_discount*(t1-d))\
               * (2/self.reward_discount**3 + 2*w1/self.reward_discount**2 + w1**2/self.reward_discount)



