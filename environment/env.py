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

def make(nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, version, dst_pref=None):

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
    if not dst_pref:
        # Uniform
        dst_pref = {fl:np.ones(nFloor-1)/(nFloor-1) for fl in range(nFloor)}
    # Returns an Environment instance
    if version == 0:
        return Environment(simenv, nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, dst_pref)
    elif version == 1:
        return Environment_v1(simenv, nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, dst_pref)

    
class Environment():
    '''
    controls the simulation, allows outside controllers (policies)
    to act at decision epochs (certain specific types of events)
    '''
    def __init__(self, simenv, nElevator, nFloor, spawnRates, avgWeight, weightLimit, loadTime, dst_pref=None):
        self.simenv      = simenv
        self.nElevator   = nElevator
        self.nFloor      = nFloor
        self.spawnRates  = spawnRates
        self.avgWeight   = avgWeight
        self.weightLimit = weightLimit
        self.loadTime    = loadTime
        self.reward_discount = 0.01
        self.dst_pref    = dst_pref
        self.logger = get_my_logger(self.__class__.__name__)

        self.action_space = Elevator.action_space
        self.action_space_size = Elevator.action_space_size

        # Initialize environment so that we have state size
        initial_states,_,_ = self.reset().items()
        self.observation_space_size = len(initial_states[1][0])

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
            self.update_all_reward()
            # There can be multiple events finished (when passenger arrives and multiple elevators go into decision mode)

            # Here is where we process the events
            # We calculate total waiting time etc, and assign loading events
            # If the event_type qualifies as a decision epoch then break 
            # out of the while loop and return the appropriate state 
            for event in finished_events:
                event_type = event.value
                if "ElevatorArrival" in event_type:
                    decision = self._process_elevator_arrival(event_type)
                elif event_type == "PassengerArrival":
                    decision = self._process_passenger_arrival()
                elif "LoadingFinished" in event_type:
                    decision = self._process_loading_finished(event_type)
                else:
                    raise ValueError("Unimplemented event type: {}".format(event_type))
            if decision:
                break

        self.logger.debug("Decision epoch triggered at time {:8.3f}, by event type: {}".format(
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
            # An elevator can only be interrupted when it has not set its' intent and is idling
            if e.intent == e.INTENT_NOT_SET and e.state == e.IDLE:
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

    def _process_loading_finished(self, event_type):
        self.decision_elevators.append(int(event_type.split('_')[-1]))
        return True

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
            if Sign(p.destination-elevator.floor) == elevator.intent:
                if p.enter(elevator):
                    num_loaded += 1
            
        self._update_hall_calls(elevator.floor, elevator.intent)
        return self.simenv.timeout((2+max(0,random.normalvariate(Log(1+num_loaded)*self.loadTime, 1)))*(num_loaded>0))

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
        return [self.elevators[idx].get_states(decision_epoch) for idx in elevator_idxes]
    
    def get_rewards(self, elevator_idxes, decision_epoch):
        return [self.elevators[idx].get_reward(decision_epoch) for idx in elevator_idxes]


    def reset(self):
        '''
        Initializes the simulation, which implies that we reset the following items:
          - simluation environment
          - decision_epoch events
        '''
        self.psngr_by_fl         = {floor:set() for floor in range(self.nFloor)}
        self.decision_elevators  = []
        self.nPassenger_served   = 0
        self.wait_time_of_served = 0
        self.last_reward_time    = 0
        self._elevator_candidate = 0

        self.simenv = simpy.Environment()
        self.simenv.process(self.passenger_generator())
        self.elevators = [Elevator(self, np.random.choice(np.arange(self.nFloor)), self.weightLimit, id) for id in range(self.nElevator)]
        self.epoch_events = {
            "PassengerArrival": self.simenv.event(),
        }
        for idx in range(self.nElevator):
            self.epoch_events["ElevatorArrival_{}".format(idx)] = self.simenv.event()
        for idx in range(self.nElevator):
            self.epoch_events["LoadingFinished_{}".format(idx)] = self.simenv.event()

        self.hall_calls_up = np.zeros(self.nFloor)
        self.hall_calls_up_pressed_at = np.zeros(self.nFloor)
        self.hall_calls_down = np.zeros(self.nFloor)
        self.hall_calls_down_pressed_at = np.zeros(self.nFloor)

        return self.step([-1])

    def _update_hall_calls(self, reset_floor=None, direction=None):
        for fl in range(self.nFloor):
            up = False
            down = False
            for p in self.psngr_by_fl[fl]:
                if p.destination > fl:
                    up = True
                    # Even if this floor's request wasn't completely filled, the time should be updated
                    if fl==reset_floor and direction==1:
                        self.hall_calls_up_pressed_at[fl] = self.simenv.now
                elif p.destination < fl:
                    down = True
                    # Even if this floor's request wasn't completely filled, the time should be updated
                    if fl==reset_floor and direction==-1:
                        self.hall_calls_up_pressed_at[fl] = self.simenv.now
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
            if sum(self.spawnRates) == 0:
                yield self.simenv.event()
            else:
                yield self.simenv.timeout(random.expovariate(sum(self.spawnRates)))
            self.logger.debug("generating new passenger! at time {}".format(self.simenv.now))
            if sum(self.spawnRates) == 0:
                floor = random.choices(range(self.nFloor), [1]*self.nFloor)[0]
            else:
                floor = random.choices(range(self.nFloor), self.spawnRates)[0]
            # Weight is normally distributed 
            self.psngr_by_fl[floor].add(
                Passenger(
                    random.normalvariate(self.avgWeight, 10), floor, self._destination(floor), self.simenv.now
                )
            )
            if len(self.psngr_by_fl[floor])>=1:
                self.trigger_epoch_event("PassengerArrival")

    def set_spawnRates(self, spawnRates):
        self.spawnRates = spawnRates

    def _destination(self, starting_floor):
        '''
        Generates destination given starting floor
        '''
        options = set(range(self.nFloor))
        options.remove(starting_floor)
        return np.random.choice(list(options), p=self.dst_pref[starting_floor])
    
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

    def update_all_reward(self):
        # Add incremental reward to all elevators and update the last_reward_time
        # The elevators will store the sum of all the rewards in one decision period
        for e in self.elevators:
            e.update_reward(self.calculate_reward(e.last_decision_epoch))
        self.last_reward_time = self.simenv.now
        return True

    def calculate_reward(self, d):
        '''
        Input:
            d: last decision time of the elevator
        Output:
            the incremental reward received between last reward calculation time and now
        '''
        output = 0
        # First calculate for all passengers in hall ways
        for fl in range(self.nFloor):
            for p in self.psngr_by_fl[fl]:
                output += self._calculate_reward(
                              d, self.last_reward_time, self.simenv.now, 
                              self.last_reward_time-p.created_at, self.simenv.now-p.created_at
                          )
        
        # Then calculate for all passengers in elevators
        for e in self.elevators:
            for p in e.carrying:
                output += self._calculate_reward(
                              d, self.last_reward_time, self.simenv.now, 
                              self.last_reward_time-p.created_at, self.simenv.now-p.created_at
                          )

        return output*1e-6

    def _calculate_reward(self, d, t0, t1, w0, w1):
        return Exp(-self.reward_discount*(t0-d))\
               * (2/self.reward_discount**3 + 2*w0/self.reward_discount**2 + w0**2/self.reward_discount)\
               - Exp(-self.reward_discount*(t1-d))\
               * (2/self.reward_discount**3 + 2*w1/self.reward_discount**2 + w1**2/self.reward_discount)

    def avg_wait_time(self):
        if self.nPassenger_served > 0 :
            return self.wait_time_of_served/self.nPassenger_served
        return -1

    def no_passenger(self):
        # First check for all passengers in hall ways
        for fl in range(self.nFloor):
            if len(self.psngr_by_fl[fl])>0:
                return False
        # Then check for all passengers in elevators
        for e in self.elevators:
            if len(e.carrying)>0:
                return False
        return True

    @staticmethod
    def parse_states(state, nFloor, nElevator):
        '''Returns a dictionary of parsed state vector'''
        hall_calls_up = state[0:nFloor]
        hall_calls_down = state[nFloor:nFloor*2]
        #hall_call_up_times = state[nFloor*2:nFloor*3]
        #hall_call_down_times = state[nFloor*3:nFloor*4]
        #onehot_elevator_positions = state[nFloor*4:(nFloor*4+nElevator*nFloor)].reshape(nElevator,nFloor)
        #onehot_elevator_states = state[
        #    (nFloor*4+nElevator*nFloor):
        #    ((nFloor*4+nElevator*nFloor)+(nElevator*Elevator.nState))
        #]
        #requested_calls = state[((nFloor*4+nElevator*nFloor)+(nElevator*Elevator.nState)): -2]
        onehot_elevator_positions = state[nFloor*2:(nFloor*2+nElevator*nFloor)].reshape(nElevator,nFloor)
        onehot_elevator_states = state[
            (nFloor*2+nElevator*nFloor):
            ((nFloor*2+nElevator*nFloor)+(nElevator*Elevator.nState))
        ]
        requested_calls = state[((nFloor*2+nElevator*nFloor)+(nElevator*Elevator.nState)): -2]
        carrying_weight = state[-2]
        time_elapsed = state[-1]
        return {
            'hall_calls_up':                hall_calls_up,
            'hall_calls_down':              hall_calls_down,
            #'hall_call_up_times':           hall_call_up_times,
            #'hall_call_down_times':         hall_call_down_times,
            'onehot_elevator_positions':    onehot_elevator_positions,
            'onehot_elevator_states':       onehot_elevator_states,
            'requested_calls':              requested_calls,
            'carrying_weight':              carrying_weight,
            'time_elapsed':                 time_elapsed
        }



class Environment_v1(Environment):
    '''
    Reward is linear in time for as long as there are passengers in the system
    In other words, reward will be negative unless there is no passenger, in which case it's zero
    Number of passengers waiting doesn't matter
    '''
    def calculate_reward(self, d):
        if self.no_passenger():
            return 0
        return -(self.simenv.now-self.last_reward_time)
