import simpy
import random

class Elevator():
    # Action functions that elevators have
    # States of the elevator
    IDLE = 0
    MOVING_UP = 1
    MOVING_DOWN = -1

    # Time it takes for elevators to move
    MOVE_MOVE = 3
    MOVE_IDLE = 5
    IDLE_IDLE = 8

    def __init__(self, env, init_floor, weightLimit):
        self.env = env
        self.floor = init_floor
        self.requested_fls = set()
        self.carrying = set()
        self.carrying_weight = 0
        self.weight_limit = weightLimit
        self.state = self.IDLE

        self.ACTION_FUNCTION_MAP = {
            0: self._move_move,
            1: self._move_idle,
            3: self._idle_up_move,
            4: self._idle_up_idle,
            5: self._idle_down_move,
            6: self._idle_down_idle,
        }

    def request(self, floor):
        '''To be called by a passenger'''
        self.requested_fls.add(floor)

    def enter(self, passenger):
        '''
        Input: a passenger

        To be called by a passenger.
        Updates the carrying weight and carrying list
        Returns False if elevator is too full
        '''
        if passenger.weight + self.carrying_weight <= self.weight_limit:
            return False
        self.carrying.add(passenger)
        self.carrying_weight += passenger.weight
        return True
    
    def leave(self, passenger):
        self.carrying.remove(passenger)
        self.carrying_weight = sum([p.weight for p in self.carrying])
        return True

    
    def act(self, action):
        '''
        Six different actions, valid at different states
        ( ) MOVING:
        (0)     Move Move
        (1)     Move Idle
        ( ) IDLE:
        (2)     UP stop
        (3)     UP move
        (4)     DOWN stop
        (5)     DOWN move
        '''
        yield self.env.simenv.process(self.ACTION_FUNCTION_MAP[action]())
        self.env.trigger_epoch_event("ElevatorArrival")
    
    def _move_move(self):
        '''should probably generate an ElevatorArrival event?'''
        # State unchanged, and next event_epoch is some time in the future
        yield self.env.simenv.timeout(self.MOVE_MOVE)
        self._update_floor()

    def _move_idle(self):
        yield self.env.simenv.timeout(self.MOVE_IDLE)
        self._update_floor()
        self.state = self.IDLE

    def _idle_up_move(self):
        # Specify the intended direction first
        self.state = self.MOVING_UP
        # Load the passengers
        yield self.env.generate_loading_event(self)
        # Move
        yield self.env.simenv.timeout(self.MOVE_IDLE)
        self._update_floor()

    def _idle_up_idle(self):
        self.state = self.MOVING_UP
        yield self.env.generate_loading_event(self)
        yield self.env.simenv.timeout(self.IDLE_IDLE)
        self._update_floor()
        self.state = self.IDLE

    def _idle_down_move(self):
        self.state = self.MOVING_DOWN
        yield self.env.generate_loading_event(self)
        yield self.env.simenv.timeout(self.MOVE_IDLE)
        self._update_floor()

    def _idle_down_idle(self):
        self.state = self.MOVING_DOWN
        yield self.env.generate_loading_event(self)
        yield self.env.simenv.timeout(self.IDLE_IDLE)
        self._update_floor()
        self.state = self.IDLE
    
    def _update_floor(self):
        self.floor += self.state
        for p in self.carrying:
            self.p.update_floor(self.floor)

