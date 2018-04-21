import simpy
import random
from .logger import get_my_logger

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

    def __init__(self, env, init_floor, weightLimit, id):
        self.env = env
        self.floor = init_floor
        self.requested_fls = set()
        self.carrying = set()
        self.carrying_weight = 0
        self.weight_limit = weightLimit
        self.state = self.IDLE
        self.id = id
        self.logger = get_my_logger("Elevator_{}".format(self.id))

        self.ACTION_FUNCTION_MAP = {
            0: self._move_move,
            1: self._move_idle,
            2: self._idle_up_move,
            3: self._idle_up_idle,
            4: self._idle_down_move,
            5: self._idle_down_idle,
            6: self._idle_idle,
        }
        self.env.simenv.process(self.act(6))

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
        if passenger.weight + self.carrying_weight > self.weight_limit:
            return False
        self.carrying.add(passenger)
        self.env.psngr_by_fl[self.floor].remove(passenger)
        self.carrying_weight += passenger.weight
        return True
    
    def leave(self, passenger):
        self.carrying.remove(passenger)
        self.carrying_weight = sum([p.weight for p in self.carrying])
        return True

    def interrupt_idling(self):
        self.idling_event.interrupt()

    
    def act(self, action):
        '''
        Six different actions, valid at different states
        ( ) MOVING:
        (0)     Move Move
        (1)     Move Idle
        ( ) IDLE:
        (2)     UP move 
        (3)     UP stop
        (4)     DOWN move 
        (5)     DOWN stop 
        (6)     Stay Idle
        '''
        # Staying IDLE is special because it may be interrupted.
        if action==6:
            self.idling_event = self.env.simenv.process(self.ACTION_FUNCTION_MAP[action]())
            try:
                self.logger.debug("Elevator {} is about to idle!".format(self.id))
                yield self.idling_event
            except simpy.Interrupt:
                self.logger.debug("Elevator {} is interrupted!".format(self.id))
                pass # Interrupted, so decision epoch came early...
        else:
            yield self.env.simenv.process(self.ACTION_FUNCTION_MAP[action]())
        self.logger.debug("Triggering Elevator {} arrival!! at time: {}".format(self.id, self.env.simenv.now))
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))
    
    def _move_move(self):
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

    def _idle_idle(self):
        assert self.state==self.IDLE
        # Stay idle for at most sometime, and then decide if it wants to stay idle again
        yield self.env.simenv.timeout(random.normalvariate(self.IDLE_IDLE, 0.01))
    
    def _update_floor(self):
        self.floor += self.state
        for p in self.carrying:
            p.update_floor(self.floor)


    def legal_actions(self):
        legal = set()
        if self.state == self.IDLE:
            legal.update([2, 3, 4, 5, 6]) 
            # If almost at the top, you have to stop at the next floor up
            if self.floor == self.env.nFloor-2:
                legal.remove(2)
            # If at the top, you can't move up
            if self.floor == self.env.nFloor-1:
                legal.remove(2)
                legal.remove(3)

            # If almost at the bottom, you have to stop at the next floow below
            if self.floor == 1:
                legal.remove(4)
            # If at the bottom, you can't move down
            if self.floor == 0:
                legal.remove(4)
                legal.remove(5)
            return legal

        else:
            legal.update([0,1])
            if self.floor == self.env.nFloor-2 and self.state==self.MOVING_UP:
                legal.remove(0)
            if self.floor == 1 and self.state==self.MOVING_DOWN:
                legal.remove(0)
            return legal

                

