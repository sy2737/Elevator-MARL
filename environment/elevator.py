import simpy
import random
import numpy as np
from .logger import get_my_logger

class Elevator():
    # Action functions that elevators have
    # States of the elevator
    nState = 3 # Number of states
    IDLE = 0
    MOVING_UP = 1
    MOVING_DOWN = -1

    # Time it takes for elevators to move
    MOVE_MOVE = 3
    MOVE_IDLE = 5
    IDLE_IDLE = 8

    # Intent 
    INTENT_IDLE = 0
    INTENT_DOWN = -1
    INTENT_UP = 1
    INTENT_NOT_SET = 2

    action_space = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    action_space_size = 10
    def __init__(self, env, init_floor, weightLimit, id):
        self.env                    = env
        self.floor                  = init_floor
        self.requested_fls          = set()
        self.carrying               = set()
        self.carrying_weight        = 0
        self.weight_limit           = weightLimit
        self.state                  = self.IDLE
        self.intent                 = self.INTENT_IDLE
        self.id                     = id
        self.current_reward         = 0
        self.last_decision_epoch    = self.env.simenv.now
        self.logger                 = get_my_logger("Elevator_{}".format(self.id))

        self.ACTION_FUNCTION_MAP = {
            0: self._move_move,
            1: self._move_idle,
            2: self._idle_up_move,
            3: self._idle_up_idle,
            4: self._idle_down_move,
            5: self._idle_down_idle,
            6: self._idle_idle,
            7: self._idle_intend_up,
            8: self._idle_intend_down,
            9: self._idle_intend_idle,
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
        self.env.nPassenger_served += 1
        self.env.wait_time_of_served += (self.env.now() - passenger.created_at)

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
        ( ) Declaring Intent:
        (7)     Intent Up
        (8)     Intent Down
        (9)     Intent Idle
        '''
        # Staying IDLE is special because it may be interrupted.
        assert action in self.legal_actions(), "Agent picked illegal action"
        if action==6:
            self.idling_event = self.env.simenv.process(self.ACTION_FUNCTION_MAP[action]())
            try:
                self.logger.debug("Elevator {} is about to idle!".format(self.id))
                yield self.idling_event
            except simpy.Interrupt:
                self.logger.debug("Elevator {} is interrupted!".format(self.id))
                # Interrupted, so decision epoch came early...
                self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

        else:
            self.logger.debug("Agent {} picked action: {}".format(self.id, self.ACTION_FUNCTION_MAP[action].__name__))
            yield self.env.simenv.process(self.ACTION_FUNCTION_MAP[action]())

    def _move_move(self):
        self.intent = self.INTENT_NOT_SET
        # State unchanged, and next event_epoch is some time in the future
        yield self.env.simenv.timeout(self.MOVE_MOVE)
        self._update_floor()
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _move_idle(self):
        self.intent = self.INTENT_NOT_SET
        yield self.env.simenv.timeout(self.MOVE_IDLE)
        self._update_floor()
        self.state = self.IDLE
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_up_move(self):
        self.intent = self.INTENT_NOT_SET
        self.state = self.MOVING_UP
        # Move
        yield self.env.simenv.timeout(self.MOVE_IDLE)
        self._update_floor()
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_up_idle(self):
        self.intent = self.INTENT_NOT_SET
        self.state = self.MOVING_UP
        yield self.env.simenv.timeout(self.IDLE_IDLE)
        self._update_floor()
        self.state = self.IDLE
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_down_move(self):
        self.intent = self.INTENT_NOT_SET
        self.state = self.MOVING_DOWN
        yield self.env.simenv.timeout(self.MOVE_IDLE)
        self._update_floor()
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_down_idle(self):
        self.intent = self.INTENT_NOT_SET
        self.state = self.MOVING_DOWN
        yield self.env.simenv.timeout(self.IDLE_IDLE)
        self._update_floor()
        self.state = self.IDLE
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_idle(self):
        self.intent = self.INTENT_NOT_SET
        assert self.state==self.IDLE
        # Stay idle for at most sometime, and then decide if it wants to stay idle again
        yield self.env.simenv.timeout(random.normalvariate(self.IDLE_IDLE, 0.01))
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def _idle_intend_up(self):
        self.intent = self.INTENT_UP
        yield self.env.generate_loading_event(self)
        self.env.trigger_epoch_event("LoadingFinished_{}".format(self.id))

    def _idle_intend_down(self):
        self.intent = self.INTENT_DOWN
        yield self.env.generate_loading_event(self)
        self.env.trigger_epoch_event("LoadingFinished_{}".format(self.id))

    def _idle_intend_idle(self):
        self.intent = self.INTENT_IDLE
        yield self.env.generate_loading_event(self)
        self.env.trigger_epoch_event("LoadingFinished_{}".format(self.id))

    def _update_floor(self):
        self.floor += self.state
        for p in self.carrying:
            p.update_floor(self.floor)

    def legal_actions(self):
        legal = set()
        if self.state == self.IDLE:
            # If the intent is not declared, then you need to declare intent
            # Intent to move up is illegal at the top floor
            # Intent to move down is illegal at the bottom floor
            if self.intent == self.INTENT_NOT_SET:
                legal.update([7, 8, 9])
                if self.floor == 0:
                    legal.remove(8)
                if self.floor == self.env.nFloor-1:
                    legal.remove(7)

            elif self.intent == self.INTENT_UP:
                legal.update([2, 3])
                # If almost at the top, you have to stop at the next floor up
                if self.floor == self.env.nFloor-2:
                    legal.remove(2)
            elif self.intent == self.INTENT_DOWN:
                legal.update([4, 5])
                # If almost at the bottom, you have to stop at the next floow below
                if self.floor == 1:
                    legal.remove(4)
            else:
                # the only option after declaring intent to idle is to idle
                legal.update([6,])

        else:
            legal.update([0,1])
            if self.floor == self.env.nFloor-2 and self.state==self.MOVING_UP:
                legal.remove(0)
            if self.floor == 1 and self.state==self.MOVING_DOWN:
                legal.remove(0)
        return legal

    def update_reward(self, reward):
        # Update the reward inccurred since the last decision epoch
        self.current_reward += reward
        return True

    def get_reward(self, decision_epoch):
        output = self.current_reward
        if decision_epoch:
            self.current_reward = 0
        return output

    def _one_hot_encode(self, x, values):
        values = np.array(values)
        return values==x

    def get_states(self, decision_epoch):
        '''generate state with respect to the elevator's perspective'''

        elevator_positions = [self.floor] + [e.floor for e in self.env.elevators if e is not self]

        onehot_elevator_positions = np.concatenate([
            self._one_hot_encode(fl, range(self.env.nFloor)) for fl in elevator_positions
        ])

        elevator_states = [self.state] + [e.state for e in self.env.elevators if e is not self]

        onehot_elevator_states = np.concatenate([
            self._one_hot_encode(
                state, [self.IDLE, self.MOVING_UP, self.MOVING_DOWN]
            ) for state in elevator_states
        ])

        hall_call_up_times = (self.env.simenv.now - self.env.hall_calls_up_pressed_at)*self.env.hall_calls_up
        hall_call_down_times = (self.env.simenv.now - self.env.hall_calls_down_pressed_at)*self.env.hall_calls_down

        # Floor calls from within
        requested_calls = [False]*self.env.nFloor
        for fl in self.requested_fls:
            requested_calls[fl] = True

        time_elapsed = [self.env.simenv.now-self.last_decision_epoch]

        state_representation = np.concatenate([
            self.env.hall_calls_up,
            self.env.hall_calls_down,
            #hall_call_up_times,
            #hall_call_down_times,
            onehot_elevator_positions,
            onehot_elevator_states,
            requested_calls,
            [self.carrying_weight],
            time_elapsed
        ])
        #assert len(state_representation)==self.env.observation_space_size, "should probably modify the obs_space in env.py to match the state output of Elevator.get_states()"
        if decision_epoch:
            self.last_decision_epoch = self.env.simenv.now
        return state_representation
