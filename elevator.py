class Elevator():
    ACTION_FUNCTION_MAP = {
        0: self._move_up_move,
        1: self._move_up_idle,
        2: self._move_down_move,
        3: self._move_down_idle,
        4: self._idle_up_move,
        5: self._idle_up_idle,
        6: self._idle_down_move,
        7: self._idle_down_idle,
    }
    IDLE = 0
    MOVING_UP = 1
    MOVING_DOWN = 2
    def __init__(self, init_floor, weightLimit):
        self.floor = init_floor
        self.requested_fls = []
        self.carrying = []
        self.carrying_weight = 0
        self.weight_limit = weightLimit
        self.state = self.IDLE

    def request(self, floor):
        '''To be called by a passenger'''
        self.requested_fls.append(floor)

    def enter(self, passenger):
        '''
        Input: a passenger

        To be called by a passenger.
        Updates the carrying weight and carrying list
        Returns False if elevator is too full
        '''
        if passenger.weight + self.carrying_weight <= self.weight_limit:
            return False
        self.carrying.append(passenger)
        self.carrying_weight += passenger.weight
        return True
    
    def act(self, action):
        '''
        Seven different actions, valid at different states
        ( ) MOVING UP:
        (0)     UP move
        (1)     UP stop
        ( ) MOVING DOWN:
        (2)     DOWN move
        (3)     DOWN stop
        ( ) IDLE:
        (4)     UP stop
        (5)     UP move
        (6)     DOWN stop
        (7)     DOWN move
        '''
        self.ACTION_FUNCTION_MAP[action]()
    
    def _move_up_move(self):
        '''should probably generate an ElevatorArrival event?'''
        pass
    def _move_up_idle(self):
        pass
    def _move_down_move(self):
        pass
    def _move_down_idle(self):
        pass
    def _idle_up_move(self):
        pass
    def _idle_up_idle(self):
        pass
    def _idle_down_move(self):
        pass
    def _idle_down_idle(self):
        pass

