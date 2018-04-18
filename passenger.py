class Passenger():
    def __init__(self, weight, floor):
        self.weight=weight
        self.floor = floor

    def enter(self, elevator):
        '''
        Input: the elevator that the passenger attempts to enter
        Checks if elevator is full:
          if not returns an event to the simulator to be added to queue
          else returns False
        '''
        raise NotImplementedError
    
    def leave(self):
        '''
        Returns an event
        '''
        