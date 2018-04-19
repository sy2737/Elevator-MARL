class Passenger():
    def __init__(self, weight, floor, destination):
        self.weight = weight
        self.floor = floor
        self.destination = destination
        self.elevator = None

    def enter(self, elevator):
        '''
        Input: the elevator that the passenger attempts to enter
        '''
        if elevator.enter(self):
            self.elevator = elevator

    
    def leave(self):
        '''
        Returns an event
        '''
        
    def update_floor(self, floor):
        '''Allows the elevator to update the floor attribute of the passenger'''
        self.floor = floor
