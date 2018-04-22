class Passenger():
    def __init__(self, weight, floor, destination, time):
        self.weight = weight
        self.floor = floor
        self.destination = destination
        self.elevator = None
        self.created_at = time


    def enter(self, elevator):
        '''
        Input: the elevator that the passenger attempts to enter
        '''
        if elevator.enter(self):
            self.elevator = elevator
            self.elevator.request(self.destination)
            return True
        return False
    
    def leave_if_arrived(self):
        if self.destination == self.floor:
            self.elevator.leave(self)
            return True
        return False
        
    def update_floor(self, floor):
        '''Allows the elevator to update the floor attribute of the passenger'''
        self.floor = floor
