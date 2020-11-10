

class Node:

    def __init__(self, id, x, y):
        self.id = id
        self._x = x
        self._y = y


    @property
    def coordinates(self):
        return [self._x, self._y]