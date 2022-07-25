

from abc import ABC, abstractmethod, abstractproperty


class Buffer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fill(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def update_policy():
        pass

    @abstractmethod
    def reset(self):
        pass





