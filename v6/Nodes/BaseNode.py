from abc import ABC, abstractmethod

class BaseNode(ABC):
    @abstractmethod
    def process(self, data):
        pass

    