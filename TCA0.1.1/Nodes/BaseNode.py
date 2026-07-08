import sys
LOGGING_ENABLED = '--debug' in sys.argv
from abc import ABC, abstractmethod

class BaseNode(ABC):
    @abstractmethod
    def process(self, data):
        if LOGGING_ENABLED:
            print(f'[DEBUG] BaseNode process called with data: {data}')
        pass

    