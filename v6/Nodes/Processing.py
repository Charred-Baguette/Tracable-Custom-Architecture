"""
from notes:
- processing : takes in some input and applies some kind of weight to it
  -- connections based on distance formula
  -- maybe bayes theorem or gauss
  -- the local modelto produce the prediciton and variance
  -- kills the signal if variance exceeds limits
"""

from .BaseNode import BaseNode

class ProcessingNode(BaseNode):
    def __init__(self, position):
        self.position = position  # Position in the nexus
    def process(self, data):
        # Implement the processing logic for the ProcessingNode
        pass
