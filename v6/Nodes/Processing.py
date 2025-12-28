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
        self.signal = None  # Current signal being processed


    def process(self, data):
        # Implement the processing logic for the ProcessingNode
        pass
    
    def receive_signal(self, signal):
        # Implement logic to receive and process the signal
        self.signal = signal
    
    def apply_local_model(self, input_data):
        # Implement logic to apply the local model for prediction and variance
        pass
    
    def check_signal_variance(self, variance_threshold):
        # Implement logic to check and possibly kill the signal based on variance
        if self.signal and self.signal.accumulated_variance > variance_threshold:
            self.signal.kill_signal()
            self.signal = None