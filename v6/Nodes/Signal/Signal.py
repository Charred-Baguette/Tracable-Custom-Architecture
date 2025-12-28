"""
from notes
- signal : The representation of the data traveling between nodes 
    -- in a 2d system there should be 4 handlers and in total 4 * (.05 * total_processing_node_count) signals
    -- can be killed partway based on variance
    -- hop based weights to emphasize progression to end
    -- carries the position, segment weight, active prediction, accumulated variance, and signal life

"""

class Signal:
    def __init__ (self, position, segment_weight, feature_relevance, active_prediction, accumulated_variance, life, input_data):
        self.position = position  # Current position of the signal in the nexus
        self.segment_weight = segment_weight  # Weight assigned to the current segment
        self.feature_relevance = feature_relevance  # Feature relevance of the signal
        self.active_prediction = active_prediction  # Current prediction value
        self.accumulated_variance = accumulated_variance  # Accumulated variance of the signal
        self.life = life  # Remaining life of the signal
        self.input_data = input_data  # Placeholder for input data

    def identify_next_node(self, connected_nodes):
        # Implement logic to identify the next node for the signal
        pass

    def subtract_life(self, amount):
        self.life -= amount
        if self.life <= 0:
            self.kill_signal()

    def kill_signal(self):
        # Implement logic to kill the signal
        pass

    