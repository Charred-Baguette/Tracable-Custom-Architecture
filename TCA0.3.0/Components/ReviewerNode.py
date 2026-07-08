import math


class ReviewerNode:
    def __init__(self, position, segment_id, Logger=None, classification=4):
        self.position = position
        self.segment_id = segment_id
        self.signals = []
        self.Logger = Logger
        self.classification = classification  
        self.distance_to_origin = math.sqrt(sum(p ** 2 for p in position))
    def __repr__(self) -> str:
        return f"Reviewer(pos={self.position})"
    
    def display(self, message, classification = None, Loud = True):
        message = f"[ReviewerNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        if classification is None:
            classification = self.classification
        self.Logger.log(message, classification, Loud)

    def receive_signal(self, signal):
        signal.signal_life = 5  # Keep Signal alive 
        signal.collected = True
        self.signals.append(signal)
        self.display(f"Received signal at position {signal.position} with prediction {signal.prediction}.", 4, Loud=False)
        return True
    
    def review_signals(self):
        weighted_sum = 0.0
        weight_sum = 0.0

        for signal in self.signals:
            if not signal.is_active():
                continue

            w = 1.0 / max(signal.variance, 1e-9)
            weighted_sum += signal.prediction * w
            weight_sum += w

        if weight_sum == 0:
            self.display("No valid weights to aggregate.", 2, Loud=False)
            return None

        final_prediction = weighted_sum / weight_sum
        self.signals = [] # Clear signals after review
        return final_prediction

    def force_connect_nearest_nodes(self, node_list, connection_percentage):
        closest_nodes = sorted(
            node_list,
            key=lambda n: math.dist(self.position, n.position)
        )[:max(1, int(len(node_list) * connection_percentage))]

        for node in closest_nodes:
            if self not in node.connected_nodes:
                node.connected_nodes.append(self)
        return closest_nodes
            
            
        