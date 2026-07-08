from Components.Signal import Signal
class SplitterNode:
    def __init__(self, position, connection_percentage, segment_id, Logger=None, classification=4):
        self.position = position
        self.connection_percentage = connection_percentage
        self.signal_weights = {}
        self.connected_nodes = []
        self.Logger = Logger
        self.classification = classification
        self.classification = None  
        segment_id = segment_id
    
    def __repr__(self) -> str:
        return f"Splitter(pos={self.position})"
    
    def display(self, message):
        message = f"[SplitterNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        self.Logger.log(message, self.classification, True)
        

    def calculate_nearest_neighbors(self, nodes):
        distances = {}
        self.connected_nodes = []
        for node in nodes:
            for _ in self.position:
                distance = sum((p1 - p2) ** 2 for p1, p2 in zip(self.position, node.position)) ** 0.5
            distances[node] = distance
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        
        while len(self.connected_nodes) < len(nodes) * self.connection_percentage:
            nearest_node = next(iter(sorted_distances))
            self.connected_nodes.append(nearest_node)
            del sorted_distances[nearest_node]
        return self.connected_nodes
    
    def initialize_signal_weights(self, input_data):
        for feature in input_data:
            self.signal_weights[feature] = 1.0  # Initial weight of 1.0 for each feature

    
    def generate_signals(self, input_data, max_x, feature_relevance):
        signals = []
        for _ in self.connected_nodes:
            signal = Signal(
                position=self.position,
                prediction=0.0,
                input=input_data,
                variance=0.0,
                feature_relevance=feature_relevance,
                max_x=max_x
            )
            signals.append(signal)
        
        return signals
    
    def calculate_feature_relevance(self, input_data):
        #placeholder for now
        feature_relevance = {}
        for feature in input_data:
            feature_relevance[feature] = 1.0  # Equal relevance for all features
        return feature_relevance

    def update_signal_weights(self, weight):
        for weight in self.signal_weights:
            self.signal_weights[weight] = weight

    def process(self, input_data, max_x):
        feature_relevance = self.calculate_feature_relevance(input_data)
        signals = self.generate_signals(input_data, max_x, feature_relevance)
        return signals
    
