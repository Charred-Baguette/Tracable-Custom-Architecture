"""
from notes:
 - splitter : takes data from judge and passes to closest 5% of processing nodes
  -- splitters are divided based on 2d quadrant or 3d octant
  -- currently no logic to how splitters get different values - judge manages that
  -- gets a feature and passes to processing nodes
  -- intent is generic routing as they create signals
  -- splitters must not be purely feature based
"""

from .BaseNode import BaseNode
import numpy as np
from .Signal.Signal import Signal

class SplitterNode(BaseNode):
    def __init__(self, position, nodes_in_segment, dimenstions, segment_id):
        self.position = position  # Position in the nexus (likely 1 from origin (1,1) or (-1, -1))
        self.closest_nodes = []  # List of closest nodes for routing
        self.nodes_in_segment = nodes_in_segment  # Nodes in the same segment
        self.all_node_count = len(nodes_in_segment)
        self.dimenstions = dimenstions  # Number of dimensions (2D or 3D)
        self.segment_id = segment_id  # Segment identifier


    def compute_closest_nodes(self, all_node_count, nodes_in_segment, percent=0.05):
        # Compute Euclidean distance to each node
        distances = []
        for node in nodes_in_segment:
            node_pos = np.array(getattr(node, 'position', (0, 0)))
            splitter_pos = np.array(self.position)
            dist = np.linalg.norm(splitter_pos - node_pos)
            distances.append((node, dist))
        # Sort by distance and select closest N%
        distances.sort(key=lambda x: x[1])
        n_closest = max(1, int(len(nodes_in_segment) * percent))
        self.closest_nodes = [node for node, _ in distances[:n_closest]]

    def handle_carrier(self, carrier_data):
        segment_features = carrier_data.get('segment_feature_relevance', {})
        feature_relevance = segment_features.get(
            self.segment_id,
            carrier_data.get('feature_relevance', {})
        )
        segment_relevance = carrier_data.get('segment_relevance', {}).get(self.segment_id, 1.0)
        return segment_relevance, feature_relevance

    def process(self, carrier_data):
        # For each node, create a new signal using the carrier data
        segment_relevance, feature_relevance = self.handle_carrier(carrier_data)
        created_signals = []
        for i in self.closest_nodes:
            signal = Signal(
                position=self.position,
                segment_weight=segment_relevance,
                feature_relevance=feature_relevance,
                active_prediction=None,
                accumulated_variance=0.0,
                life=10,  # Arbitrary initial life value
                input_data=carrier_data.get('input_data', None)
            )
            created_signals.append(signal)
            i.receive_signal(signal)
        return created_signals
    

