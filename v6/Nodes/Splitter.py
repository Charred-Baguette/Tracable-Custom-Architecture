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

class SplitterNode(BaseNode):
    def __init__(self, position):
        self.position = position  # Position in the nexus (likely 1 from origin (1,1) or (-1, -1))
        self.closest_nodes = []  # List of closest nodes for routing
    def compute_closest_nodes(self, all_node_count, nodes_in_segment):
        # Implement logic to find and set the closest n nodes in same segment based on the current position
        pass
    def process(self, data):
        # Implement the processing logic for the SplitterNode
        pass
