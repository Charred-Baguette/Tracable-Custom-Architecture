import Nodes.BaseNode as BaseNode
import Nodes.JudgeNode as JudgeNode
import Nodes.Handler as HandlerNode
import Nodes.Splitter as SplitterNode
import Nodes.Reviewer as ReviewerNode
import Nodes.Processing as ProcessingNode


class main:
    def __init__(self):
        self.dataset = None
        self.segments = []
        self.segments_count = 0
        self.judge_node = None
        self.handler_node = None

    def load_dataset(self, dataset):
        self.dataset = dataset
        self.dataset_features = dataset.columns.tolist()
        
    def initialize_base_framework(self, dimensions=2):
        # Initialize Judge Node at origin
        self.judge_node = JudgeNode.JudgeNode(position=(0,) * dimensions)
        self.judge_node.set_dataset_features(self.dataset_features)
        
        # Initialize Handler Node at origin
        self.handler_node = HandlerNode.HandlerNode(position=(0,) * dimensions)

    def create_default_segments(self, dimensions):
        self.segments = []
        self.segments_count = 2 ** dimensions
        for i in range(self.segments_count):
            segment = {
                'index': i,
                'splitter': SplitterNode.SplitterNode(position=(0,) * dimensions),
                'processing_nodes': [],
                'reviewer_node': ReviewerNode.ReviewerNode(position=(0,) * dimensions)
            }
            self.segments.append(segment)
    
    def set_custom_segments(self, segments):
        self.segments = segments
        self.segments_count = len(segments)