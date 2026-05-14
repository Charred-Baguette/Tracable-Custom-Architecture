# Code required for the nexus system handler
from asyncio.log import logger

from SegmentHandler import SegmentHandler
from Components.JudgeNode import JudgeNode
from Components.Logger import Logger
from Components.ReviewerNode import ReviewerNode
from Components.PreProcessingNode import PreProcesingNode, PreProcessingNode
from Components.HandlerNode import HandlerNode

class SystemHandler:
    def __init__(self, maxX, target='exam_score', logger = None, connection_percentage=.08, density = .95, dimensions = 2, classification = 1):
        self.dimensions = dimensions
        self.max_x = maxX
        self.target = target
        self.logger = logger
        self.dimensions = dimensions
        self.connection_percentage = connection_percentage
        self.density = density
        self.classification = classification
        self.segments = []
        self.JudgeNode = JudgeNode(logger=self.logger, target=self.target, classification=self.classification)
        self.HandlerNode = HandlerNode(logger=self.logger, target=self.target, classification=self.classification)
        self.preprocessor = PreProcesingNode(Logger=logger, logger_classification=4)


    def display(self, message, classification = None, Loud = True):
        message = f"[Main]: {message}"
        if self.logger is None:
            raise ValueError("Logger not assigned")
        if classification is None:
            classification = self.classification
        self.logger.log(message, classification, Loud)

    def initializeAllSegments(self, Loud = False):
        segmentCount = 2 ** self.dimensions
        for i in range(segmentCount):
            self.segments.append(SegmentHandler(maxX=self.max_x, target=self.target, logger=self.logger, connection_percentage=self.connection_percentage, density=self.density, dimensions=self.dimensions, classification=self.classification, segment_id=i))
        for segment in self.segments:
            segment.initializeSegment()
        
        self.display(f"Initialized {segmentCount} segments", Loud=Loud)

    def runInfer(self, input, loud = True):
        if self.JudgeNode is None or self.HandlerNode is None:
            raise ValueError("JudgeNode or HandlerNode not assigned")
        
        if self.JudgeNode.segment_weights.shape[0] != len(self.segments):
            raise ValueError("JudgeNode segment weights do not match number of segments")
        
        self.display("Running inference on JudgeNode", Loud=loud)
        pre_input = self.preprocessor.process_data(input)
        relevance_scores = self.JudgeNode.infer(pre_input, self.segments)
        selected_segments = self.JudgeNode.find_relevant_segments(relevance_scores)
        self.display(f"Selected segments: {selected_segments}", Loud=loud)


        for segment_id in selected_segments:
            segment = self.segments[segment_id]
            