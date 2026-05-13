# Code required for the nexus system handler
from SegmentHandler import SegmentHandler
from Components.JudgeNode import JudgeNode
from Components.Logger import Logger
from Components.ReviewerNode import ReviewerNode
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

    def initializeAllSegments(self):
        segmentCount = 2 ** self.dimensions
        for i in range(segmentCount):
            self.segments.append(SegmentHandler(maxX=self.max_x, target=self.target, logger=self.logger, connection_percentage=self.connection_percentage, density=self.density, dimensions=self.dimensions, classification=self.classification, segment_id=i))
        for segment in self.segments:
            segment.initializeSegment()
