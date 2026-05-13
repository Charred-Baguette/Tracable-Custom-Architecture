class HandlerNode:
    def __init__(self, Logger=None, classification=4):
        self.reports = {
            'segment': [],
            'segment_relevance': [],
            'predictions': []
        }
        self.Logger = Logger  # To be assigned externally
        self.classification = classification

    def display(self, message, Loud=False):
        message = f"[HandlerNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        self.Logger.log(message, self.classification, Loud)
    def receive_report(self, segment_id, segment_relevance, prediction):
        self.reports['segment'].append(segment_id)
        self.reports['segment_relevance'].append(segment_relevance)
        self.reports['predictions'].append(prediction)

    def process_reports(self, loud):
        predictions_weighted = []
        for relevance, prediction in zip(self.reports['segment_relevance'], self.reports['predictions']):
            predictions_weighted.append(prediction * relevance)

        if not predictions_weighted:
            self.display("No predictions to process.", Loud=loud)
            return None
        final_prediction = sum(predictions_weighted) / len(predictions_weighted)
        self.display(f"Final aggregated prediction: {final_prediction}", Loud = loud)
        self.reports = {
            'segment': [],
            'segment_relevance': [],
            'predictions': []
        }
        return final_prediction

        

        
        