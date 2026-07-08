import Components.RichConsole as RichConsole
import Components.JudgeNode as JudgeNode
import Components.HandlerNode as HandlerNode
import Components.ProcessingNode as ProcessingNode
import Components.ReviewerNode as ReviewerNode
import Components.Signal as Signal
import Components.PreProcessingNode as PreProcessingNode
import Components.SplitterNode as SplitterNode
import concurrent.futures
import pandas as pd
import random
random.seed(42)
import sys
import time
import math
from typing import Any

class Main:
    def __init__(self, mode, dimensions, connection_percentage, max_x, logger, target, classification=4, ignored_features = None):
        self.JudgeNode = None
        self.HandlerNode = None
        self.segments = []
        self.Logger = logger
        self.mode = mode
        self.Logger.log(f"Application started in {self.mode} mode.", 4, True)
        self.dimensions = dimensions
        self.connection_percentage = connection_percentage
        self.max_x = max_x
        self.ignored_features = ignored_features if ignored_features else []
        self.classification = classification  # Default classification level
        self.processing_coverage = .95
        self.y_column = target

    def display(self, message, classification = None, Loud = True):
        message = f"[Main]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        if classification is None:
            classification = self.classification
        self.Logger.log(message, classification, Loud)


    def initialize_nexus(self, Loud):
        self.display("Initializing Nexus structure.", classification=4)
        with self.Logger.make_progress() as progress:
            self.display(f"Nexus dimensions: {self.dimensions}, max_x: {self.max_x}, processing node percentage: {self.processing_coverage}", classification=4)
            if self.max_x is None:
                raise ValueError("max_x must be defined to initialize Nexus.")
            self.Judge = JudgeNode.JudgeNode(ignored_features= self.ignored_features, Logger=self.Logger, classification=self.classification)
            self.Handler = HandlerNode.HandlerNode(Logger=self.Logger, classification=self.classification)
            self.PreProcessor = PreProcessingNode.PreProcesingNode(Logger=self.Logger, logger_classification=self.classification)
            total_segments = 2 ** self.dimensions

            nexus_task   = progress.add_task("Nexus Initialization", total=3)
            segment_task = progress.add_task("Creating Segments",    total=total_segments)
            progress.update(nexus_task, advance=1, description="Creating Segments")


            def create_segment(i):
                splitter_pos = []
                reviewer_pos = []
                if self.max_x is None:
                    raise ValueError("max_x must be defined to create segments.")
                for dim in range(self.dimensions):
                    if (i >> dim) & 1:
                        splitter_pos.append(-1)
                        reviewer_pos.append(-self.max_x)
                    else:
                        splitter_pos.append(1)
                        reviewer_pos.append(self.max_x)

                splitter = SplitterNode.SplitterNode(position=splitter_pos, Logger=self.Logger, classification=self.classification, connection_percentage=self.connection_percentage, segment_id=i)
                reviewer = ReviewerNode.ReviewerNode(position=reviewer_pos, Logger=self.Logger, classification=self.classification, segment_id=i)
                full_grid    = int(self.max_x) ** self.dimensions
                usable_slots = max(1, full_grid - 2)
                num_nodes    = max(2, int(math.floor(usable_slots * self.processing_coverage)))

                def calculate_node_positions(num_nodes, dimensions, max_x, segment_id):
                    positions = set()

                    while len(positions) < num_nodes:
                        coords = []
                        for dim in range(dimensions):
                            bit = (segment_id >> dim) & 1
                            if bit == 0:
                                coords.append(random.randint(1, max_x))
                            else:
                                coords.append(random.randint(-max_x, -1))

                        pos = tuple(coords)

                        if pos != tuple(splitter_pos) and pos != tuple(reviewer_pos):
                            positions.add(pos)

                    return list(positions)

                node_positions = calculate_node_positions(num_nodes, self.dimensions, self.max_x, segment_id=i)

                for node_pos in node_positions:
                    if node_pos == tuple(splitter_pos) or node_pos == tuple(reviewer_pos):
                        raise ValueError("Processing node position conflicts with splitter or reviewer position.")


                processing_nodes = [
                    ProcessingNode.ProcessingNode(
                        position=pos,
                        Logger=self.Logger,
                        classification=self.classification,
                        segment_id=i
                    )
                    for pos in node_positions
                ]
                splitter.calculate_nearest_neighbors(processing_nodes)
                return {
                    'id': i,
                    'reviewer': reviewer,
                    'splitter': splitter,
                    'processor': processing_nodes
                }

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(create_segment, i) for i in range(total_segments)]
                for future in concurrent.futures.as_completed(futures):
                    progress.update(segment_task, advance=1)
                self.segments = [f.result() for f in futures]

            self.Judge.segments = self.segments
            progress.update(nexus_task, advance=1, description="Connecting processing nodes")

            connection_tasks = {}
            for segment in self.segments:
                connection_tasks[segment['id']] = progress.add_task(
                    f"Segment {segment['id']} Connecting",
                    total=len(segment['processor'])
                )

            def connect_segment(segment):
                node_list = list(segment['processor'] + [segment['reviewer']])
                for node in segment['processor']:
                    node.connect_nearest_nodes(node_list=node_list, connection_percentage=self.connection_percentage)
                    progress.update(connection_tasks[segment['id']], advance=1)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(connect_segment, segment) for segment in self.segments]
                concurrent.futures.wait(futures)

            for segment in self.segments:
                nodes_connected = [
                    node for node in segment['processor']
                    if segment['reviewer'] in node.connected_nodes
                ]

                self.display(f"Segment {segment['id']} - Reviewer connected to {len(nodes_connected)}/{len(segment['processor'])} processing nodes.", classification=4)
                if len(nodes_connected) == 0:
                    self.display(f"Warning: Reviewer in segment {segment['id']} has no connected processing nodes! Forcibly connecting", classification=2)
                    segment['reviewer'].force_connect_nearest_nodes(segment['processor'], self.connection_percentage)
                    nodes_connected = [node for node in segment['processor'] if segment['reviewer'] in node.connected_nodes]
                    self.display(f"After forcing, Reviewer in segment {segment['id']} connected to {len(nodes_connected)}/{len(segment['processor'])} processing nodes.", classification=4)

            progress.update(nexus_task, advance=1, description="Initialization Complete")

            self.display(f"Initialized {total_segments} segments with processing nodes.", classification=4)

    def infer(self, input, loud):
        disable_bars = not loud
        if self.Judge is None:
            raise ValueError("Nexus not initialized. Call initialize_nexus() first.")
        with self.Logger.make_progress(disable=disable_bars) as progress:
            infer_task = progress.add_task("Inference", total=4)
            self.display("Starting inference process.", classification=4, Loud=loud)
            progress.update(infer_task, description="Pre-processing input data")
            pre_processed    = self.PreProcessor.process_data(input)
            vectorized_input = self.PreProcessor.vectorize_input(pre_processed)
            progress.update(infer_task, advance=1, description="Judge node operations")

            self.display("Judge node operations.", classification=4, Loud=loud)
            relevance_scores  = self.Judge.calculate_input_segment_relevance(vectorized_input, Loud=loud)
            cluster_ids       = [c['id'] for c in relevance_scores['clusters']]
            scores            = relevance_scores['scores']

            relevance_map     = dict(zip(cluster_ids, scores))
            selected_segments = self.Judge.find_relevant_segments(relevance_scores, Loud=loud)
            self.display(f"Selected segments for inference: {[seg['id'] for seg in selected_segments]}", classification=4, Loud=loud)
            progress.update(infer_task, advance=1, description="Segment Operations")

            self.display("Segment Operations.", classification=4, Loud=loud)

            def segment_infer(segment, bar):
            #try:
                bar.set_description(f"Segment {segment['id']} Generating signals")

                splitter         = segment['splitter']
                reviewer         = segment['reviewer']
                processing_nodes = segment['processor']

                # Clear state from previous inference (matches training forwardSegment)
                for node in processing_nodes:
                    node.clear_signals()
                reviewer.signals = []

                signals = list(splitter.process(pre_processed, self.max_x))
                self.display(
                    f"Segment {segment['id']}: Generated {len(signals)} signals, "
                    f"Splitter has {len(splitter.connected_nodes)} connected nodes",
                    classification=4,
                    Loud=loud
                )
                for node, signal in zip(splitter.connected_nodes, signals):
                    node.receive_signal(signal)
                self.display(
                    f"Segment {segment['id']}: Dispatched signals to processing nodes.",
                    classification=4,
                    Loud=loud
                )
                bar.update(1)
                bar.set_description(f"Segment {segment['id']} Processing signals")

                active = True
                loop_iteration = 0
                max_iter    = 500  # Safety cap matching training forwardSegment
                num_signals = len(signals)

                while active and loop_iteration < max_iter:
                    for node in processing_nodes:
                        scaled_delta = node.process_signal()
                        if scaled_delta is not None:
                            ret = node.forward_signal()
                            # if ret is False, signal could not be forwarded
                            if not ret:
                                self.display(f"Processing node at position {node.position} could not forward signal; it may be expired.", 1, Loud=False)
                                continue
                    active = False
                    active_count = 0
                    for signal in signals:
                        if signal.is_active() and not signal.collected:
                            active = True
                            active_count += 1
                    if not signals:
                        active = False
                    loop_iteration += 1
                    if loop_iteration % 1000 == 0:
                        bar.set_description(f"Segment {segment['id']} Processing signals ({active_count}/{num_signals} active, iter {loop_iteration})")

                bar.update(1)
                bar.set_description(f"Segment {segment['id']} Reviewing signals")
                self.display(
                    f"Segment {segment['id']}: Signals processing complete after {loop_iteration} iterations.",
                    classification=4,
                    Loud=loud

                )
                self.display(
                    f"Segment {segment['id']}: Reviewer Signal successfully received percentage: {len([s for s in reviewer.signals if s.collected]) / len(reviewer.signals) if reviewer.signals else 0} ",
                    classification=4,
                    Loud=loud
                )
                segment_prediction = reviewer.review_signals()

                bar.update(1)
                bar.set_description(f"Segment {segment['id']} Complete")
                if segment_prediction is not None:
                    self.display(f"Segment {segment['id']} prediction: {segment_prediction}", classification=4, Loud=loud)
                    return segment_prediction
            """
            except Exception as e:
                self.display(f"Error in segment {segment['id']}: {e}", classification=1)
                return None
            """
            segment_tasks: dict[int, Any] = {}
            for i, segment in enumerate(selected_segments):
                segment_tasks[segment['id']] = progress.add_task(
                    f"Segment {segment['id']} Inference",
                    total=3,
                )

            segment_predictions: list[dict[str, Any]] = []
            """
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(segment_infer, segment, bars[selected_segments.index(segment)]) for segment in selected_segments]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        segment_predictions.append(result)
            """
            for segment in selected_segments:
                bar    = RichConsole.ProgressBar(progress, segment_tasks[segment['id']])
                result = segment_infer(segment, bar)
                if result is not None:
                    segment_predictions.append({
                        'id':         segment['id'],
                        'prediction': result
                    })

            progress.update(infer_task, advance=1, description="Aggregating predictions")
            self.display("Aggregating segment predictions.", classification=4, Loud=loud)
            if not segment_predictions:
                self.display("No valid segment predictions were made.", classification=2, Loud=loud)
                return None
            for segment_prediction in segment_predictions:
                if segment_prediction is None:
                    continue
                self.display(f"Segment prediction: {segment_prediction}", classification=4, Loud=loud)

                assert len(relevance_map) == len(relevance_scores['clusters'])
                self.Handler.receive_report(
                    segment_id=segment_prediction['id'],
                    segment_relevance = relevance_map.get(segment_prediction['id']),
                    prediction=segment_prediction['prediction']
                    )
            final_prediction = self.Handler.process_reports(loud = loud)
            self.display(f"Final inference prediction: {final_prediction}", classification=4, Loud=loud)
            progress.update(infer_task, advance=1, description="Inference Complete")
            return final_prediction

    def train(self, unprocessed_dataset, epochCount):
        from Train import Trainer
        trainer = Trainer(self)
        trainer.train(unprocessed_dataset, epochCount)


    def render_nexus(self):
        if self.dimensions != 2:
            self.display("Rendering is only supported for 2D Nexus.", classification=2)
            return

        import matplotlib.pyplot as plt

        self.display("Rendering Nexus structure.", classification=4)

        plt.figure(figsize=(10, 10))

        # Legend flags (only label once)
        splitter_labeled = False
        reviewer_labeled = False
        processor_labeled = False
        edge_labeled = False

        for segment in self.segments:
            splitter = segment['splitter']
            reviewer = segment['reviewer']
            processors = segment['processor']

            # --- Splitter ---
            plt.scatter(
                splitter.position[0],
                splitter.position[1],
                color='blue',
                s=60,
                label='Splitter' if not splitter_labeled else None,
                zorder=3
            )
            splitter_labeled = True

            # --- Reviewer ---
            plt.scatter(
                reviewer.position[0],
                reviewer.position[1],
                color='red',
                s=60,
                label='Reviewer' if not reviewer_labeled else None,
                zorder=3
            )
            reviewer_labeled = True

            # --- Processing nodes ---
            for node in processors:
                plt.scatter(
                    node.position[0],
                    node.position[1],
                    color='green',
                    s=10,
                    alpha=0.7,
                    label='Processing Node' if not processor_labeled else None,
                    zorder=2
                )
                processor_labeled = True

                # --- Connections ---
                for target in node.connected_nodes:
                    plt.plot(
                        [node.position[0], target.position[0]],
                        [node.position[1], target.position[1]],
                        color='red',
                        linewidth=0.5,
                        alpha=0.6,
                        label='Connection' if not edge_labeled else None,
                        zorder=1
                    )
                    edge_labeled = True
            # Render connections from splitter to processing nodes
            for target in splitter.connected_nodes:
                plt.plot(
                    [splitter.position[0], target.position[0]],
                    [splitter.position[1], target.position[1]],
                    color='red',
                    linewidth=0.5,
                    alpha=0.6,
                    label='Splitter Connection' if not edge_labeled else None,
                    zorder=1
                )
                edge_labeled = True
        plt.title("Nexus Structure (2D)")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()
        plt.savefig("nexus_structure.png")
        plt.show()

    def save_nexus_basic_info(self, filename):
        with open(filename, 'w') as f:
            f.write(f"Nexus Mode: {self.mode}\n")
            f.write(f"Dimensions: {self.dimensions}\n")
            f.write(f"Max X: {self.max_x}\n")
            f.write(f"Connection Percentage: {self.connection_percentage}\n")
            f.write(f"Total Segments: {len(self.segments)}\n")
            for segment in self.segments:
                f.write(f"\nSegment ID: {segment['id']}, Splitter Pos: {segment['splitter'].position}, Reviewer Pos: {segment['reviewer'].position}, Processing Nodes: {len(segment['processor'])}\n")
                f.write(f'Splitter Connections ({len(segment["splitter"].connected_nodes)}): {[node.position for node in segment["splitter"].connected_nodes]}\n')
                nodes_connected_to_reviewer = [node.position for node in segment['processor'] if segment['reviewer'] in node.connected_nodes]
                f.write(f'Processing Nodes connected to Reviewer: {nodes_connected_to_reviewer}\n')
                for node in segment['processor']:
                    f.write(f'  Node Pos: {node.position}, Connections: {[n.position for n in node.connected_nodes]}\n')

if __name__ == "__main__":
    max_x = 1000
    if len(sys.argv) < 2:
        sys.argv.append(input("Which run type? (demo/1-4): "))

    run_type = str(sys.argv[1]).strip().lower()

    if run_type == "demo":
        mode = "demo"
        log_level = 4  # INFO level for demo
        max_x = 25
    else:
        if run_type.isdigit() and 1 <= int(run_type) <= 4:
            log_level = int(run_type)
        else:
            log_level = 4  # Default to INFO level
        mode = "normal"

    logger = RichConsole.RichLogger(f"app_{int(time.time())}.log", log_level)

    app = Main(mode, connection_percentage=.08, dimensions= 2, target='exam_score', max_x=max_x, logger=logger)

    # Print options to the user including Load Config, Start Nexus, Run Training, Run Inference
    if mode != "demo":
        print("Options:")
        print("1. Load Config")
        print("2. Start Nexus")
        print("3. Run Training")
        print("4. Run Inference")
        print("5. render Nexus (2d)")
        choice = input("Select an option (1-5): ")
        app.Logger.log(f"User selected option {choice}.", 4, True)

        if choice == "1":
            app.Logger.log("Load Config selected. (Not yet implemented)", 4, True)
        elif choice == "2":
            app.initialize_nexus(Loud=True)

        elif choice == "3":
            app.Logger.log("Training mode selected.", 4, True)
            dataset = pd.read_csv("Exam_Score_Prediction.csv")
            app.train(dataset, epochCount=3)
        elif choice == "4":
            app.Logger.log("Inference mode selected. (Not yet implemented)", 4, True)
        elif choice == "5":
            app.render_nexus()

    else:
        app.Logger.log("Running in demo mode. Skipping user options.", 4, True)
        app.Logger.log("Initializing Nexus for demo.", 4, True)
        time.sleep(1)
        app.initialize_nexus(Loud=True)
        app.mode = "demo"
        app.Judge.mode = "demo"

        app.Logger.log("Nexus initialized for demo.", 4, True)
        app.Logger.log("Saving Nexus basic info to file.", 4, True)
        time.sleep(1)
        app.save_nexus_basic_info("nexus_basic_info.txt")


        if max_x <= 15:
            app.Logger.log("Rendering Nexus structure for demo.", 4, True)
            time.sleep(1)
            app.render_nexus()

        app.Logger.log("Preparing demo dataset for inference.", 4, True)
        time.sleep(1)
        dataset = pd.read_csv("Exam_Score_Prediction.csv")
        df = app.PreProcessor.process_dataset(dataset)
        app.Logger.log("Processed dataset.", 4, True)

        app.Logger.log("Running inference on a sample input from the dataset.", 4, True)
        time.sleep(1)
        sample_input = dataset.iloc[0].to_dict()
        app.Logger.log(f"Sample input for inference: {sample_input}", 4, True)
        prediction = app.infer(sample_input, loud = True)
        app.Logger.log(f"Inference prediction: {prediction}", 4, True)
        app.Logger.log(f"Percent Error: {abs(prediction - sample_input[app.y_column]) / sample_input[app.y_column] * 100:.2f}%", 4, True)

        app.Logger.log("Will begin training phase now.", 4, True)
        time.sleep(1)
        app.train(dataset, 3)

        app.Logger.log("Demo run complete.", 4, True)
    app.Logger.log("Application finished execution.", 4, True)
