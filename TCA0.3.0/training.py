from tqdm import tqdm
import time
import math

class training:
    def __init__(self, app, epochCount=3):
        self.app = app
        self.epochCount = epochCount

        # Training hyperparameters
        self.initial_lr = 0.01         # Learning rate for initial gradient pass
        self.position_lr = 0.005       # Learning rate for position optimization
        self.weight_lr = 0.001         # Smaller alpha for weight phase
        self.lr_decay = 0.8            # Decay factor applied to lr each oscillation
        self.convergence_threshold = 1e-4
        self.max_oscillations = epochCount  # Reuse epochCount for oscillation budget
        self.path_radius = 1.0         # Circle radius for path nodes during position phase
        self.neighbor_radius = 0.3     # Smaller radius for connected non-path neighbors
        self.grad_clip = 1.0           # Max absolute value for dL/dpred before backprop

        if self.app.Judge is None:
            raise ValueError("Nexus not initialized. Call initialize_nexus() first.")
        
    # This method will be used for initializing any necessary components or data structures before training begins in the traditional train methodology
    def PrepTradTraining(self, unprocessed_dataset, batch=False):
        with tqdm(total=2 if batch else 1, desc="Initializing Training Preparations", leave=True) as init_pbar:
            init_pbar.set_description("Preprocessing dataset")
            df = unprocessed_dataset.copy()

            y = df.pop(self.app.y_column).to_numpy()
            x = self.app.PreProcessor.process_dataset(df)

            assert len(x) == len(y)
            init_pbar.update(1)

            return x, y
            
            
    # This method will be used for training the entire system end-to-end in the traditional train methodology
    def totalTrain(self, x, y):
        self.app.Logger.log("Starting total training pipeline", 4, True)
        num_segments = len(self.app.segments)

        with tqdm(total=4, desc="Training Pipeline", leave=True) as pipeline_pbar:
            # Phase 1: Train Judge node clusters
            pipeline_pbar.set_description("Phase 1: Judge Training")
            self.app.Logger.log("Phase 1: Training global master components (Judge node)", 4, True)
            self.globalComponentTrain(x, y)
            pipeline_pbar.update(1)

            # Phase 2: End-to-end gradient calculation for initial weights
            pipeline_pbar.set_description("Phase 2: Gradient Weight Init")
            self.app.Logger.log("Phase 2: End-to-end gradient weight initialization", 4, True)
            self.initializeNodeWeights(x)
            avg_loss = self.gradientWeightPhase(x, y)
            self.app.Logger.log(f"Initial gradient pass complete. Avg loss: {avg_loss:.6f}", 4, True)
            pipeline_pbar.update(1)

            # Phase 3: Per-segment multi-epoch oscillation with decaying lr
            pipeline_pbar.set_description("Phase 3: Segment Oscillation")
            self.app.Logger.log("Phase 3: Beginning per-segment position-weight oscillation", 4, True)
            with tqdm(total=num_segments, desc="Segment Training", leave=True) as seg_pbar:
                for segment in self.app.segments:
                    seg_pbar.set_description(f"Segment {segment['id']} Training")
                    self.segmentTrain(segment, x, y, connection_percentage=self.app.connection_percentage)
                    seg_pbar.update(1)
            pipeline_pbar.update(1)

            # Phase 4: Prune unused nodes
            pipeline_pbar.set_description("Phase 4: Pruning")
            self.app.Logger.log("Phase 4: Pruning unused nodes", 4, True)
            self.pruneUnusedNodes()
            pipeline_pbar.update(1)

            pipeline_pbar.set_description("Training Complete")

        self.app.Logger.log("Training pipeline complete", 4, True)
    
    # This method will be used for training individual segments of the system
    def segmentTrain(self, segment, x, y, connection_percentage):
        """Train a single segment across multiple epochs, oscillating between
        position and weight optimization with a decaying learning alpha."""
        pos_lr = self.position_lr
        wt_lr = self.weight_lr
        prev_loss = float('inf')
        seg_id = segment['id']

        with tqdm(total=self.epochCount, desc=f"Seg {seg_id} Epochs", leave=False) as epoch_pbar:
            for epoch in range(self.epochCount):
                # Position phase for this segment
                epoch_pbar.set_description(f"Seg {seg_id} E{epoch+1} Position (lr={pos_lr:.6f})")
                pos_loss = self.segmentPositionPass(segment, x, y, pos_lr, connection_percentage)
                self.app.Logger.log(
                    f"Segment {seg_id} Epoch {epoch+1} - Position loss: {pos_loss:.6f} (lr={pos_lr:.6f})", 4, True
                )

                # Weight phase for this segment
                epoch_pbar.set_description(f"Seg {seg_id} E{epoch+1} Weights (lr={wt_lr:.6f})")
                wt_loss = self.segmentWeightPass(segment, x, y, wt_lr)
                self.app.Logger.log(
                    f"Segment {seg_id} Epoch {epoch+1} - Weight loss: {wt_loss:.6f} (lr={wt_lr:.6f})", 4, True
                )

                # Decay learning rates
                pos_lr *= self.lr_decay
                wt_lr *= self.lr_decay

                # Check convergence
                rate_of_change = abs(prev_loss - wt_loss)
                self.app.Logger.log(
                    f"Segment {seg_id} Epoch {epoch+1} - Rate of change: {rate_of_change:.6f}", 4, True
                )
                if rate_of_change < self.convergence_threshold:
                    self.app.Logger.log(
                        f"Segment {seg_id} converged after {epoch+1} epochs", 4, True
                    )
                    epoch_pbar.update(self.epochCount - epoch)
                    break

                prev_loss = wt_loss
                epoch_pbar.update(1)
        
    # This method is used for the global master components: currently only Judge node
    def globalComponentTrain(self, x, y):
        self.app.Logger.log("Training global master components (Judge node)", 4, True)
        self.app.Judge.train(x, self.epochCount)

    # ── Initialization ──────────────────────────────────────────────────
    def initializeNodeWeights(self, x):
        """Initialize weights on all processing nodes using feature names from preprocessed data"""
        sample = x.iloc[0].to_dict() if hasattr(x, 'iloc') else x[0]
        features = list(sample.keys()) if isinstance(sample, dict) else list(range(len(sample)))

        for segment in self.app.segments:
            for node in segment['processor']:
                node.initialize_weights(features)

    # ── Forward Pass ────────────────────────────────────────────────────
    def forwardSegment(self, segment, input_data, max_x):
        """Run a single training sample through a segment.
        Mirrors the structure of Main.infer()'s segment_infer, but uses
        train_process_signal (which records path_contributions for gradients).
        Returns (prediction, signals)."""
        splitter = segment['splitter']
        reviewer = segment['reviewer']
        processing_nodes = segment['processor']

        # Clear state from previous sample (matches inference clearing)
        for node in processing_nodes:
            node.clear_signals()
        reviewer.signals = []

        # Generate and dispatch signals (same as segment_infer)
        signals = list(splitter.process(input_data, max_x))
        for node, signal in zip(splitter.connected_nodes, signals):
            node.receive_signal(signal)

        # Process until all signals are collected or expired
        # Mirrors segment_infer loop structure with a safety cap
        active = True
        iteration = 0
        max_iter = 500  # Safety cap (inference has none, but training repeats many times)
        while active and iteration < max_iter:
            for node in processing_nodes:
                scaled_delta = node.train_process_signal()
                if scaled_delta is not None:
                    ret = node.forward_signal()
                    if not ret:
                        # Signal could not be forwarded (expired or no viable nodes)
                        # Matches inference behavior of continuing past failed forwards
                        continue

            # Active check — same logic as segment_infer
            active = False
            for signal in signals:
                if signal.is_active() and not signal.collected:
                    active = True
                    break
            if not signals:
                active = False
            iteration += 1

        prediction = reviewer.review_signals()
        return prediction, signals

    # ── Phase 2: End-to-end gradient weight calculation ─────────────────
    def gradientWeightPhase(self, x, y):
        """Compute end-to-end gradients and determine initial weight for each node"""
        total_loss = 0.0
        n_samples = len(y)

        with tqdm(total=n_samples, desc="Gradient Weight Phase", leave=True) as pbar:
            for i in range(n_samples):
                sample = x.iloc[i].to_dict() if hasattr(x, 'iloc') else x[i]
                target = y[i]

                segment_preds = []
                all_signals = []
                for segment in self.app.segments:
                    pred, signals = self.forwardSegment(segment, sample, self.app.max_x)
                    if pred is not None:
                        segment_preds.append(pred)
                        all_signals.extend(signals)

                if not segment_preds:
                    pbar.update(1)
                    continue

                prediction = sum(segment_preds) / len(segment_preds)
                loss = (prediction - target) ** 2 / 2.0
                total_loss += loss
                dL_dpred = (prediction - target) / len(segment_preds)
                dL_dpred = max(-self.grad_clip, min(self.grad_clip, dL_dpred))  # Clip gradient

                # Backpropagate to each node in each signal path
                for signal in all_signals:
                    for node_id, contrib in signal.path_contributions.items():
                        node = contrib['node']
                        node.accumulate_weight_gradient(dL_dpred, signal)

                pbar.update(1)

        # Apply accumulated gradients
        for segment in self.app.segments:
            for node in segment['processor']:
                node.apply_weight_gradient(self.initial_lr)

        return total_loss / max(n_samples, 1)

    # ── Segment-level position pass ──────────────────────────────────────
    def segmentPositionPass(self, segment, x, y, lr, connection_percentage):
        """Position optimization for a single segment over all samples.
        Path nodes get a larger circle radius; connected non-path neighbors a smaller one."""
        total_loss = 0.0
        n_samples = len(y)
        path_node_ids = set()

        for i in range(n_samples):
            sample = x.iloc[i].to_dict() if hasattr(x, 'iloc') else x[i]
            target = y[i]

            pred, signals = self.forwardSegment(segment, sample, self.app.max_x)
            if pred is None:
                continue

            loss = (pred - target) ** 2 / 2.0
            total_loss += loss
            dL_dpred = pred - target
            dL_dpred = max(-self.grad_clip, min(self.grad_clip, dL_dpred))  # Clip gradient

            for signal in signals:
                for node_id, contrib in signal.path_contributions.items():
                    node = contrib['node']
                    path_node_ids.add(id(node))
                    node.accumulate_position_gradient(dL_dpred, signal)

        # Apply position gradients with circle constraints
        for node in segment['processor']:
            if id(node) in path_node_ids:
                node.apply_position_gradient(lr, self.path_radius)
            else:
                is_neighbor = any(id(n) in path_node_ids for n in node.connected_nodes)
                if is_neighbor:
                    node.apply_position_gradient(lr, self.neighbor_radius)
                else:
                    node.reset_gradients()

        self.reconnectSegment(segment, connection_percentage)
        return total_loss / max(n_samples, 1)

    # ── Segment-level weight pass ───────────────────────────────────────
    def segmentWeightPass(self, segment, x, y, lr):
        """Weight optimization for a single segment over all samples."""
        total_loss = 0.0
        n_samples = len(y)

        for i in range(n_samples):
            sample = x.iloc[i].to_dict() if hasattr(x, 'iloc') else x[i]
            target = y[i]

            pred, signals = self.forwardSegment(segment, sample, self.app.max_x)
            if pred is None:
                continue

            loss = (pred - target) ** 2 / 2.0
            total_loss += loss
            dL_dpred = pred - target
            dL_dpred = max(-self.grad_clip, min(self.grad_clip, dL_dpred))  # Clip gradient

            for signal in signals:
                for node_id, contrib in signal.path_contributions.items():
                    node = contrib['node']
                    node.accumulate_weight_gradient(dL_dpred, signal)

        for node in segment['processor']:
            node.apply_weight_gradient(lr)

        return total_loss / max(n_samples, 1)

    # ── Reconnection helpers ────────────────────────────────────────────
    def reconnectSegment(self, segment, connection_percentage):
        """Rebuild connections after position updates and verify end-to-end
        reachability from the splitter to the reviewer via BFS."""
        reviewer = segment['reviewer']
        splitter = segment['splitter']
        processing_nodes = segment['processor']
        node_list = list(processing_nodes) + [reviewer]

        # 1. Recalculate distances and rebuild outward connections
        for node in processing_nodes:
            node.connected_nodes = []
            node.distance_to_origin = sum(p ** 2 for p in node.position) ** 0.5
            node.connect_nearest_nodes(node_list=node_list, connection_percentage=connection_percentage)

        # 2. Re-link splitter to nearest processing nodes
        splitter.calculate_nearest_neighbors(processing_nodes)

        # 3. Ensure at least some processing nodes connect to the reviewer
        nodes_connected_to_rev = [n for n in processing_nodes if reviewer in n.connected_nodes]
        if not nodes_connected_to_rev:
            reviewer.force_connect_nearest_nodes(processing_nodes, connection_percentage)

        # 4. BFS from splitter's connected nodes outward to verify reviewer is reachable
        reachable = set()
        frontier = list(splitter.connected_nodes)
        for node in frontier:
            reachable.add(id(node))
        while frontier:
            current = frontier.pop(0)
            for neighbor in current.connected_nodes:
                if id(neighbor) not in reachable:
                    reachable.add(id(neighbor))
                    if neighbor is not reviewer:  # don't expand past reviewer
                        frontier.append(neighbor)

        if id(reviewer) not in reachable:
            # Reviewer unreachable — bridge the gap by connecting the
            # furthest reachable node to the nearest node that connects to reviewer
            nodes_with_rev = [n for n in processing_nodes if reviewer in n.connected_nodes]
            reachable_nodes = [n for n in processing_nodes if id(n) in reachable]

            if reachable_nodes and nodes_with_rev:
                # Find the reachable node closest to any reviewer-connected node
                import math as _math
                best_pair = None
                best_dist = float('inf')
                for rn in reachable_nodes:
                    for rn2 in nodes_with_rev:
                        if id(rn2) in reachable:
                            continue  # already reachable
                        d = _math.dist(rn.position, rn2.position)
                        if d < best_dist:
                            best_dist = d
                            best_pair = (rn, rn2)
                if best_pair:
                    src, dst = best_pair
                    if dst not in src.connected_nodes:
                        src.connected_nodes.append(dst)
            elif reachable_nodes:
                # No node connects to reviewer; force-connect the furthest reachable node
                furthest = max(reachable_nodes, key=lambda n: n.distance_to_origin)
                if reviewer not in furthest.connected_nodes:
                    furthest.connected_nodes.append(reviewer)
            
    def reconnectNodes(self, connection_percentage):
        """Reconnect all segments after position updates"""
        for segment in self.app.segments:
            self.reconnectSegment(segment, connection_percentage)

    # ── Phase 4: Pruning ────────────────────────────────────────────────
    def pruneUnusedNodes(self):
        """Remove processing nodes that were never activated during training"""
        total_pruned = 0
        for segment in self.app.segments:
            original_count = len(segment['processor'])
            segment['processor'] = [
                node for node in segment['processor']
                if node.activation_count > 0
            ]
            pruned = original_count - len(segment['processor'])
            total_pruned += pruned
            if pruned > 0:
                self.app.Logger.log(
                    f"Segment {segment['id']}: Pruned {pruned}/{original_count} unused nodes",
                    4, True
                )

        if total_pruned > 0:
            self.reconnectNodes(self.app.connection_percentage)

        self.app.Logger.log(f"Total nodes pruned: {total_pruned}", 4, True)

if __name__ == "__main__":
    import Main
    import pandas as pd
    import Components.Logger as Logger
    logger = Logger.Logger(f"Train_{int(time.time())}.log", log_level=4)
    app = Main.Main('demo', connection_percentage=.08, dimensions= 2, target='exam_score', max_x=10, logger=logger)
    app.initialize_nexus(Loud = False)
    trainingD = training(app, epochCount=3)
    dataset = pd.read_csv("Exam_Score_Prediction.csv")
    prep = trainingD.PrepTradTraining(dataset, batch=False)
    # Save the preprocessed dataset for later use in training and inference as file
    writer = pd.ExcelWriter('preprocessed_dataset.xlsx', engine='xlsxwriter')
    pd.DataFrame(prep[0]).to_excel(writer, sheet_name='X', index=False)
    pd.DataFrame(prep[1], columns=[app.y_column]).to_excel(writer, sheet_name='Y', index=False)
    writer.close()

    x, y = prep

    # ── Segment Learning Demo ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SEGMENT LEARNING PROCESS DEMONSTRATION")
    print("=" * 70)

    # Phase 1: Global component training (Judge)
    print("\n--- Phase 1: Global Component Training (Judge) ---")
    trainingD.globalComponentTrain(x, y)
    print("Judge node clustering complete.")

    # Phase 2: Initialize weights & run initial gradient pass
    print("\n--- Phase 2: Weight Initialization & Gradient Pass ---")
    trainingD.initializeNodeWeights(x)
    avg_loss = trainingD.gradientWeightPhase(x, y)
    print(f"Initial gradient pass avg loss: {avg_loss:.6f}")

    # Phase 3: Per-segment oscillation demo
    # Show detailed per-segment training with position-weight oscillation
    print("\n--- Phase 3: Per-Segment Position-Weight Oscillation ---")
    num_segments = len(app.segments)
    print(f"Total segments to train: {num_segments}")

    for seg_idx, segment in enumerate(app.segments):
        seg_id = segment['id']
        num_proc = len(segment['processor'])
        print(f"\n{'─' * 60}")
        print(f"Segment {seg_id}  |  Processing nodes: {num_proc}")
        print(f"  Splitter pos: {segment['splitter'].position}")
        print(f"  Reviewer pos: {segment['reviewer'].position}")

        # Snapshot positions before training
        pre_positions = {id(n): tuple(n.position) for n in segment['processor']}

        # Run a single forward pass on first sample to show signal flow
        sample = x.iloc[0].to_dict() if hasattr(x, 'iloc') else x[0]
        target = y[0]
        pred_before, signals_before = trainingD.forwardSegment(segment, sample, app.max_x)
        print(f"  Pre-train forward pass  → prediction: {pred_before}, target: {target}")
        if pred_before is not None:
            print(f"    Loss: {((pred_before - target) ** 2 / 2.0):.6f}")
            print(f"    Signals generated: {len(signals_before)}")
            active_paths = sum(1 for s in signals_before if s.collected)
            print(f"    Signals reaching reviewer: {active_paths}/{len(signals_before)}")

        # Train this segment (position-weight oscillation across epochs)
        print(f"\n  Training segment {seg_id} ({trainingD.epochCount} epochs) ...")
        trainingD.segmentTrain(segment, x, y, connection_percentage=app.connection_percentage)

        # Post-training snapshot
        post_positions = {id(n): tuple(n.position) for n in segment['processor']}

        # Report position movement
        moved_count = 0
        total_displacement = 0.0
        for nid, pre_pos in pre_positions.items():
            post_pos = post_positions.get(nid)
            if post_pos is not None and pre_pos != post_pos:
                moved_count += 1
                disp = sum((a - b) ** 2 for a, b in zip(pre_pos, post_pos)) ** 0.5
                total_displacement += disp
        print(f"\n  Post-train node movement:")
        print(f"    Nodes moved: {moved_count}/{num_proc}")
        if moved_count > 0:
            print(f"    Avg displacement: {total_displacement / moved_count:.6f}")

        # Run forward pass again on the same sample to compare
        pred_after, signals_after = trainingD.forwardSegment(segment, sample, app.max_x)
        print(f"  Post-train forward pass → prediction: {pred_after}, target: {target}")
        if pred_after is not None:
            loss_after = (pred_after - target) ** 2 / 2.0
            print(f"    Loss: {loss_after:.6f}")
            if pred_before is not None:
                loss_before = (pred_before - target) ** 2 / 2.0
                improvement = loss_before - loss_after
                print(f"    Loss change: {improvement:+.6f} ({'improved' if improvement > 0 else 'worsened'})")

    # Phase 4: Pruning
    print(f"\n{'─' * 60}")
    print("--- Phase 4: Pruning Unused Nodes ---")
    total_before = sum(len(seg['processor']) for seg in app.segments)
    trainingD.pruneUnusedNodes()
    total_after = sum(len(seg['processor']) for seg in app.segments)
    print(f"Nodes before pruning: {total_before}")
    print(f"Nodes after pruning:  {total_after}")
    print(f"Total pruned:         {total_before - total_after}")

    # Final end-to-end evaluation on a few samples
    print(f"\n{'=' * 70}")
    print("  POST-TRAINING EVALUATION (first 5 samples)")
    print("=" * 70)
    eval_count = min(5, len(y))
    total_error = 0.0
    for i in range(eval_count):
        sample = x.iloc[i].to_dict() if hasattr(x, 'iloc') else x[i]
        target_val = y[i]
        seg_preds = []
        for segment in app.segments:
            pred, _ = trainingD.forwardSegment(segment, sample, app.max_x)
            if pred is not None:
                seg_preds.append(pred)
        if seg_preds:
            final_pred = sum(seg_preds) / len(seg_preds)
            error = abs(final_pred - target_val)
            pct_error = (error / abs(target_val) * 100) if target_val != 0 else float('inf')
            total_error += pct_error
            print(f"  Sample {i}: predicted={final_pred:.4f}, actual={target_val}, error={pct_error:.2f}%")
        else:
            print(f"  Sample {i}: No predictions produced")
    if eval_count > 0:
        print(f"\n  Mean percent error: {total_error / eval_count:.2f}%")

    print(f"\n{'=' * 70}")
    print("  SEGMENT LEARNING DEMO COMPLETE")
    print("=" * 70)
    logger.log("Segment learning demo finished.", 4, True)