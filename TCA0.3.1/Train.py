class Trainer:
    """
    Trains the Nexus network following the updated plan (2/10/26):

      Step 1 – Pre-process the raw dataset via PreProcessingNode.
      Step 2 – Initialise all processing-node weights.
      Step 3 – Alternating training phases until convergence or epoch limit:
                  A. Weight phase   : end-to-end gradient descent on node weights.
                  B. Position phase : treat node position as a learnable feature,
                                      use gradients to move each node within a
                                      bounded "circle" around its current location.
                                      Connected neighbours not on the active path
                                      receive a proportionally smaller step radius.
                  A'. Weight refinement : re-train weights with a smaller LR now
                                          that positions (and therefore connections)
                                          may have changed.
                  After each full epoch, evaluate progress via Main.infer() on a
                  held-out sample as directed by the plan.
      Step 4 – Train each segment's SplitterNode feature relevance weights using
               gradients so that signal_weights act like per-feature attention.
      Step 5 – Train the JudgeNode (segment relevance / cluster scoring) once
               every segment has finished its own weight + position training.
      Step 6 – Prune processing nodes whose activation_count is zero (never
               reached during training) to reduce inference cost.
    """

    WEIGHT_LR             = 0.01
    WEIGHT_LR_REDUCED     = 0.001   # smaller alpha for post-position weight refinement
    POSITION_LR           = 0.005
    SPLITTER_LR           = 0.01    # learning rate for splitter feature relevance
    MAX_POSITION_STEP     = 2.0     # max coordinate shift per update ("circle radius")
    CONVERGENCE_THRESHOLD = 1e-4    # stop early when combined loss delta < this
    JUDGE_ITERATIONS      = 3       # k-means iterations passed to JudgeNode.train

    def __init__(self, main):
        self.main = main

    # ------------------------------------------------------------------
    # Training-mode forward pass
    # ------------------------------------------------------------------

    def _forward_segment(self, segment, pre_processed):
        """
        Training forward pass for one segment.

        Mirrors infer()'s segment_infer exactly – same loop structure, same
        state-clearing pattern – but calls train_process_signal() instead of
        process_signal() so that each node records its path_contributions for
        later gradient computation.

        Returns (signals, segment_prediction).
        segment_prediction is None when no signals reached the reviewer.
        """
        splitter         = segment['splitter']
        reviewer         = segment['reviewer']
        processing_nodes = segment['processor']

        # Clear state from the previous sample (matches infer.segment_infer)
        for node in processing_nodes:
            node.clear_signals()
        reviewer.signals = []

        # Lazy weight initialisation on first contact
        for node in processing_nodes:
            if not node.weights:
                node.initialize_weights(pre_processed)

        # Generate signals and dispatch to the splitter's nearest neighbours
        signals = list(splitter.process(pre_processed, self.main.max_x))
        for node, signal in zip(splitter.connected_nodes, signals):
            node.receive_signal(signal)

        # Propagation loop – identical to infer except for the training call
        active         = True
        loop_iteration = 0
        max_iter       = 500        # same safety cap as infer

        while active and loop_iteration < max_iter:
            for node in processing_nodes:
                delta = node.train_process_signal()
                if delta is not None:
                    node.forward_signal()

            active = any(s.is_active() and not s.collected for s in signals)
            if not signals:
                active = False
            loop_iteration += 1

        segment_prediction = reviewer.review_signals()
        return signals, segment_prediction

    # ------------------------------------------------------------------
    # Loss & backprop
    # ------------------------------------------------------------------

    @staticmethod
    def _mse(prediction, target):
        if prediction is None:
            return None
        return (prediction - target) ** 2

    def _backprop(self, signals, segment_prediction, target, mode):
        """
        Backpropagate through all collected signal paths.

        The chain rule is applied through ReviewerNode.review_signals()'s
        inverse-variance weighted mean so the per-signal gradient is correct:

            dL/d(pred_i) = dL/d(final_pred) * w_i / sum(w_j)

        mode: 'weights' | 'positions'
        Returns mean per-signal MSE for the loss tracker, or None.
        """
        if segment_prediction is None:
            return None

        dL_dseg = 2.0 * (segment_prediction - target)

        # Mirror ReviewerNode.review_signals weighting
        collected = [s for s in signals if s.collected and s.is_active()]
        if not collected:
            return None

        inv_vars = [1.0 / max(s.variance, 1e-9) for s in collected]
        total_w  = sum(inv_vars)
        if total_w == 0:
            return None

        total_loss = 0.0
        for signal, w in zip(collected, inv_vars):
            dL_dpred_i = dL_dseg * (w / total_w)
            for contrib in signal.path_contributions.values():
                node = contrib['node']
                if mode == 'weights':
                    node.accumulate_weight_gradient(dL_dpred_i, signal)
                else:
                    # Position phase: generate a gradient "circle" around each
                    # path node (full LR); non-path connected neighbours receive
                    # a smaller effective step via MAX_POSITION_STEP clamping in
                    # ProcessingNode.apply_position_gradient.
                    node.accumulate_position_gradient(dL_dpred_i, signal)

            loss_i = self._mse(signal.prediction, target)
            if loss_i is not None:
                total_loss += loss_i

        return total_loss / len(collected)

    # ------------------------------------------------------------------
    # Gradient application
    # ------------------------------------------------------------------

    def _apply_and_reset(self, segment, mode, lr):
        """Apply accumulated gradients to every node in the segment, then reset."""
        for node in segment['processor']:
            if mode == 'weights':
                node.apply_weight_gradient(lr)
            else:
                node.apply_position_gradient(lr, self.MAX_POSITION_STEP)
            node.reset_gradients()

        if mode == 'positions':
            node_list = segment['processor'] + [segment['reviewer']]
            for node in segment['processor']:
                node.connected_nodes = []
                node.connect_nearest_nodes(
                    node_list=node_list,
                    connection_percentage=self.main.connection_percentage
                )

    # ------------------------------------------------------------------
    # Splitter feature-relevance backprop & epoch
    # ------------------------------------------------------------------

    def _backprop_splitter(self, signals, segment_prediction, target, splitter):
        """
        Backpropagate feature relevance gradients to the segment's SplitterNode.

        The same reviewer chain-rule weighting used in _backprop is applied so
        the per-signal gradient is consistent:
            dL/d(rel_j) += dL_dpred_i * sum_k[ value_j * w_j_k / (1 + dist_k) ]
        The accumulation across nodes k is handled inside
        SplitterNode.accumulate_feature_relevance_gradient.
        """
        if segment_prediction is None:
            return None

        dL_dseg = 2.0 * (segment_prediction - target)

        collected = [s for s in signals if s.collected and s.is_active()]
        if not collected:
            return None

        inv_vars = [1.0 / max(s.variance, 1e-9) for s in collected]
        total_w  = sum(inv_vars)
        if total_w == 0:
            return None

        total_loss = 0.0
        for signal, w in zip(collected, inv_vars):
            dL_dpred_i = dL_dseg * (w / total_w)
            splitter.accumulate_feature_relevance_gradient(dL_dpred_i, signal)
            loss_i = self._mse(signal.prediction, target)
            if loss_i is not None:
                total_loss += loss_i

        return total_loss / len(collected)

    def _epoch_splitter(self, sample_list, lr, desc):
        """
        One complete pass over sample_list training each splitter's feature
        relevance weights.  Runs a full training forward pass per segment so
        path_contributions are populated, then backprops only to the splitter.
        Returns average loss across all (segment, sample) pairs.
        """
        main       = self.main
        total_loss = 0.0
        count      = 0

        with main.Logger.make_progress(transient=True) as progress:
            task = progress.add_task(desc, total=len(sample_list))
            for sample in sample_list:
                target = sample.get(main.y_column)
                if target is None:
                    progress.update(task, advance=1)
                    continue

                for segment in main.segments:
                    splitter = segment['splitter']
                    splitter.reset_feature_relevance_gradients()

                    signals, seg_pred = self._forward_segment(segment, sample)
                    loss = self._backprop_splitter(signals, seg_pred, target, splitter)
                    splitter.apply_feature_relevance_gradient(lr)

                    if loss is not None:
                        total_loss += loss
                        count      += 1

                progress.update(task, advance=1)

        return total_loss / count if count > 0 else float('inf')

    # ------------------------------------------------------------------
    # Single-epoch pass
    # ------------------------------------------------------------------

    def _epoch(self, sample_list, mode, lr, desc):
        """
        One complete pass over sample_list in the given mode.
        Trains ALL segments per sample (Judge cluster→segment mapping is not
        yet meaningful until JudgeNode.train() has run; that happens in Step 4).
        Returns average loss across all (segment, sample) pairs.
        """
        main       = self.main
        total_loss = 0.0
        count      = 0

        with main.Logger.make_progress(transient=True) as progress:
            task = progress.add_task(desc, total=len(sample_list))
            for sample in sample_list:
                target = sample.get(main.y_column)
                if target is None:
                    progress.update(task, advance=1)
                    continue

                for segment in main.segments:
                    signals, seg_pred = self._forward_segment(segment, sample)
                    loss = self._backprop(signals, seg_pred, target, mode)
                    self._apply_and_reset(segment, mode, lr)

                    if loss is not None:
                        total_loss += loss
                        count      += 1

                progress.update(task, advance=1)

        return total_loss / count if count > 0 else float('inf')

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune(self):
        """Remove processing nodes that were never activated during training."""
        main = self.main
        for segment in main.segments:
            before    = len(segment['processor'])
            surviving = [n for n in segment['processor'] if n.activation_count > 0]
            pruned_set = set(segment['processor']) - set(surviving)

            segment['processor'] = surviving

            # Remove dangling references to pruned nodes from surviving nodes'
            # connected_nodes lists so they can be garbage-collected.
            if pruned_set:
                for node in surviving:
                    node.connected_nodes = [
                        n for n in node.connected_nodes if n not in pruned_set
                    ]

            pruned = before - len(surviving)
            if pruned > 0:
                main.display(
                    f"Segment {segment['id']}: pruned {pruned}/{before} unused nodes.",
                    classification=4
                )

    # ------------------------------------------------------------------
    # Main entry point  (called from Main.train)
    # ------------------------------------------------------------------

    def train(self, unprocessed_dataset, epoch_count):
        main = self.main

        # Judge must be in demo mode during training so that find_relevant_segments
        # falls back gracefully before cluster->segment mapping is established.
        original_judge_mode = main.Judge.mode
        main.Judge.mode = "demo"

        # ── Step 1: Pre-process the raw dataset ───────────────────────
        main.display("Training Step 1: Processing dataset.", classification=4)
        # Pass a copy so drop_removable_columns does not mutate the caller's df
        processed_df = main.PreProcessor.process_dataset(unprocessed_dataset.copy())
        sample_list  = processed_df.to_dict(orient='records')
        main.display(f"  {len(sample_list)} samples ready.", classification=4)

        # ── Step 2: Initialise processing-node and splitter weights ──────
        main.display("Training Step 2: Initialising node weights.", classification=4)
        if sample_list:
            init_sample = sample_list[0]
            for segment in main.segments:
                for node in segment['processor']:
                    if not node.weights:
                        node.initialize_weights(init_sample)
                splitter = segment['splitter']
                if not splitter.signal_weights:
                    splitter.initialize_signal_weights(init_sample)
        main.display("  Node and splitter weights initialised.", classification=4)

        # ── Step 3: Alternating weight / position training ────────────
        main.display(
            f"Training Step 3: Alternating weight/position training "
            f"for up to {epoch_count} epoch(s).",
            classification=4
        )
        prev_loss = float('inf')

        with main.Logger.make_progress() as progress:
            epoch_task = progress.add_task("Training Epochs", total=epoch_count)
            for epoch in range(epoch_count):
                tag = f"Epoch {epoch + 1}/{epoch_count}"

                # --- A. Weight phase: end-to-end gradient descent --------
                w_loss = self._epoch(
                    sample_list,
                    mode='weights',
                    lr=self.WEIGHT_LR,
                    desc=f"{tag} | Weight Phase"
                )
                main.display(f"{tag} | Weight phase loss:      {w_loss:.6f}", classification=4)

                # --- B. Position phase: gradient-based node repositioning -
                p_loss = self._epoch(
                    sample_list,
                    mode='positions',
                    lr=self.POSITION_LR,
                    desc=f"{tag} | Position Phase"
                )
                main.display(f"{tag} | Position phase loss:    {p_loss:.6f}", classification=4)

                # --- A'. Weight refinement with smaller LR ----------------
                # After positions shift, connections may change; retrain weights
                # at a reduced alpha to adapt without over-correcting.
                w2_loss = self._epoch(
                    sample_list,
                    mode='weights',
                    lr=self.WEIGHT_LR_REDUCED,
                    desc=f"{tag} | Weight Refinement"
                )
                main.display(f"{tag} | Weight refinement loss: {w2_loss:.6f}", classification=4)

                # --- C. Splitter feature-relevance phase ------------------
                # Train each splitter's signal_weights as per-feature attention
                # after processing nodes have an updated set of weights/positions.
                s_loss = self._epoch_splitter(
                    sample_list,
                    lr=self.SPLITTER_LR,
                    desc=f"{tag} | Splitter Phase"
                )
                main.display(f"{tag} | Splitter phase loss:    {s_loss:.6f}", classification=4)

                combined = (w_loss + p_loss + w2_loss + s_loss) / 4.0
                delta    = abs(prev_loss - combined)
                main.display(
                    f"{tag} | Combined loss: {combined:.6f}  delta: {delta:.2e}",
                    classification=4
                )

                # --- Evaluate via Main.infer (as directed by the plan) ----
                eval_sample = unprocessed_dataset.iloc[0].to_dict()
                eval_pred   = main.infer(eval_sample, loud=False)
                actual      = eval_sample.get(main.y_column)
                if eval_pred is not None and actual is not None and actual != 0:
                    pct_err = abs(eval_pred - actual) / abs(actual) * 100.0
                    main.display(
                        f"{tag} | Infer check: pred={eval_pred:.4f}  "
                        f"actual={actual:.4f}  err={pct_err:.2f}%",
                        classification=4
                    )

                progress.update(epoch_task, advance=1)

                # Convergence check – stop early if loss has plateaued
                if delta < self.CONVERGENCE_THRESHOLD:
                    main.display(
                        f"Converged at epoch {epoch + 1} (delta={delta:.2e}). "
                        f"Stopping early.",
                        classification=4
                    )
                    break

                prev_loss = combined

        # ── Step 5: Train JudgeNode for segment relevance ─────────────
        # The plan specifies training segment relevance only after each
        # segment's processing nodes and splitters have defined training.
        main.display(
            "Training Step 5: Training JudgeNode for segment relevance.",
            classification=4
        )
        main.Judge.train(processed_df, iterations=self.JUDGE_ITERATIONS)

        # ── Step 6: Prune unused processing nodes ─────────────────────
        main.display("Training Step 6: Pruning unused processing nodes.", classification=4)
        self._prune()

        # Restore Judge mode
        main.Judge.mode = original_judge_mode

        main.display("Training complete.", classification=4)
