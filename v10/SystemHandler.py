# Code required for the nexus system handler
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Components"))

from SegmentHandler import SegmentHandler
from Components.JudgeNode import JudgeNode
from Components.Logger import Logger
from Components.PreProcessingNode import PreProcesingNode
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
        self.HandlerNode = HandlerNode(logger=self.logger, classification=self.classification)
        self.preprocessor = PreProcesingNode(Logger=self.logger, logger_classification=4)


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
            # Derive axis signs from segment_id bits: bit d=0 → +1, bit d=1 → -1
            loc = [1 - 2 * ((segment.segment_id >> d) & 1) for d in range(self.dimensions)]
            segment.initializeSegment(loc=loc)

        self.display(f"Initialized {segmentCount} segments", Loud=Loud)

    def train(self, dataset, epoch_count: int = 3, judge_iterations: int = 5, loud: bool = True) -> None:
        from collections import defaultdict
        if not self.segments:
            raise ValueError("Segments must be initialized before training. Call initializeAllSegments() first.")

        # Step 1: Cluster the full dataset and assign clusters to segments
        self.display("Training JudgeNode — clustering full dataset...", Loud=loud)
        preprocessed = self.preprocessor.process_dataset(dataset.copy())
        self.JudgeNode.train(preprocessed, judge_iterations, segments=self.segments)

        # Step 2: Map each cluster's points back to original dataset row indices
        preprocessed_vectors = [list(r.values()) for r in preprocessed.to_dict(orient='records')]
        vector_to_indices: dict[tuple, list[int]] = defaultdict(list)
        for i, vec in enumerate(preprocessed_vectors):
            vector_to_indices[tuple(vec)].append(i)

        segment_indices: dict[int, set[int]] = defaultdict(set)
        for cluster in self.JudgeNode.segment_weights['clusters']:
            sid = cluster.get('segment_id')
            if sid is None:
                continue
            for point in cluster['points']:
                for idx in vector_to_indices.get(tuple(point), []):
                    segment_indices[sid].add(idx)

        # Step 3: Train each segment only on its assigned rows
        for segment in self.segments:
            indices = sorted(segment_indices.get(segment.segment_id, []))
            if not indices:
                self.display(f"Segment {segment.segment_id} has no assigned data — skipping.", Loud=loud)
                continue
            subset = dataset.iloc[indices].reset_index(drop=True)
            self.display(f"Training segment {segment.segment_id} on {len(subset)}/{len(dataset)} rows...", Loud=loud)
            segment.train(subset, epoch_count=epoch_count)

    def runInfer(self, input, loud = True):
        if self.JudgeNode is None or self.HandlerNode is None:
            raise ValueError("JudgeNode or HandlerNode not assigned")

        self.display("Running inference on JudgeNode", Loud=loud)
        pre_input = self.preprocessor.process_data(input)

        # If JudgeNode has not been trained, activate all segments equally
        if not self.JudgeNode.segment_weights['segment']:
            self.display("JudgeNode not trained — using all segments at equal relevance.", Loud=loud)
            selected_segments = [(s.segment_id, 1.0) for s in self.segments]
        else:
            relevance_scores  = self.JudgeNode.calculate_input_segment_relevance(pre_input, Loud=loud)
            selected_segments = self.JudgeNode.find_relevant_segments(relevance_scores, Loud=loud)

        self.display(f"Selected segments: {[sid for sid, _ in selected_segments]}", Loud=loud)

        segment_map = {s.segment_id: s for s in self.segments}
        for segment_id, relevance in selected_segments:
            segment = segment_map[segment_id]
            reports = segment.segmentInfer(pre_input, loud=loud)
            for report in reports:
                self.HandlerNode.receive_report(segment_id, relevance, report['prediction'])

        return self.HandlerNode.process_reports(loud)


if __name__ == "__main__":
    import sys
    import time
    import math as _math
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, ".")
    import Components.RichConsole as RichConsole
    from Components.PreProcessingNode import PreProcesingNode
    from rich.table import Table
    from rich.rule import Rule
    from rich.panel import Panel
    from rich import box

    DATASET_CSV    = "Exam_Score_Prediction.csv"
    TARGET_COL     = "exam_score"
    EPOCH_COUNT    = 5
    JUDGE_ITERS    = 5
    MAX_X          = 10
    DIMENSIONS     = 2
    CONN_PCT       = 0.1
    DENSITY        = 0.8

    # Per-segment colours for the combined graph
    SEG_COLORS = ["steelblue", "tomato", "mediumseagreen", "darkorchid",
                  "darkorange", "hotpink", "teal", "saddlebrown"]

    # ── Helper: save combined nexus graph ─────────────────────────────────
    def save_nexus_graph(system, save_path, title):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axhline(0, color='black', linewidth=0.6, alpha=0.4)
        ax.axvline(0, color='black', linewidth=0.6, alpha=0.4)

        for seg in system.segments:
            sc   = SEG_COLORS[seg.segment_id % len(SEG_COLORS)]
            comp = seg.segmentComponents
            if comp is None:
                continue
            splitter   = comp['splitter']
            reviewers  = comp['reviewer']
            processors = comp['processing_nodes']

            # Processing nodes + connections
            for node in processors:
                ax.scatter(node.position[0], node.position[1],
                           color=sc, s=12, alpha=0.6, zorder=2)
                for tgt in node.connected_nodes:
                    ax.plot([node.position[0], tgt.position[0]],
                            [node.position[1], tgt.position[1]],
                            color=sc, linewidth=0.4, alpha=0.25, zorder=1)

            # Splitter connections
            for tgt in splitter.connected_nodes:
                ax.plot([splitter.position[0], tgt.position[0]],
                        [splitter.position[1], tgt.position[1]],
                        color=sc, linewidth=0.5, alpha=0.35, zorder=1, linestyle='--')

            # Splitter marker
            ax.scatter(splitter.position[0], splitter.position[1],
                       color=sc, s=120, marker='D', zorder=4,
                       label=f"Seg {seg.segment_id} splitter")

            # Reviewer markers
            for rev in reviewers:
                ax.scatter(rev.position[0], rev.position[1],
                           color=sc, s=150, marker='*', zorder=4,
                           label=f"Seg {seg.segment_id} reviewer {rev.position}")

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    # ── Helper: error % ───────────────────────────────────────────────────
    def _err_pct(pred, actual):
        if pred is None or actual is None or actual == 0:
            return None
        return abs(pred - actual) / abs(actual) * 100.0

    def _fmt(val, decimals=4):
        return f"{val:.{decimals}f}" if val is not None else "—"

    def _pct(val):
        return f"{val:.2f}%" if val is not None else "—"

    # ─────────────────────────────────────────────────────────────────────
    logger = RichConsole.RichLogger(f"system_demo_{int(time.time())}.log", log_level=4)
    logger.console.print(Rule("[bold cyan]System Handler Demo[/bold cyan]"))

    # ── Load dataset ──────────────────────────────────────────────────────
    dataset    = pd.read_csv(DATASET_CSV)
    sample_raw = dataset.iloc[0].to_dict()
    actual     = sample_raw.get(TARGET_COL)
    logger.log(f"Dataset loaded: {len(dataset)} rows. Evaluation sample actual {TARGET_COL}: {actual}", 4, True)

    # ── Warm up shared preprocessor vocabulary ────────────────────────────
    _warmup_pre = PreProcesingNode(Logger=logger, logger_classification=4)
    _warmup_pre.process_dataset(dataset.copy())

    # ── Build & initialise system ─────────────────────────────────────────
    logger.console.print(Rule("[bold cyan]Initialising System[/bold cyan]"))
    system = SystemHandler(
        maxX=MAX_X, target=TARGET_COL, logger=logger,
        connection_percentage=CONN_PCT, density=DENSITY,
        dimensions=DIMENSIONS, classification=4
    )
    system.initializeAllSegments(Loud=True)
    logger.log(f"Segments initialised: {[s.segment_id for s in system.segments]}", 4, True)

    # ── Pre-train graph ───────────────────────────────────────────────────
    logger.console.print(Rule("[bold yellow]Pre-Training Graph[/bold yellow]"))
    save_nexus_graph(system, "nexus_pretrain.png", "Nexus — Pre-Training Structure")
    logger.log("Pre-training graph saved → nexus_pretrain.png", 4, True)

    # ── Pre-training inference (all segments equally weighted) ────────────
    logger.console.print(Rule("[bold yellow]Pre-Training Inference[/bold yellow]"))
    pre_prediction = system.runInfer(sample_raw.copy(), loud=False)
    logger.log(f"Pre-training prediction: {pre_prediction}", 4, True)

    # ── Train ─────────────────────────────────────────────────────────────
    logger.console.print(Rule("[bold green]Training[/bold green]"))
    system.train(dataset, epoch_count=EPOCH_COUNT, judge_iterations=JUDGE_ITERS, loud=True)

    # ── Post-train graph ──────────────────────────────────────────────────
    logger.console.print(Rule("[bold yellow]Post-Training Graph[/bold yellow]"))
    save_nexus_graph(system, "nexus_posttrain.png", "Nexus — Post-Training Structure")
    logger.log("Post-training graph saved → nexus_posttrain.png", 4, True)

    # ── Post-training inference ───────────────────────────────────────────
    logger.console.print(Rule("[bold yellow]Post-Training Inference[/bold yellow]"))
    post_prediction = system.runInfer(sample_raw.copy(), loud=True)

    pre_err  = _err_pct(pre_prediction, actual)
    post_err = _err_pct(post_prediction, actual)
    improvement = (pre_err - post_err) if (pre_err is not None and post_err is not None) else None

    # ═════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═════════════════════════════════════════════════════════════════════
    logger.console.print(Rule("[bold white on blue] FINAL REPORT [/bold white on blue]"))

    # ── 1. System configuration ───────────────────────────────────────────
    cfg_table = Table(title="System Configuration", box=box.ROUNDED,
                      border_style="cyan", show_lines=True)
    cfg_table.add_column("Parameter", style="bold white")
    cfg_table.add_column("Value",     style="bright_cyan", justify="right")
    cfg_table.add_row("Target Column",        TARGET_COL)
    cfg_table.add_row("Dataset Rows",         str(len(dataset)))
    cfg_table.add_row("Dimensions",           str(DIMENSIONS))
    cfg_table.add_row("Max X",                str(MAX_X))
    cfg_table.add_row("Connection %",         str(CONN_PCT))
    cfg_table.add_row("Density",              str(DENSITY))
    cfg_table.add_row("Segment Count",        str(len(system.segments)))
    cfg_table.add_row("Epoch Count",          str(EPOCH_COUNT))
    cfg_table.add_row("Judge Iterations",     str(JUDGE_ITERS))
    cfg_table.add_row("Cluster Count (final)",str(str(len(system.JudgeNode.segment_weights['clusters']))))
    logger.console.print(cfg_table)

    # ── 2. Segment structure ──────────────────────────────────────────────
    seg_table = Table(title="Segment Structure", box=box.ROUNDED,
                      border_style="steelblue", show_lines=True)
    seg_table.add_column("Seg ID",        style="bold white",   justify="right")
    seg_table.add_column("Loc (signs)",   style="bright_cyan")
    seg_table.add_column("Train Rows",    style="bright_green", justify="right")
    seg_table.add_column("Splitter Pos",  style="yellow")
    seg_table.add_column("Reviewer Positions", style="white")
    seg_table.add_column("Nodes",         style="green",        justify="right")
    seg_table.add_column("Connections",   style="blue",         justify="right")
    seg_table.add_column("Mean Weight",   style="magenta",      justify="right")

    # Count training rows per segment from cluster assignments
    seg_train_rows: dict[int, int] = {}
    for c in system.JudgeNode.segment_weights['clusters']:
        sid = c.get('segment_id')
        if sid is not None:
            seg_train_rows[sid] = seg_train_rows.get(sid, 0) + len(c['points'])

    for seg in system.segments:
        loc = [1 - 2 * ((seg.segment_id >> d) & 1) for d in range(system.dimensions)]
        comp = seg.segmentComponents
        splitter   = comp['splitter']
        reviewers  = comp['reviewer']
        processors = comp['processing_nodes']

        total_connections = sum(len(n.connected_nodes) for n in processors)
        all_weights = [w for n in processors for w in n.weights.values()]
        mean_w = sum(all_weights) / len(all_weights) if all_weights else 0.0
        train_rows = seg_train_rows.get(seg.segment_id, 0)

        rev_str = "  ".join(str(r.position) for r in reviewers)
        seg_table.add_row(
            str(seg.segment_id),
            str(loc),
            str(train_rows),
            str(splitter.position),
            rev_str,
            str(len(processors)),
            str(total_connections),
            f"{mean_w:.4f}",
        )
    logger.console.print(seg_table)

    # ── 3. JudgeNode cluster report ───────────────────────────────────────
    clusters = system.JudgeNode.segment_weights['clusters']
    cl_table = Table(
        title=f"JudgeNode Clusters ({len(clusters)} clusters → {len(system.segments)} segments)",
        box=box.ROUNDED, border_style="magenta", show_lines=True
    )
    cl_table.add_column("Cluster",   style="bold white",    justify="right")
    cl_table.add_column("Segment",   style="bright_cyan",   justify="right")
    cl_table.add_column("Size",      style="white",         justify="right")
    cl_table.add_column("Geo Uniq",  style="yellow",        justify="right")
    cl_table.add_column("Behav Div", style="green",         justify="right")
    cl_table.add_column("Info Gain", style="blue",          justify="right")
    cl_table.add_column("Score",     style="bright_white",  justify="right")
    for i, c in enumerate(clusters):
        m = c.get('metrics', {})
        cl_table.add_row(
            str(i),
            str(c.get('segment_id', '—')),
            str(len(c['points'])),
            f"{m.get('GeometricUniqueness', 0):.4f}",
            f"{m.get('BehavioralDiversity', 0):.4f}",
            f"{m.get('InformationGain', 0):.4f}",
            f"{m.get('ClusterScore', 0):.4f}",
        )
    # Cluster averages footer
    if clusters:
        avg_geo  = sum(c.get('metrics', {}).get('GeometricUniqueness', 0) for c in clusters) / len(clusters)
        avg_beh  = sum(c.get('metrics', {}).get('BehavioralDiversity', 0) for c in clusters) / len(clusters)
        avg_info = sum(c.get('metrics', {}).get('InformationGain', 0) for c in clusters) / len(clusters)
        avg_sc   = sum(c.get('metrics', {}).get('ClusterScore', 0) for c in clusters) / len(clusters)
        cl_table.add_section()
        cl_table.add_row("[bold]Average[/bold]", "—", "—",
                         f"[bold]{avg_geo:.4f}[/bold]", f"[bold]{avg_beh:.4f}[/bold]",
                         f"[bold]{avg_info:.4f}[/bold]", f"[bold]{avg_sc:.4f}[/bold]")
    logger.console.print(cl_table)

    # ── 4. Inference results ──────────────────────────────────────────────
    infer_table = Table(title="Inference Results", box=box.ROUNDED,
                        border_style="yellow", show_lines=True)
    infer_table.add_column("Stage",      style="bold white")
    infer_table.add_column("Prediction", style="yellow",      justify="right")
    infer_table.add_column("Actual",     style="bold white",  justify="right")
    infer_table.add_column("Abs Error",  style="red",         justify="right")
    infer_table.add_column("Error %",    style="red",         justify="right")

    abs_pre  = abs(pre_prediction - actual)  if pre_prediction  is not None else None
    abs_post = abs(post_prediction - actual) if post_prediction is not None else None

    infer_table.add_row("Pre-Training",  _fmt(pre_prediction),  _fmt(actual), _fmt(abs_pre),  _pct(pre_err))
    infer_table.add_row("Post-Training", _fmt(post_prediction), _fmt(actual), _fmt(abs_post), _pct(post_err))
    if improvement is not None:
        sign = "[green]▼[/green]" if improvement > 0 else "[red]▲[/red]"
        infer_table.add_section()
        infer_table.add_row(f"Improvement", "—", "—", "—", f"{sign} {_pct(abs(improvement))}")
    logger.console.print(infer_table)

    # ── 5. Saved artefacts ────────────────────────────────────────────────
    art_table = Table(title="Saved Artefacts", box=box.SIMPLE,
                      border_style="dim", show_lines=False)
    art_table.add_column("File", style="bold white")
    art_table.add_column("Description", style="dim white")
    art_table.add_row("nexus_pretrain.png",  "Combined 4-segment graph before training")
    art_table.add_row("nexus_posttrain.png", "Combined 4-segment graph after training")
    for seg in system.segments:
        art_table.add_row(f"segment_{seg.segment_id}.nexseg",
                          f"Trained weights for segment {seg.segment_id}")
    logger.console.print(art_table)

    logger.console.print(Rule("[bold green]Demo complete[/bold green]"))
