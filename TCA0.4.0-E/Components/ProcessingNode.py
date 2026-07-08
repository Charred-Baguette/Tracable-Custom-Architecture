import math
import random
random.seed(42)

class ProcessingNode:
    # Stability constants
    DELTA_CLIP = 10.0      # Max absolute value for any single node's delta
    PRED_CLIP = 1e6        # Max absolute prediction value (generous, prevents inf)
    GRAD_CLIP = 1.0        # Max absolute value per accumulated gradient element
    WEIGHT_CLIP = 5.0      # Max absolute weight value after update

    # Connectivity floor: every node is guaranteed at least this many outward
    # connections after connect_nearest_nodes(), regardless of connection_percentage.
    # Raises the floor so signals are less likely to reach a dead end mid-graph.
    MIN_CONNECTIONS = 3

    # Displacement penalty: soft L2 regulariser that penalises moving far from
    # the node's original position.  Gradient = POSITION_PENALTY * displacement,
    # so the further a node drifts, the harder it is pulled back.  Tune this to
    # balance topology freedom vs. collapse prevention.
    POSITION_PENALTY = .1

    def __init__(self, position, Logger=None, classification=4):
        self.position          = position
        self.original_position = list(position)   # anchor for displacement penalty
        self.Logger = Logger
        self.classification = classification
        self.signal = None
        self.signal_queue = []
        self.connected_nodes = []
        self.weights = {}
        self.distance_to_origin = sum(p ** 2 for p in position) ** 0.5
        self.activation_count = 0
        self.weight_gradients = {}
        self.position_gradient = [0.0] * len(position)

    def __repr__(self) -> str:
        return f"ProcessingNode(pos={self.position})"
    
    def display(self, message, classification = None, Loud = True):
        message = f"[ProcessingNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        if classification is None:
            classification = self.classification
        self.Logger.log(message, classification, Loud)
    
    def initialize_weights(self, input_data):
        # Random init breaks weight symmetry so nodes follow different gradient
        # trajectories from the first update and genuinely specialise.
        # Mean = 1/n (same as before), std = 0.5/n, clamped non-negative.
        n = max(len(input_data), 1)
        mu = 1.0 / n
        sigma = 0.5 / n
        for feature in input_data:
            self.weights[feature] = max(0.0, random.gauss(mu, sigma))
        self.weights['input_prediction'] = max(0.0, random.gauss(mu, sigma))

    def receive_signal(self, signal):
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
                self.signal_queue.append(signal)
            else:
                self.signal = signal
                
        else:
            self.signal_queue.append(signal)
        
        # Only the newly arriving signal pays 1 life.  Signals already held in
        # self.signal or waiting in self.signal_queue have already paid on arrival.
        signal.signal_life -= 1
        signal.position = self.position
        return True
    
    def connect_nearest_nodes(self, node_list, connection_percentage):
        candidates = [
            n for n in node_list
            if n is not self and
            getattr(n, "distance_to_origin", None) > self.distance_to_origin
        ]

        if not candidates:
            return []

        # Compute distances and sort nearest first.
        distances = [(n, math.dist(self.position, n.position)) for n in candidates]
        distances.sort(key=lambda x: x[1])

        # Honour connection_percentage but never go below MIN_CONNECTIONS,
        # capped at the number of available candidates.
        target_count = max(self.MIN_CONNECTIONS, int(math.ceil(connection_percentage * 50)))
        target_count = min(target_count, len(candidates))
        selected = [node for node, _ in distances[:target_count]]

        for node in selected:
            if node not in self.connected_nodes:
                self.connected_nodes.append(node)

        return self.connected_nodes
    
    def forward_signal(self):
        if self.signal is None:
            self.display("No signal to forward. Checking queue", 1, Loud=False)

        if self.signal is None:
            return False
        self.signal.visited_nodes.append(self)

        # Track recent 3 for routing exclusion
        self.signal.recent_visited.append(self)
        if len(self.signal.recent_visited) > 3:
            self.signal.recent_visited.pop(0)

        viable_nodes = []
        for node in self.connected_nodes:
            if node in self.signal.recent_visited:
                continue
            viable_nodes.append(node)

        if not viable_nodes:
            # Dead end: clear recent_visited but keep this node on it so signal
            # can backtrack without immediately returning here.
            self.display("No viable connected nodes to forward the signal.", 1, Loud=False)
            self.signal.recent_visited = [self]
            # Allow all connected nodes except this one as candidates
            viable_nodes = [n for n in self.connected_nodes if n is not self]
            if not viable_nodes:
                self.signal.signal_life = 0
                self.signal = None
                return False

        # Outward-alignment bias: reward movement away from origin.
        # For each candidate, compute how well the movement direction aligns
        # with the away-from-origin direction at this node.
        origin_dist = self.distance_to_origin + 1e-9
        self_norm   = [p / origin_dist for p in self.position]   # unit vec pointing outward

        REVIEWER_BONUS = 3.0   # reviewers are preferred terminal targets

        def _weight(node):
            if hasattr(node, 'review_signals'):
                return node.distance_to_origin * REVIEWER_BONUS

            # Geometric component: prefer nodes farther from origin and
            # in the outward direction from the current node.
            move = [b - a for a, b in zip(self.position, node.position)]
            move_len = math.sqrt(sum(v * v for v in move)) + 1e-9
            alignment = sum(s * m / move_len for s, m in zip(self_norm, move))
            geometric = node.distance_to_origin * (1.0 + max(0.0, alignment))

            # Correction-seeking component: prefer nodes whose weights
            # would produce evidence far from the signal's current prediction.
            # A large gap means that node can still meaningfully update the signal.
            if self.signal is None:
                return geometric
            node_evidence = sum(
                self.signal.input.get(f, 0) * w * self.signal.feature_relevance.get(f, 1.0)
                for f, w in node.weights.items()
            )
            expected_correction = abs(node_evidence - self.signal.prediction) / (
                1.0 + node.distance_to_origin
            )

            return geometric * max(1e-9, expected_correction)

        weights = [max(1e-9, _weight(n)) for n in viable_nodes]
        selected_node = random.choices(viable_nodes, weights=weights, k=1)[0]
        selected_node.receive_signal(self.signal)
        #self.display(f"Forwarded signal to node at position {selected_node.position}.", 4)
        self.signal = None
            
        return True

    def process_signal(self):
        """Inference forward pass — same math as train_process_signal but without gradient recording."""
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
            else:
                return None

        # 1. Feature evidence
        evidence = sum(
            value * self.weights.get(feature, 1.0) * self.signal.feature_relevance.get(feature, 1.0)
            for feature, value in self.signal.input.items()
        )

        # 2. Context: running prior weighted by input_prediction weight
        pred_w   = self.weights.get('input_prediction', 1.0)
        context  = self.signal.prediction * pred_w

        # 3. Distance-shaped blend: near-origin nodes are context-driven (broad
        #    corrections); far nodes are evidence-driven (fine adjustments).
        distance    = self.distance_to_origin + 1e-6
        precision   = distance / (1.0 + distance)    # → 1 near reviewers
        uncertainty = 1.0 - precision                # → 1 near origin
        raw         = uncertainty * context + precision * evidence

        # tanh naturally bounds delta; DELTA_CLIP kept as hard safety only
        scaled_delta = math.tanh(raw) / (1.0 + distance)
        scaled_delta = max(-self.DELTA_CLIP, min(self.DELTA_CLIP, scaled_delta))

        # 4. Update prediction
        self.signal.prediction += scaled_delta
        self.signal.prediction = max(-self.PRED_CLIP, min(self.PRED_CLIP, self.signal.prediction))

        if hasattr(self.signal, "variance"):
            self.signal.variance += abs(scaled_delta)

        return scaled_delta

    def train_process_signal(self):
        """Forward pass for training - computes delta correctly and records contribution for gradients"""
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
            else:
                return None

        self.activation_count += 1

        # 1. Feature evidence
        feature_details = {}
        evidence = 0.0
        for feature, value in self.signal.input.items():
            w   = self.weights.get(feature, 1.0)
            rel = self.signal.feature_relevance.get(feature, 1.0)
            evidence += value * w * rel
            feature_details[feature] = {'value': value, 'weight': w, 'relevance': rel}

        # 2. Context: running prior weighted by input_prediction weight.
        #    Record prev_prediction before this node updates the signal.
        pred_w         = self.weights.get('input_prediction', 1.0)
        prev_prediction = self.signal.prediction
        context        = prev_prediction * pred_w

        # 3. Distance-shaped blend
        distance    = self.distance_to_origin + 1e-6
        precision   = distance / (1.0 + distance)
        uncertainty = 1.0 - precision
        raw         = uncertainty * context + precision * evidence

        tanh_raw    = math.tanh(raw)
        tanh_deriv  = 1.0 - tanh_raw ** 2   # d(tanh)/d(raw)
        scaled_delta = tanh_raw / (1.0 + distance)
        scaled_delta = max(-self.DELTA_CLIP, min(self.DELTA_CLIP, scaled_delta))

        # 4. Update prediction
        self.signal.prediction += scaled_delta
        self.signal.prediction = max(-self.PRED_CLIP, min(self.PRED_CLIP, self.signal.prediction))

        if hasattr(self.signal, 'variance'):
            self.signal.variance += abs(scaled_delta)

        # 5. Record contribution for gradient computation
        self.signal.path_contributions[id(self)] = {
            'node':            self,
            'scaled_delta':    scaled_delta,
            'evidence':        evidence,
            'context':         context,
            'prev_prediction': prev_prediction,
            'precision':       precision,
            'uncertainty':     uncertainty,
            'tanh_raw':        tanh_raw,
            'tanh_deriv':      tanh_deriv,
            'distance':        distance,
            'feature_details': feature_details,
        }

        return scaled_delta

    def accumulate_weight_gradient(self, dL_dpred, signal):
        """Accumulate weight gradients from one signal path.

        d(scaled_delta)/d(w_feature)  = tanh_deriv * precision * value * relevance / (1+d)
        d(scaled_delta)/d(pred_w)     = tanh_deriv * uncertainty * prev_prediction / (1+d)
        """
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        distance    = contrib['distance']
        tanh_deriv  = contrib['tanh_deriv']
        precision   = contrib['precision']
        uncertainty = contrib['uncertainty']
        prev_pred   = contrib['prev_prediction']
        scale       = tanh_deriv / (1.0 + distance)

        for feature, fd in contrib['feature_details'].items():
            dL_dw = dL_dpred * scale * precision * fd['value'] * fd['relevance']
            dL_dw = max(-self.GRAD_CLIP, min(self.GRAD_CLIP, dL_dw))
            self.weight_gradients[feature] = self.weight_gradients.get(feature, 0.0) + dL_dw

        # Gradient for input_prediction weight
        dL_dpred_w = dL_dpred * scale * uncertainty * prev_pred
        dL_dpred_w = max(-self.GRAD_CLIP, min(self.GRAD_CLIP, dL_dpred_w))
        self.weight_gradients['input_prediction'] = (
            self.weight_gradients.get('input_prediction', 0.0) + dL_dpred_w
        )


    def accumulate_position_gradient(self, dL_dpred, signal):
        """Accumulate position gradients from one signal path.

        scaled_delta = tanh(raw) / (1+d)
        Full derivative w.r.t. position_j (derived via chain rule on d = ||pos||):
            d(raw)/d(d)   = (evidence - context) / (1+d)²
            d(d)/d(pos_j) = pos_j / d
            d(1/(1+d))/d(pos_j) = -pos_j / (d*(1+d)²)

        grad_j = dL_dpred * pos_j/d *
                 [tanh_deriv*(evidence-context)/(1+d)³ - tanh_raw/(1+d)²]
        """
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        distance  = contrib['distance']
        tanh_raw  = contrib['tanh_raw']
        tanh_deriv = contrib['tanh_deriv']
        evidence  = contrib['evidence']
        context   = contrib['context']

        for j, p in enumerate(self.position):
            if distance < 1e-9:
                continue
            d1 = 1.0 + distance
            grad_j = dL_dpred * (p / distance) * (
                tanh_deriv * (evidence - context) / (d1 ** 3)
                - tanh_raw / (d1 ** 2)
            )
            grad_j = max(-self.GRAD_CLIP, min(self.GRAD_CLIP, grad_j))
            self.position_gradient[j] += grad_j

    def apply_weight_gradient(self, learning_rate):
        """Apply accumulated weight gradients"""
        for feature in self.weight_gradients:
            current = self.weights.get(feature, 1.0)
            updated = current - learning_rate * self.weight_gradients[feature]
            self.weights[feature] = max(-self.WEIGHT_CLIP, min(self.WEIGHT_CLIP, updated))  # Clamp weights
        self.weight_gradients = {}

    def apply_position_gradient(self, learning_rate, max_step, max_x=None):
        """Apply position gradient, clamp step, then enforce quarter-circle bounds.

        Before stepping, the displacement penalty gradient is injected:
            grad_j += POSITION_PENALTY * (pos_j - original_pos_j)
        This is the gradient of λ||pos - pos_original||², pulling the node back
        toward its starting position proportionally to how far it has drifted.
        """
        # Inject displacement penalty (grows with drift, no hard limit)
        for j in range(len(self.position)):
            displacement = float(self.position[j]) - self.original_position[j]
            self.position_gradient[j] += self.POSITION_PENALTY * displacement

        new_position = list(self.position)
        for j in range(len(self.position)):
            step = -learning_rate * self.position_gradient[j]
            step = max(-max_step, min(max_step, step))
            new_position[j] = self.position[j] + step

        if max_x is not None:
            # Stay in the first quadrant (non-negative coordinates)
            new_position = [max(0.0, min(float(max_x), c)) for c in new_position]
            # Project back onto arc if the move pushed the node outside the quarter-circle
            dist = math.sqrt(sum(c ** 2 for c in new_position))
            if dist > max_x:
                scale = max_x / dist
                new_position = [c * scale for c in new_position]

        self.position = tuple(new_position) if isinstance(self.position, tuple) else new_position
        self.distance_to_origin = sum(p ** 2 for p in self.position) ** 0.5
        self.position_gradient = [0.0] * len(self.position)

    def reset_gradients(self):
        """Reset all gradient accumulators"""
        self.weight_gradients = {}
        self.position_gradient = [0.0] * len(self.position)

    def clear_signals(self):
        """Clear signal state between training samples"""
        self.signal = None
        self.signal_queue = []

