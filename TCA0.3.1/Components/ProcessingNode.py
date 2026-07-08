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

    def __init__(self, position, segment_id, Logger=None, classification=4):
        self.position = position
        self.Logger = Logger
        self.classification = classification
        self.signal = None
        self.signal_queue = []
        self.connected_nodes = []
        self.weights = {}
        self.segment_id = segment_id
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
        # Scale initial weight by 1/num_features so weighted_sum starts at a
        # reasonable magnitude regardless of how many features exist.
        n = max(len(input_data), 1)
        init_w = 1.0 / n
        for feature in input_data:
            self.weights[feature] = init_w

        # input_prediction weight kept small to dampen the feedback loop
        # (prediction gets multiplied by this and re-added every node)
        self.weights['input_prediction'] = init_w

    def receive_signal(self, signal):
        if self.signal is None:
            if self.signal_queue:
                self.signal = self.signal_queue.pop(0)
                self.signal_queue.append(signal)
            else:
                self.signal = signal
                
        else:
            self.signal_queue.append(signal)
        
        if self.signal is not None:
            self.signal.position = self.position
            self.signal.signal_life -= 1

        for signal in self.signal_queue:
            signal.position = self.position
            signal.signal_life -= 1
        return True
    
    def connect_nearest_nodes(self, node_list, connection_percentage):
        candidates = [
            n for n in node_list
            if n is not self and 
            getattr(n, "segment_id", None) == self.segment_id and 
            getattr(n, "distance_to_origin", None) > self.distance_to_origin
        ]

        if not candidates:
            return []

        # Compute distances
        distances = [
            (n, math.dist(self.position, n.position))
            for n in candidates
        ]

        # Sort by distance (nearest first)
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
        viable_nodes = []
        for node in self.connected_nodes:
            if node.distance_to_origin < self.distance_to_origin or node in self.signal.visited_nodes:
                continue
            else:
                viable_nodes.append(node)
        if not viable_nodes:
            # Expected: signal has exhausted all unvisited outward-reachable nodes.
            # Logged at DEBUG (file only) so it never floods the console during training.
            self.display("No viable connected nodes to forward the signal.", 1, Loud=False)
            self.signal.signal_life = 0  # Expire the signal
            self.signal = None           # Release so node doesn't reprocess it
            return False
        total_distance = sum(node.distance_to_origin for node in viable_nodes)
        probabilities = [node.distance_to_origin / total_distance for node in viable_nodes]
        selected_node = random.choices(viable_nodes, weights=probabilities, k=1)[0]
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

        # 1. Input prediction as a feature
        pred_w = self.weights.get('input_prediction', 1.0)
        weighted_sum = self.signal.prediction * pred_w

        # 2. Weighted feature contributions (same formula as train_process_signal)
        for feature, value in self.signal.input.items():
            w = self.weights.get(feature, 1.0)
            rel = self.signal.feature_relevance.get(feature, 1.0)
            weighted_sum += value * w * rel

        # 3. Distance-based precision scaling
        distance = self.distance_to_origin + 1e-6
        scaled_delta = weighted_sum / (1.0 + distance)

        # Clamp delta to prevent explosion
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

        # 1. Input prediction as a feature
        pred_w = self.weights.get('input_prediction', 1.0)
        prev_prediction = self.signal.prediction
        weighted_sum = self.signal.prediction * pred_w

        # 2. Weighted feature contributions
        feature_details = {}
        for feature, value in self.signal.input.items():
            w = self.weights.get(feature, 1.0)
            rel = self.signal.feature_relevance.get(feature, 1.0)
            contrib = value * w * rel
            weighted_sum += contrib
            feature_details[feature] = {'value': value, 'weight': w, 'relevance': rel}

        # 3. Distance-based scaling
        distance = self.distance_to_origin + 1e-6
        scaled_delta = weighted_sum / (1.0 + distance)

        # Clamp delta to prevent explosion
        scaled_delta = max(-self.DELTA_CLIP, min(self.DELTA_CLIP, scaled_delta))

        # 4. Update prediction
        self.signal.prediction += scaled_delta
        self.signal.prediction = max(-self.PRED_CLIP, min(self.PRED_CLIP, self.signal.prediction))

        if hasattr(self.signal, 'variance'):
            self.signal.variance += abs(scaled_delta)

        # 5. Record contribution for gradient computation
        self.signal.path_contributions[id(self)] = {
            'node': self,
            'scaled_delta': scaled_delta,
            'raw_delta': weighted_sum,
            'distance': distance,
            'feature_details': feature_details,
            'pred_weight': pred_w,
            'prev_prediction': prev_prediction
        }

        return scaled_delta

    def accumulate_weight_gradient(self, dL_dpred, signal):
        """Accumulate weight gradients from one signal path"""
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        distance = contrib['distance']
        scale = 1.0 / (1.0 + distance)

        for feature, fd in contrib['feature_details'].items():
            dL_dw = dL_dpred * fd['value'] * fd['relevance'] * scale
            dL_dw = max(-self.GRAD_CLIP, min(self.GRAD_CLIP, dL_dw))  # Clip per-element gradient
            self.weight_gradients[feature] = self.weight_gradients.get(feature, 0.0) + dL_dw

        dL_dw_pred = dL_dpred * contrib['prev_prediction'] * scale
        dL_dw_pred = max(-self.GRAD_CLIP, min(self.GRAD_CLIP, dL_dw_pred))
        self.weight_gradients['input_prediction'] = self.weight_gradients.get('input_prediction', 0.0) + dL_dw_pred

    def accumulate_position_gradient(self, dL_dpred, signal):
        """Accumulate position gradients from one signal path"""
        contrib = signal.path_contributions.get(id(self))
        if contrib is None:
            return

        raw_delta = contrib['raw_delta']
        distance = contrib['distance']

        for j, p in enumerate(self.position):
            if distance < 1e-9:
                continue
            dscale_dpos = -p / (distance * (1.0 + distance) ** 2)
            grad_j = dL_dpred * raw_delta * dscale_dpos
            grad_j = max(-self.GRAD_CLIP, min(self.GRAD_CLIP, grad_j))  # Clip
            self.position_gradient[j] += grad_j

    def apply_weight_gradient(self, learning_rate):
        """Apply accumulated weight gradients"""
        for feature in self.weight_gradients:
            current = self.weights.get(feature, 1.0)
            updated = current - learning_rate * self.weight_gradients[feature]
            self.weights[feature] = max(-self.WEIGHT_CLIP, min(self.WEIGHT_CLIP, updated))  # Clamp weights
        self.weight_gradients = {}

    def apply_position_gradient(self, learning_rate, max_step):
        """Apply position gradient with circle-constrained step size"""
        new_position = list(self.position)
        for j in range(len(self.position)):
            step = -learning_rate * self.position_gradient[j]
            step = max(-max_step, min(max_step, step))  # Clamp to circle radius
            new_position[j] = self.position[j] + step
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

