"""
from notes:
- judge : puts weights on different categories potentially based on category variables : distributes based on top weights
  -- note useful categories: age, gender, study hours, attendance, internet access, sleep hours, sleep quality, study method, facility rating, exam difficulty
  ---exam score is goal
  -- determines relevance of different features to find splitter priority (lowest 25% splitters are dropped for current input)
  -- will connect to splitter nodes to begin NN processing
  -- lightweight prediction to determine both feature relevance (depending on other features) and segment relevance (based on predicted variance of segments)
  -- can be static model if needed
  -- for each input features will be ranked based on current input. variance will be predicted for each segment
  --- allowing for segment pruning and feature weighting in predictions
feature priority determiner
routing determination

Feature Priority Flow:
    1. calculate_feature_relevance(data) → {feature: score}
    2. rank_features(relevance_scores) → [(feature, score), ...] sorted desc
    3. get_priority_features(ranked_features, drop_percentile=0.25) → top features (drops lowest 25%)

Gaussian Predictive Approach:
    - Each segment has its own BayesianRidge for per-segment feature relevance and variance
    - Global model used when segments predict similarly (use_global_model flag)
    - Feature relevance = coefficient * input_value (per-input contribution)
    - Segment relevance = inverse of predicted variance
"""

import numpy as np
from sklearn.linear_model import BayesianRidge
from .BaseNode import BaseNode

class JudgeNode(BaseNode):
    def __init__(self, position, splitters=None, dataset_features=None, segments=None, learning=False):
        self.position = position
        self.splitters = splitters or []
        self.dataset_features = dataset_features or []  # Feature names
        self.segments = segments or []  # Segment references
        self.learning = learning
        
        # Global Bayesian Ridge model (used when segments predict similarly)
        self.global_model = BayesianRidge(compute_score=True)
        self.global_is_fitted = False
        self.global_coefficients = None
        self.global_alpha_ = None
        self.global_lambda_ = None
        
        # Per-segment models and optimization flag
        self.use_global_model = False  # Auto-set when segment predictions are similar
        self.similarity_threshold = 0.1  # Threshold for considering predictions "similar"
        
        # Per-segment variance learning
        self._init_segment_models()

    def _init_segment_models(self):
        """Initialize per-segment BayesianRidge models and variance arrays."""
        n_seg = max(len(self.segments), 1)
        n_feat = max(len(self.dataset_features), 1)
        
        # Per-segment models
        self.segment_models = [BayesianRidge(compute_score=True) for _ in range(n_seg)]
        self.segment_is_fitted = [False] * n_seg
        self.segment_coefficients = [None] * n_seg
        self.segment_alpha = [None] * n_seg  # Per-segment noise precision
        self.segment_lambda = [None] * n_seg  # Per-segment weight precision
        
        # Per-segment variance tracking
        self.segment_variance_weights = np.ones((n_seg, n_feat)) * 0.1
        self.segment_base_variance = np.ones(n_seg)
        self.segment_variance_count = np.zeros(n_seg)

    def fit_global(self, X_train, y_train, feature_names=None):
        """Fit the global Bayesian Ridge model on training data."""
        self.global_model.fit(X_train, y_train)
        self.global_coefficients = self.global_model.coef_
        self.global_alpha_ = self.global_model.alpha_
        self.global_lambda_ = self.global_model.lambda_
        self.global_is_fitted = True
        if feature_names is not None:
            self.dataset_features = feature_names

    def fit_segment(self, segment_idx, X_train, y_train):
        """Fit a specific segment's Bayesian Ridge model."""
        if segment_idx >= len(self.segment_models):
            return
        self.segment_models[segment_idx].fit(X_train, y_train)
        self.segment_coefficients[segment_idx] = self.segment_models[segment_idx].coef_
        self.segment_alpha[segment_idx] = self.segment_models[segment_idx].alpha_
        self.segment_lambda[segment_idx] = self.segment_models[segment_idx].lambda_
        self.segment_is_fitted[segment_idx] = True

    def fit(self, X_train, y_train, feature_names=None):
        """Fit all models (global + all segments) on training data."""
        self.fit_global(X_train, y_train, feature_names)
        for i in range(len(self.segments)):
            self.fit_segment(i, X_train, y_train)

    # Configuration methods
    def set_splitters(self, splitters):
        self.splitters = splitters

    def set_dataset_features(self, features):
        self.dataset_features = features

    def set_segments(self, segments):
        self.segments = segments
        self._init_segment_models()

    def _input_to_array(self, data):
        """Convert input to numpy array."""
        if isinstance(data, dict):
            return np.array([data.get(f, 0) for f in self.dataset_features], dtype=np.float64)
        return np.asarray(data, dtype=np.float64)

    # Feature Relevance determination methods
    def calculate_feature_relevance(self, data, segment_idx=None):
        """
        Compute per-input feature relevance using segment-specific or global model.
        Args:
            data: dict {feature_name: value} or np.ndarray
            segment_idx: int, if provided uses that segment's model; else uses global
        Returns: {feature_name: relevance_score} dictionary
        """
        input_values = self._input_to_array(data)
        
        # Determine which coefficients to use
        if segment_idx is not None and segment_idx < len(self.segment_is_fitted):
            if self.segment_is_fitted[segment_idx] and not self.use_global_model:
                coefs = self.segment_coefficients[segment_idx]
            elif self.global_is_fitted:
                coefs = self.global_coefficients
            else:
                return {f: 1.0 for f in self.dataset_features}
        elif self.global_is_fitted:
            coefs = self.global_coefficients
        else:
            return {f: 1.0 for f in self.dataset_features}
        
        if coefs is None:
            return {f: 1.0 for f in self.dataset_features}
        
        coefs = np.asarray(coefs, dtype=np.float64)
        contributions = coefs * input_values
        return {f: float(c) for f, c in zip(self.dataset_features, contributions)}

    def calculate_all_segment_relevances(self, data):
        """
        Calculate feature relevance for all segments and check similarity.
        Returns: {segment_id: {feature: score}}
        """
        input_values = self._input_to_array(data)
        all_relevances = {}
        predictions = []
        
        for i, seg in enumerate(self.segments):
            seg_id = getattr(seg, 'id', i)
            rel = self.calculate_feature_relevance(data, segment_idx=i)
            all_relevances[seg_id] = rel
            
            # Track predictions for similarity check
            if self.segment_is_fitted[i] and self.segment_coefficients[i] is not None:
                coefs = np.asarray(self.segment_coefficients[i], dtype=np.float64)
                pred = np.dot(coefs, input_values)
                predictions.append(pred)
        
        # Check if predictions are similar → enable global optimization
        if len(predictions) >= 2:
            pred_std = np.std(predictions)
            pred_mean = np.mean(np.abs(predictions)) + 1e-6
            self.use_global_model = (pred_std / pred_mean) < self.similarity_threshold
        
        return all_relevances

    def rank_features(self, relevance_scores):
        """
        Rank features by absolute relevance score descending.
        Returns: list of (feature, score) tuples
        """
        if relevance_scores is None:
            return []
        return sorted(relevance_scores.items(), key=lambda x: abs(x[1]), reverse=True)

    def get_priority_features(self, ranked_features, drop_percentile=0.25):
        """
        Select top-priority features for this input, dropping the lowest percentile.
        Note: Actual pruning decision is made in main, this just provides the ranking.
        """
        n_drop = int(len(ranked_features) * drop_percentile)
        return ranked_features[:-n_drop] if n_drop > 0 else ranked_features


    # Segment relevance and routing
    def determine_segment_relevance(self, priority_features, input_data=None):
        """
        Predict segment relevance using per-segment learned variance.
        Relevance = 1 / (variance + epsilon). Lower variance = higher relevance.
        Returns: {segment_id: relevance_score}
        """
        if not self.segments:
            return {}
        
        input_arr = self._input_to_array(input_data) if input_data is not None else \
                    self._priority_to_array(priority_features)
        
        segment_variances = self.calculate_segment_variance(input_arr)
        segment_relevance = {}
        for i, seg in enumerate(self.segments):
            seg_id = getattr(seg, 'id', i)
            segment_relevance[seg_id] = 1.0 / (segment_variances[i] + 1e-6)
        
        return segment_relevance

    def _priority_to_array(self, priority_features):
        """Convert priority features to array."""
        input_arr = np.zeros(len(self.dataset_features))
        feat_to_idx = {f: i for i, f in enumerate(self.dataset_features)}
        for f, s in priority_features:
            if f in feat_to_idx:
                input_arr[feat_to_idx[f]] = abs(s)
        return input_arr

    def calculate_segment_variance(self, input_arr):
        """
        Calculate per-segment variance based on features and learned weights.
        Uses segment's alpha (noise precision) if fitted, else base variance.
        Returns: np.ndarray (n_segments,)
        """
        if len(input_arr) != self.segment_variance_weights.shape[1]:
            self._init_segment_models()
        
        variances = np.zeros(len(self.segments))
        for i in range(len(self.segments)):
            if self.segment_is_fitted[i] and self.segment_alpha[i]:
                # Use model's noise variance (1/alpha) + feature contribution
                base = (self.segment_alpha[i] or 1.0)
            else:
                base = self.segment_base_variance[i]
            feature_contrib = self.segment_variance_weights[i] @ np.abs(input_arr)
            variances[i] = base + feature_contrib
        
        return variances

    def update_segment_variance(self, segment_idx, input_arr, observed_variance, lr=0.01):
        """
        Update segment variance model from observed prediction error.
        """
        if segment_idx >= len(self.segments):
            return
        self.segment_variance_count[segment_idx] += 1
        n = self.segment_variance_count[segment_idx]
        delta = observed_variance - self.segment_base_variance[segment_idx]
        self.segment_base_variance[segment_idx] += delta / n
        
        predicted = self.segment_base_variance[segment_idx] + \
                    self.segment_variance_weights[segment_idx] @ np.abs(input_arr)
        error = observed_variance - predicted
        self.segment_variance_weights[segment_idx] += lr * error * np.abs(input_arr)
        self.segment_variance_weights[segment_idx] = np.clip(
            self.segment_variance_weights[segment_idx], 0, 10
        )

    def get_splitter_scores(self, priority_features):
        """
        Score splitters by their associated feature relevance.
        Returns: list of (splitter, score) - no pruning applied here.
        """
        if not self.splitters:
            return []
        
        feature_scores = {f: abs(s) for f, s in priority_features}
        splitter_scores = []
        
        for splitter in self.splitters:
            assigned_feature = getattr(splitter, 'assigned_feature', None)
            if assigned_feature and assigned_feature in feature_scores:
                score = feature_scores[assigned_feature]
            else:
                score = np.mean(list(feature_scores.values())) if feature_scores else 1.0
            splitter_scores.append((splitter, score))
        
        return splitter_scores

    def route_to_splitters(self, data, priority_features, active_splitters=None):
        """
        Route the input to splitters (all or specified subset).
        Signals carry feature relevance, segment relevance/variance, position, and initial variance.
        Returns: list of (splitter, signal_data) tuples
        """
        if active_splitters is None:
            active_splitters = self.splitters
        
        input_arr = self._input_to_array(data)
        all_segment_relevances = self.calculate_all_segment_relevances(data)
        segment_relevance = self.determine_segment_relevance(priority_features, data)
        segment_variance = self.calculate_segment_variance(input_arr)
        
        # Initial variance from global model or default
        initial_variance = (1.0 / self.global_alpha_) if self.global_is_fitted and self.global_alpha_ else 1.0
        
        feature_weights = {f: s for f, s in priority_features}
        segment_var_dict = {getattr(self.segments[i], 'id', i): float(segment_variance[i])
                            for i in range(len(self.segments))}
        
        routed = []
        for splitter in active_splitters:
            signal_data = {
                'input': data,
                'feature_weights': feature_weights,
                'segment_relevance': segment_relevance,
                'segment_feature_relevance': all_segment_relevances,
                'segment_variance': segment_var_dict,
                'position': self.position,
                'initial_variance': initial_variance,
                'use_global_model': self.use_global_model,
            }
            routed.append((splitter, signal_data))
        
        return routed

    # ===== Feedback Updates =====

    def update_feature_relevance_from_feedback(self, feedback, segment_idx=None):
        """
        Update model coefficients incrementally based on prediction feedback.
        """
        if feedback is None:
            return
        error = feedback.get('error', 0)
        input_vals = self._input_to_array(feedback.get('input', {}))
        
        if segment_idx is not None and segment_idx < len(self.segment_is_fitted):
            if self.segment_is_fitted[segment_idx]:
                lr = 0.01 / (self.segment_lambda[segment_idx] or 1.0)
                self.segment_coefficients[segment_idx] += lr * error * input_vals
        elif self.global_is_fitted:
            lr = 0.01 / (self.global_lambda_ or 1.0)
            self.global_coefficients += lr * error * input_vals

    def recalculate_splitter_relevance_from_feedback(self, feedback):
        """
        Recalculate splitter relevance based on prediction feedback.
        """
        if feedback is None:
            return
        input_data = feedback.get('input', None)
        if input_data is None:
            return
        relevance = self.calculate_feature_relevance(input_data)
        ranked = self.rank_features(relevance)
        self.get_priority_features(ranked)

    def update_segment_variance_from_feedback(self, feedback):
        """
        Update per-segment variance from feedback with segment errors.
        """
        if feedback is None:
            return
        input_arr = self._input_to_array(feedback.get('input', {}))
        segment_errors = feedback.get('segment_errors', {})
        for i, seg in enumerate(self.segments):
            seg_id = getattr(seg, 'id', i)
            if seg_id in segment_errors:
                self.update_segment_variance(i, input_arr, segment_errors[seg_id])

    def update_feature_relevance_from_batch_feedback(self, batch_feedback):
        """
        Update model from a batch of feedback samples (re-fit).
        """
        if not batch_feedback:
            return
        X, y = [], []
        for fb in batch_feedback:
            inp, tgt = fb.get('input'), fb.get('target')
            if inp is not None and tgt is not None:
                X.append(self._input_to_array(inp))
                y.append(tgt)
        if X and y:
            self.fit(np.array(X), np.array(y), list(self.dataset_features))

    def recalculate_splitter_relevance_from_batch_feedback(self, batch_feedback):
        for fb in (batch_feedback or []):
            self.recalculate_splitter_relevance_from_feedback(fb)

    # ===== Main Processing =====
    def process(self, data):
        """
        Executes the JudgeNode logic for a single input:
        1. Calculate feature relevance (per-segment)
        2. Rank features
        3. Determine segment relevance/variance
        4. Score splitters
        5. Return routing data (pruning done in main)
        """
        all_segment_relevances = self.calculate_all_segment_relevances(data)
        # Use global or average relevance for ranking
        if self.use_global_model:
            relevance_scores = self.calculate_feature_relevance(data)
        else:
            # Average across segments
            relevance_scores = {}
            for f in self.dataset_features:
                relevance_scores[f] = np.mean([all_segment_relevances[sid].get(f, 0) 
                                               for sid in all_segment_relevances])
        
        ranked_features = self.rank_features(relevance_scores)
        priority_features = self.get_priority_features(ranked_features, drop_percentile=0.25)
        segment_relevance = self.determine_segment_relevance(priority_features, data)
        splitter_scores = self.get_splitter_scores(priority_features)
        
        return {
            'priority_features': priority_features,
            'feature_relevance': relevance_scores,
            'segment_relevance': segment_relevance,
            'segment_feature_relevance': all_segment_relevances,
            'splitter_scores': splitter_scores,
            'use_global_model': self.use_global_model,
            'input_data' : data
        }


    


    