import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import math
from collections import deque


class SegmentType(Enum):
    """Types of neural brain segments"""
    ATTENTION = "attention_segment"
    FEED_FORWARD = "feed_forward_segment"
    CONVOLUTIONAL = "convolutional_segment"
    RECURRENT = "recurrent_segment"
    TRANSFORMER = "transformer_segment"
    MEMORY = "memory_segment"
    REASONING = "reasoning_segment"


class ActivationStrategy(Enum):
    """Strategies for segment activation"""
    GRADIENT_BASED = "gradient_activation"
    ATTENTION_WEIGHTED = "attention_weighted"
    THRESHOLD_GATED = "threshold_gated"
    DYNAMIC_ROUTING = "dynamic_routing"


@dataclass
class AttentionMask:
    """Attention mask with metadata"""
    mask: torch.Tensor
    confidence: float
    focus_regions: List[Tuple[int, int]]  # (start, end) pairs
    attention_type: str
    timestamp: int


@dataclass
class SegmentOutput:
    """Output from a brain segment"""
    features: torch.Tensor
    attention_weights: torch.Tensor
    confidence_score: float
    processing_cost: float
    segment_id: str
    metadata: Dict[str, Any]


@dataclass
class JudgeDecision:
    """Decision made by judge"""
    selected_segments: List[str]
    attention_allocation: Dict[str, float]
    confidence: float
    reasoning: str
    estimated_performance: float


class BrainSegment(nn.Module):
    """Individual brain segment that can be guided by judges"""
    
    def __init__(
        self,
        segment_id: str,
        segment_type: SegmentType,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_sequence_length: int = 2048
    ):
        super().__init__()
        
        self.segment_id = segment_id
        self.segment_type = segment_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length
        
        # Build segment architecture based on type
        self._build_segment()
        
        # Attention mechanism for this segment
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Segment-specific normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Performance tracking
        self.usage_count = 0
        self.average_confidence = 0.5
        self.processing_history = deque(maxlen=100)
    
    def _build_segment(self):
        """Build segment architecture based on type"""
        if self.segment_type == SegmentType.ATTENTION:
            self.core_network = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            
        elif self.segment_type == SegmentType.FEED_FORWARD:
            self.core_network = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.Dropout(0.1)
            )
            
        elif self.segment_type == SegmentType.CONVOLUTIONAL:
            # Assuming input can be reshaped for conv operations
            self.core_network = nn.Sequential(
                nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            
        elif self.segment_type == SegmentType.RECURRENT:
            self.core_network = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
            
        elif self.segment_type == SegmentType.TRANSFORMER:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim,
                dropout=0.1,
                batch_first=True
            )
            self.core_network = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
        elif self.segment_type == SegmentType.MEMORY:
            self.memory_bank = nn.Parameter(torch.randn(1000, self.hidden_dim))
            self.core_network = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            
        elif self.segment_type == SegmentType.REASONING:
            self.core_network = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        
        else:
            # Default architecture
            self.core_network = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
    
    def forward(
        self, 
        input_data: torch.Tensor, 
        attention_mask: Optional[AttentionMask] = None,
        external_attention: Optional[torch.Tensor] = None
    ) -> SegmentOutput:
        """Process input through segment with optional attention guidance"""
        
        batch_size = input_data.shape[0]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if input_data.shape[-1] == attention_mask.mask.shape[-1]:
                input_data = input_data * attention_mask.mask.unsqueeze(0)
        
        # Process through core network based on segment type
        if self.segment_type == SegmentType.CONVOLUTIONAL:
            # Reshape for conv1d if needed
            if input_data.dim() == 2:
                input_data = input_data.unsqueeze(-1)
            if input_data.shape[1] != self.input_dim:
                input_data = input_data.transpose(1, 2)
            
            features = self.core_network(input_data)
            features = features.squeeze(-1)  # Remove spatial dimension
            
        elif self.segment_type == SegmentType.RECURRENT:
            features, _ = self.core_network(input_data)
            features = features[:, -1, :]  # Take last timestep
            
        elif self.segment_type == SegmentType.TRANSFORMER:
            # Ensure input has correct dimension
            if input_data.shape[-1] != self.input_dim:
                linear_proj = nn.Linear(input_data.shape[-1], self.input_dim).to(input_data.device)
                input_data = linear_proj(input_data)
            
            features = self.core_network(input_data)
            features = features.mean(dim=1)  # Global average pooling
            
        elif self.segment_type == SegmentType.MEMORY:
            # Memory-augmented processing
            query = self.core_network(input_data.mean(dim=1, keepdim=True))
            
            # Attention over memory bank
            memory_attention = torch.softmax(
                torch.matmul(query, self.memory_bank.T) / math.sqrt(self.hidden_dim),
                dim=-1
            )
            
            # Retrieve from memory
            retrieved = torch.matmul(memory_attention, self.memory_bank)
            features = query + retrieved
            features = features.squeeze(1)
            
        else:
            # Standard processing for other types
            if input_data.dim() > 2:
                input_data = input_data.view(batch_size, -1)
            features = self.core_network(input_data)
        
        # Ensure features have correct shape
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # Apply layer normalization
        if features.shape[-1] == self.hidden_dim:
            features = self.layer_norm(features)
        
        # Self-attention with external guidance
        if external_attention is not None and features.dim() >= 2:
            attended_features, attention_weights = self.self_attention(
                features.unsqueeze(1) if features.dim() == 2 else features,
                features.unsqueeze(1) if features.dim() == 2 else features,
                features.unsqueeze(1) if features.dim() == 2 else features,
                attn_mask=external_attention if external_attention.dim() == 2 else None
            )
            features = attended_features.squeeze(1) if attended_features.dim() == 3 else attended_features
        else:
            attention_weights = torch.ones(batch_size, 1, features.shape[-1]).to(features.device)
        
        # Output projection
        output_features = self.output_projection(features)
        
        # Calculate confidence based on attention distribution
        if attention_weights.numel() > 1:
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8), 
                dim=-1
            ).mean()
            confidence = 1.0 / (1.0 + attention_entropy.item())
        else:
            confidence = 0.5
        
        # Update tracking
        self.usage_count += 1
        self.average_confidence = 0.9 * self.average_confidence + 0.1 * confidence
        self.processing_history.append(confidence)
        
        # Calculate processing cost (simplified)
        processing_cost = float(input_data.numel() * features.numel()) / 1e6
        
        return SegmentOutput(
            features=output_features,
            attention_weights=attention_weights.mean(dim=0) if attention_weights.dim() > 2 else attention_weights,
            confidence_score=confidence,
            processing_cost=processing_cost,
            segment_id=self.segment_id,
            metadata={
                'segment_type': self.segment_type.value,
                'usage_count': self.usage_count,
                'average_confidence': self.average_confidence
            }
        )


class JudgeNode(nn.Module):
    """
    Judge node that guides and allocates attention to different neural brain segments.
    Each judge can manage multiple segments and make dynamic routing decisions.
    """
    
    def __init__(
        self,
        judge_id: int,
        embedding_dim: int = 1024,
        max_segments: int = 10,
        segment_types: Optional[List[SegmentType]] = None,
        activation_strategy: ActivationStrategy = ActivationStrategy.ATTENTION_WEIGHTED
    ):
        super().__init__()
        
        self.judge_id = judge_id
        self.embedding_dim = embedding_dim
        self.max_segments = max_segments
        self.activation_strategy = activation_strategy
        
        # Default segment types if not provided
        if segment_types is None:
            segment_types = [
                SegmentType.ATTENTION,
                SegmentType.FEED_FORWARD,
                SegmentType.TRANSFORMER,
                SegmentType.REASONING
            ]
        
        # Create brain segments
        self.segments = nn.ModuleDict()
        for i, seg_type in enumerate(segment_types[:max_segments]):
            segment_id = f"judge_{judge_id}_segment_{i}_{seg_type.value}"
            self.segments[segment_id] = BrainSegment(
                segment_id=segment_id,
                segment_type=seg_type,
                input_dim=embedding_dim,
                hidden_dim=embedding_dim,
                output_dim=embedding_dim
            )
        
        # Judge decision network
        self.decision_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Segment selector
        self.segment_selector = nn.Sequential(
            nn.Linear(embedding_dim, len(self.segments)),
            nn.Softmax(dim=-1)
        )
        
        # Attention mask generator
        self.attention_generator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_decisions = 0
        self.performance_history = deque(maxlen=1000)
        
        # Dimensional position (set by controller)
        self.dimensional_position = np.zeros(32)
        
        # Judge specialization (learned through experience)
        self.task_specialization = {}
        
    def set_dimensional_position(self, position: np.ndarray):
        """Set judge's position in hyperdimensional space"""
        self.dimensional_position = position.copy()
    
    def generate_attention_mask(
        self, 
        task_embedding: torch.Tensor,
        controller_mask: Optional[torch.Tensor] = None
    ) -> AttentionMask:
        """Generate attention mask for guiding segments"""
        
        # Generate base attention mask
        base_mask = self.attention_generator(task_embedding)
        
        # Combine with controller-provided mask if available
        if controller_mask is not None:
            if base_mask.shape == controller_mask.shape:
                combined_mask = 0.7 * base_mask + 0.3 * controller_mask
            else:
                # Resize controller mask to match
                combined_mask = base_mask
        else:
            combined_mask = base_mask
        
        # Calculate confidence
        mask_entropy = -torch.sum(
            combined_mask * torch.log(combined_mask + 1e-8)
        )
        confidence = 1.0 / (1.0 + mask_entropy.item() / combined_mask.numel())
        
        # Find focus regions (areas with high attention)
        focus_threshold = combined_mask.mean() + combined_mask.std()
        focus_regions = []
        
        mask_1d = combined_mask.flatten()
        in_region = False
        start_idx = 0
        
        for i, val in enumerate(mask_1d):
            if val > focus_threshold and not in_region:
                start_idx = i
                in_region = True
            elif val <= focus_threshold and in_region:
                focus_regions.append((start_idx, i-1))
                in_region = False
        
        if in_region:
            focus_regions.append((start_idx, len(mask_1d)-1))
        
        return AttentionMask(
            mask=combined_mask,
            confidence=confidence,
            focus_regions=focus_regions,
            attention_type=self.activation_strategy.value,
            timestamp=self.total_decisions
        )
    
    def make_decision(
        self,
        task_embedding: torch.Tensor,
        controller_attention: Optional[torch.Tensor] = None,
        task_context: Optional[Dict[str, Any]] = None
    ) -> JudgeDecision:
        """Make decision about segment activation and attention allocation"""
        
        # Process task through decision network
        decision_features = self.decision_network(task_embedding)
        
        # Select segments to activate
        segment_probs = self.segment_selector(decision_features)
        
        # Apply activation strategy
        if self.activation_strategy == ActivationStrategy.THRESHOLD_GATED:
            threshold = 0.1  # Only activate segments above threshold
            selected_segments = [
                seg_id for i, seg_id in enumerate(self.segments.keys())
                if segment_probs[i] > threshold
            ]
        elif self.activation_strategy == ActivationStrategy.GRADIENT_BASED:
            # Select top segments based on gradients (simplified)
            top_k = min(3, len(self.segments))
            top_indices = torch.topk(segment_probs, top_k).indices
            selected_segments = [
                list(self.segments.keys())[i] for i in top_indices
            ]
        else:  # ATTENTION_WEIGHTED or DYNAMIC_ROUTING
            # Select segments with probability-weighted sampling
            selected_segments = []
            for i, (seg_id, prob) in enumerate(zip(self.segments.keys(), segment_probs)):
                if torch.rand(1).item() < prob.item():
                    selected_segments.append(seg_id)
            
            # Ensure at least one segment is selected
            if not selected_segments:
                best_segment = list(self.segments.keys())[torch.argmax(segment_probs)]
                selected_segments = [best_segment]
        
        # Calculate attention allocation
        attention_allocation = {}
        total_attention = 0.0
        
        for seg_id in selected_segments:
            seg_idx = list(self.segments.keys()).index(seg_id)
            attention_allocation[seg_id] = segment_probs[seg_idx].item()
            total_attention += segment_probs[seg_idx].item()
        
        # Normalize attention allocation
        if total_attention > 0:
            for seg_id in attention_allocation:
                attention_allocation[seg_id] /= total_attention
        
        # Estimate confidence
        combined_features = torch.cat([task_embedding, decision_features], dim=-1)
        confidence = self.confidence_estimator(combined_features).item()
        
        # Generate reasoning (simplified)
        reasoning = f"Selected {len(selected_segments)} segments based on {self.activation_strategy.value}"
        
        # Estimate performance based on historical data
        if len(self.performance_history) > 0:
            estimated_performance = np.mean(list(self.performance_history))
        else:
            estimated_performance = 0.5
        
        self.total_decisions += 1
        
        return JudgeDecision(
            selected_segments=selected_segments,
            attention_allocation=attention_allocation,
            confidence=confidence,
            reasoning=reasoning,
            estimated_performance=estimated_performance
        )
    
    def process_segments(
        self,
        input_data: torch.Tensor,
        judge_decision: JudgeDecision,
        attention_mask: AttentionMask
    ) -> Dict[str, SegmentOutput]:
        """Process input through selected segments"""
        
        segment_outputs = {}
        
        for seg_id in judge_decision.selected_segments:
            if seg_id in self.segments:
                # Get external attention for this segment
                attention_weight = judge_decision.attention_allocation.get(seg_id, 1.0)
                external_attention = attention_mask.mask * attention_weight if attention_mask else None
                
                # Process through segment
                output = self.segments[seg_id](
                    input_data=input_data,
                    attention_mask=attention_mask,
                    external_attention=external_attention
                )
                
                segment_outputs[seg_id] = output
        
        return segment_outputs
    
    def forward(
        self,
        task_embedding: torch.Tensor,
        input_data: torch.Tensor,
        controller_attention: Optional[torch.Tensor] = None,
        task_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, SegmentOutput], JudgeDecision, AttentionMask]:
        """
        Full forward pass: make decision, generate attention, process segments
        """
        
        # Make decision about segment activation
        decision = self.make_decision(task_embedding, controller_attention, task_context)
        
        # Generate attention mask
        attention_mask = self.generate_attention_mask(task_embedding, controller_attention)
        
        # Process through selected segments
        segment_outputs = self.process_segments(input_data, decision, attention_mask)
        
        return segment_outputs, decision, attention_mask
    
    def update_performance(self, success: bool, performance_score: float):
        """Update judge's performance tracking"""
        if success:
            self.successful_decisions += 1
        
        self.performance_history.append(performance_score)
        
        # Update success rate
        self.success_rate = self.successful_decisions / max(1, self.total_decisions)
    
    def get_specialization_score(self, task_type: str) -> float:
        """Get judge's specialization score for a task type"""
        return self.task_specialization.get(task_type, 0.5)
    
    def update_specialization(self, task_type: str, performance: float):
        """Update specialization for a task type"""
        if task_type not in self.task_specialization:
            self.task_specialization[task_type] = 0.5
        
        # Exponential moving average
        alpha = 0.1
        self.task_specialization[task_type] = (
            alpha * performance + (1 - alpha) * self.task_specialization[task_type]
        )
    
    def get_judge_status(self) -> Dict[str, Any]:
        """Get current judge status and metrics"""
        return {
            'judge_id': self.judge_id,
            'total_decisions': self.total_decisions,
            'success_rate': self.successful_decisions / max(1, self.total_decisions),
            'average_performance': np.mean(list(self.performance_history)) if self.performance_history else 0.5,
            'active_segments': len(self.segments),
            'dimensional_position': self.dimensional_position.tolist(),
            'specializations': dict(self.task_specialization),
            'segment_usage': {
                seg_id: segment.usage_count 
                for seg_id, segment in self.segments.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Create a judge node
    judge = JudgeNode(
        judge_id=42,
        embedding_dim=512,
        max_segments=5,
        segment_types=[
            SegmentType.ATTENTION,
            SegmentType.FEED_FORWARD,
            SegmentType.TRANSFORMER,
            SegmentType.REASONING,
            SegmentType.MEMORY
        ]
    )
    
    # Set dimensional position
    judge.set_dimensional_position(np.array([1.0, -1.0, 1.0, -1.0]))
    
    # Create sample inputs
    task_embedding = torch.randn(1, 512)
    input_data = torch.randn(1, 512)
    
    # Process through judge
    segment_outputs, decision, attention_mask = judge(task_embedding, input_data)
    
    # Print results
    print(f"Judge {judge.judge_id} processed task:")
    print(f"Selected segments: {decision.selected_segments}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Attention mask shape: {attention_mask.mask.shape}")
    print(f"Number of segment outputs: {len(segment_outputs)}")
    print("\nJudge status:", judge.get_judge_status())