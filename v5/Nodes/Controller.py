import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math


class TaskType(Enum):
    """Supported AI task types"""
    LLM = "language_modeling"
    VISION = "computer_vision"
    AUDIO = "audio_processing"
    MULTIMODAL = "multimodal"
    REASONING = "logical_reasoning"
    GENERATION = "content_generation"


@dataclass
class JudgeMetrics:
    """Metrics for judge performance and relevance"""
    relevance_score: float
    historical_performance: float
    task_affinity: float
    dimensional_position: List[float]
    activation_count: int
    last_used: int


@dataclass
class ProcessingTask:
    """Task specification for the nexus"""
    task_type: TaskType
    input_data: torch.Tensor
    context: Optional[Dict[str, Any]] = None
    priority: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


class HyperdimensionalMapper:
    """Maps judges to hyperdimensional space positions"""
    
    def __init__(self, max_dimensions: int = 32):
        self.max_dimensions = max_dimensions
        self.dimension_cache = {}
    
    def calculate_judge_positions(self, judge_count: int) -> np.ndarray:
        """Calculate hyperdimensional positions for judges"""
        if judge_count in self.dimension_cache:
            return self.dimension_cache[judge_count]
        
        # Calculate required dimensions (each pair of judges adds a dimension)
        dimensions = math.ceil(math.log2(judge_count)) if judge_count > 1 else 1
        dimensions = min(dimensions, self.max_dimensions)
        
        positions = np.zeros((judge_count, dimensions))
        
        # Assign judges to dimensional polarities
        for i in range(judge_count):
            for dim in range(dimensions):
                # Distribute judges across +/- polarities in each dimension
                bit_pos = (i >> dim) & 1
                positions[i, dim] = 1.0 if bit_pos else -1.0
        
        self.dimension_cache[judge_count] = positions
        return positions


class NexusController(nn.Module):
    """
    Main controller for the AI NN Nexus system.
    Manages dynamic judge selection, dimensional mapping, and task distribution.
    """
    
    def __init__(
        self,
        total_judges: int = 1000,
        max_active_judges: int = 500,
        embedding_dim: int = 1024,
        max_dimensions: int = 32,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        
        # Core parameters
        self.total_judges = total_judges
        self.max_active_judges = max_active_judges
        self.embedding_dim = embedding_dim
        self.max_dimensions = max_dimensions
        
        # Judge management
        self.judge_metrics = {}
        self.active_judges = set()
        self.dimensional_mapper = HyperdimensionalMapper(max_dimensions)
        
        # Neural components
        self.task_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.judge_selector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, total_judges),
            nn.Softmax(dim=-1)
        )
        
        self.relevance_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Judge embeddings (learnable)
        self.judge_embeddings = nn.Parameter(
            torch.randn(total_judges, embedding_dim) * 0.1
        )
        
        # Task type embeddings
        self.task_type_embeddings = nn.Parameter(
            torch.randn(len(TaskType), embedding_dim) * 0.1
        )
        
        # Initialize judge metrics
        self._initialize_judge_metrics()
        
        # Training components
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.training_step = 0
    
    def _initialize_judge_metrics(self):
        """Initialize metrics for all judges"""
        for judge_id in range(self.total_judges):
            self.judge_metrics[judge_id] = JudgeMetrics(
                relevance_score=0.5,
                historical_performance=0.5,
                task_affinity=0.5,
                dimensional_position=[0.0] * self.max_dimensions,
                activation_count=0,
                last_used=0
            )
    
    def encode_task(self, task: ProcessingTask) -> torch.Tensor:
        """Encode task into embedding space"""
        # Get task type embedding
        task_type_idx = list(TaskType).index(task.task_type)
        task_type_emb = self.task_type_embeddings[task_type_idx]
        
        # Process input data to get task embedding
        if task.input_data.dim() == 1:
            task_emb = task.input_data.unsqueeze(0)
        else:
            task_emb = task.input_data.mean(dim=0, keepdim=True)
        
        # Ensure correct embedding dimension
        if task_emb.shape[-1] != self.embedding_dim:
            task_emb = F.adaptive_avg_pool1d(
                task_emb.unsqueeze(0), self.embedding_dim
            ).squeeze(0)
        
        # Combine task type and task content
        combined_emb = task_emb + task_type_emb.unsqueeze(0)
        
        # Encode through task encoder
        encoded_task = self.task_encoder(combined_emb)
        
        return encoded_task
    
    def select_active_judges(self, task_embedding: torch.Tensor) -> List[int]:
        """Select top judges for the given task"""
        # Get judge selection probabilities
        judge_probs = self.judge_selector(task_embedding).squeeze(0)
        
        # Calculate relevance scores for each judge
        relevance_scores = torch.zeros(self.total_judges)
        
        for judge_id in range(self.total_judges):
            # Combine neural prediction with historical metrics
            judge_emb = self.judge_embeddings[judge_id].unsqueeze(0)
            combined_emb = torch.cat([task_embedding, judge_emb], dim=-1)
            neural_relevance = self.relevance_predictor(combined_emb).item()
            
            # Incorporate historical performance
            historical = self.judge_metrics[judge_id].historical_performance
            task_affinity = self.judge_metrics[judge_id].task_affinity
            
            # Weighted combination
            relevance_scores[judge_id] = (
                0.4 * neural_relevance +
                0.3 * historical +
                0.2 * task_affinity +
                0.1 * judge_probs[judge_id].item()
            )
        
        # Select top 50% judges
        top_k = min(self.max_active_judges, self.total_judges // 2)
        top_judges = torch.topk(relevance_scores, top_k).indices.tolist()
        
        return top_judges
    
    def calculate_dimensional_positions(self, active_judges: List[int]) -> Dict[int, np.ndarray]:
        """Calculate hyperdimensional positions for active judges"""
        judge_count = len(active_judges)
        positions = self.dimensional_mapper.calculate_judge_positions(judge_count)
        
        judge_positions = {}
        for i, judge_id in enumerate(active_judges):
            judge_positions[judge_id] = positions[i]
            # Update stored position
            self.judge_metrics[judge_id].dimensional_position = positions[i].tolist()
        
        return judge_positions
    
    def generate_attention_masks(
        self, 
        active_judges: List[int], 
        task_embedding: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Generate attention masks for each active judge"""
        attention_masks = {}
        
        for judge_id in active_judges:
            judge_emb = self.judge_embeddings[judge_id]
            
            # Calculate attention based on judge-task compatibility
            compatibility = torch.cosine_similarity(
                task_embedding.squeeze(0), 
                judge_emb, 
                dim=0
            )
            
            # Generate attention mask (simplified - in practice would be more complex)
            mask_size = task_embedding.shape[-1]
            attention_mask = torch.softmax(
                judge_emb[:mask_size] * compatibility, 
                dim=0
            )
            
            attention_masks[judge_id] = attention_mask
        
        return attention_masks
    
    def update_judge_metrics(
        self, 
        judge_id: int, 
        performance: float, 
        task_type: TaskType
    ):
        """Update metrics for a judge based on performance"""
        metrics = self.judge_metrics[judge_id]
        
        # Update historical performance with exponential moving average
        alpha = 0.1
        metrics.historical_performance = (
            alpha * performance + (1 - alpha) * metrics.historical_performance
        )
        
        # Update task affinity
        metrics.task_affinity = (
            alpha * performance + (1 - alpha) * metrics.task_affinity
        )
        
        # Update usage statistics
        metrics.activation_count += 1
        metrics.last_used = self.training_step
    
    def forward(self, task: ProcessingTask) -> Dict[str, Any]:
        """
        Main forward pass for task processing
        Returns judge assignments and control signals
        """
        # Encode the task
        task_embedding = self.encode_task(task)
        
        # Select active judges
        active_judges = self.select_active_judges(task_embedding)
        
        # Calculate dimensional positions
        judge_positions = self.calculate_dimensional_positions(active_judges)
        
        # Generate attention masks
        attention_masks = self.generate_attention_masks(active_judges, task_embedding)
        
        # Update active judges set
        self.active_judges = set(active_judges)
        
        # Prepare control signals
        control_signals = {
            'task_embedding': task_embedding,
            'active_judges': active_judges,
            'judge_positions': judge_positions,
            'attention_masks': attention_masks,
            'judge_strengths': {
                judge_id: self.judge_metrics[judge_id].relevance_score 
                for judge_id in active_judges
            }
        }
        
        return control_signals
    
    def train_step(self, task: ProcessingTask, performance_feedback: Dict[int, float]):
        """Training step with performance feedback"""
        self.training_step += 1
        
        # Update judge metrics based on feedback
        for judge_id, performance in performance_feedback.items():
            self.update_judge_metrics(judge_id, performance, task.task_type)
        
        # Calculate loss for judge selection improvement
        if len(performance_feedback) > 0:
            task_embedding = self.encode_task(task)
            judge_probs = self.judge_selector(task_embedding).squeeze(0)
            
            # Create target based on performance feedback
            target = torch.zeros(self.total_judges)
            for judge_id, perf in performance_feedback.items():
                target[judge_id] = perf
            
            # Normalize target
            if target.sum() > 0:
                target = target / target.sum()
            
            # Calculate loss
            loss = F.kl_div(
                torch.log(judge_probs + 1e-8), 
                target, 
                reduction='batchmean'
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        
        return 0.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        active_count = len(self.active_judges)
        avg_performance = np.mean([
            m.historical_performance for m in self.judge_metrics.values()
        ])
        
        return {
            'training_step': self.training_step,
            'active_judges_count': active_count,
            'total_judges': self.total_judges,
            'average_performance': avg_performance,
            'active_judge_ids': list(self.active_judges)
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize controller
    controller = NexusController(
        total_judges=100,
        max_active_judges=50,
        embedding_dim=512
    )
    
    # Create sample task
    sample_task = ProcessingTask(
        task_type=TaskType.LLM,
        input_data=torch.randn(512),  # Sample input embedding
        priority=1.0
    )
    
    # Process task
    control_signals = controller(sample_task)
    
    # Print results
    print("Active judges:", len(control_signals['active_judges']))
    print("Judge positions shape:", len(control_signals['judge_positions']))
    print("System status:", controller.get_system_status())