import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import math
from collections import defaultdict, deque
import heapq


class DistanceMetric(Enum):
    """Distance metrics for finding closest nodes"""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    HAMMING = "hamming"
    HYPERDIMENSIONAL = "hyperdimensional"


class RoutingStrategy(Enum):
    """Strategies for routing information to computational nodes"""
    CLOSEST_FIRST = "closest_first"
    LOAD_BALANCED = "load_balanced"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    ADAPTIVE_CLUSTERING = "adaptive_clustering"
    DYNAMIC_ALLOCATION = "dynamic_allocation"


@dataclass
class ComputationalNode:
    """Represents a computational node in the nexus"""
    node_id: str
    position: np.ndarray
    capacity: float
    current_load: float
    processing_speed: float
    specialization_scores: Dict[str, float]
    connection_strength: float
    last_used: int
    performance_history: List[float]
    is_available: bool = True


@dataclass
class RoutingDecision:
    """Decision about routing information to nodes"""
    selected_nodes: List[str]
    allocation_weights: Dict[str, float]
    routing_confidence: float
    expected_processing_time: float
    load_distribution: Dict[str, float]
    reasoning: str


@dataclass
class SplitPacket:
    """Information packet sent to computational nodes"""
    packet_id: str
    source_judge_id: int
    source_segment_id: str
    data: torch.Tensor
    attention_mask: torch.Tensor
    positional_encoding: torch.Tensor
    metadata: Dict[str, Any]
    priority: float
    expected_processing_time: float


@dataclass
class NodeConnection:
    """Connection between nodes with strength and history"""
    source_id: str
    target_id: str
    connection_strength: float
    usage_count: int
    average_latency: float
    success_rate: float
    last_updated: int


class SpatialIndex:
    """Efficient spatial indexing for finding closest nodes"""
    
    def __init__(self, max_dimensions: int = 32):
        self.max_dimensions = max_dimensions
        self.nodes = {}  # node_id -> ComputationalNode
        self.spatial_tree = {}  # For efficient spatial queries
        self.dimension_weights = np.ones(max_dimensions)
    
    def add_node(self, node: ComputationalNode):
        """Add a node to the spatial index"""
        self.nodes[node.node_id] = node
        self._update_spatial_tree()
    
    def remove_node(self, node_id: str):
        """Remove a node from the spatial index"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._update_spatial_tree()
    
    def update_node_load(self, node_id: str, new_load: float):
        """Update a node's current load"""
        if node_id in self.nodes:
            self.nodes[node_id].current_load = new_load
    
    def _update_spatial_tree(self):
        """Update internal spatial tree structure (simplified implementation)"""
        # In practice, this would use a KD-tree or similar structure
        pass
    
    def find_closest_nodes(
        self, 
        query_position: np.ndarray, 
        k: int = 10,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        filter_available: bool = True
    ) -> List[Tuple[str, float]]:
        """Find k closest nodes to a query position"""
        
        distances = []
        
        for node_id, node in self.nodes.items():
            if filter_available and not node.is_available:
                continue
            
            # Calculate distance based on metric
            if distance_metric == DistanceMetric.EUCLIDEAN:
                dist = np.linalg.norm(query_position - node.position)
            elif distance_metric == DistanceMetric.MANHATTAN:
                dist = np.sum(np.abs(query_position - node.position))
            elif distance_metric == DistanceMetric.COSINE:
                dist = 1 - np.dot(query_position, node.position) / (
                    np.linalg.norm(query_position) * np.linalg.norm(node.position) + 1e-8
                )
            elif distance_metric == DistanceMetric.HAMMING:
                dist = np.sum(query_position != node.position) / len(query_position)
            elif distance_metric == DistanceMetric.HYPERDIMENSIONAL:
                # Custom hyperdimensional distance with dimension weighting
                weighted_diff = (query_position - node.position) * self.dimension_weights[:len(query_position)]
                dist = np.linalg.norm(weighted_diff)
            else:
                dist = np.linalg.norm(query_position - node.position)
            
            distances.append((node_id, dist))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def find_nodes_in_radius(
        self, 
        query_position: np.ndarray, 
        radius: float,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    ) -> List[Tuple[str, float]]:
        """Find all nodes within a given radius"""
        
        nodes_in_radius = []
        
        for node_id, node in self.nodes.items():
            if not node.is_available:
                continue
            
            if distance_metric == DistanceMetric.EUCLIDEAN:
                dist = np.linalg.norm(query_position - node.position)
            elif distance_metric == DistanceMetric.MANHATTAN:
                dist = np.sum(np.abs(query_position - node.position))
            else:
                dist = np.linalg.norm(query_position - node.position)
            
            if dist <= radius:
                nodes_in_radius.append((node_id, dist))
        
        return sorted(nodes_in_radius, key=lambda x: x[1])


class ConnectionManager:
    """Manages connections between nodes and tracks their performance"""
    
    def __init__(self):
        self.connections = {}  # (source, target) -> NodeConnection
        self.connection_history = defaultdict(list)
        self.performance_cache = {}
    
    def add_connection(self, source_id: str, target_id: str, initial_strength: float = 1.0):
        """Add a new connection between nodes"""
        conn_key = (source_id, target_id)
        
        if conn_key not in self.connections:
            self.connections[conn_key] = NodeConnection(
                source_id=source_id,
                target_id=target_id,
                connection_strength=initial_strength,
                usage_count=0,
                average_latency=0.0,
                success_rate=1.0,
                last_updated=0
            )
    
    def update_connection_performance(
        self, 
        source_id: str, 
        target_id: str, 
        latency: float, 
        success: bool,
        timestamp: int
    ):
        """Update connection performance metrics"""
        conn_key = (source_id, target_id)
        
        if conn_key in self.connections:
            conn = self.connections[conn_key]
            
            # Update usage count
            conn.usage_count += 1
            
            # Update average latency with exponential moving average
            alpha = 0.1
            if conn.average_latency == 0.0:
                conn.average_latency = latency
            else:
                conn.average_latency = alpha * latency + (1 - alpha) * conn.average_latency
            
            # Update success rate
            if success:
                conn.success_rate = alpha * 1.0 + (1 - alpha) * conn.success_rate
            else:
                conn.success_rate = alpha * 0.0 + (1 - alpha) * conn.success_rate
            
            # Update connection strength based on performance
            performance_factor = conn.success_rate * (1.0 / (1.0 + conn.average_latency))
            conn.connection_strength = alpha * performance_factor + (1 - alpha) * conn.connection_strength
            
            conn.last_updated = timestamp
            
            # Store in history
            self.connection_history[conn_key].append({
                'timestamp': timestamp,
                'latency': latency,
                'success': success,
                'strength': conn.connection_strength
            })
    
    def get_connection_strength(self, source_id: str, target_id: str) -> float:
        """Get connection strength between two nodes"""
        conn_key = (source_id, target_id)
        if conn_key in self.connections:
            return self.connections[conn_key].connection_strength
        return 0.0
    
    def get_best_connections(self, source_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Get best connections from a source node"""
        connections_from_source = [
            (conn.target_id, conn.connection_strength)
            for conn in self.connections.values()
            if conn.source_id == source_id
        ]
        
        # Sort by connection strength
        connections_from_source.sort(key=lambda x: x[1], reverse=True)
        return connections_from_source[:limit]


class SplitterNode(nn.Module):
    """
    Splitter node that receives judge outputs and routes them to the closest 1% 
    of computational nodes based on space and connections.
    """
    
    def __init__(
        self,
        splitter_id: str,
        embedding_dim: int = 1024,
        max_computational_nodes: int = 10000,
        top_percentage: float = 0.01,
        routing_strategy: RoutingStrategy = RoutingStrategy.PERFORMANCE_WEIGHTED,
        distance_metric: DistanceMetric = DistanceMetric.HYPERDIMENSIONAL
    ):
        super().__init__()
        
        self.splitter_id = splitter_id
        self.embedding_dim = embedding_dim
        self.max_computational_nodes = max_computational_nodes
        self.top_percentage = top_percentage
        self.routing_strategy = routing_strategy
        self.distance_metric = distance_metric
        
        # Calculate number of nodes to select (closest 1%)
        self.target_nodes_count = max(1, int(max_computational_nodes * top_percentage))
        
        # Spatial indexing for efficient node lookup
        self.spatial_index = SpatialIndex()
        
        # Connection management
        self.connection_manager = ConnectionManager()
        
        # Neural components for intelligent routing
        self.routing_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 64),  # Route selection features
            nn.Sigmoid()
        )
        
        self.load_predictor = nn.Sequential(
            nn.Linear(embedding_dim + 10, 64),  # +10 for node features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.priority_estimator = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Positional encoding generator
        self.positional_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Performance tracking
        self.routing_history = deque(maxlen=10000)
        self.performance_metrics = {
            'total_routings': 0,
            'successful_routings': 0,
            'average_latency': 0.0,
            'load_balance_score': 0.0
        }
        
        # Packet management
        self.active_packets = {}
        self.packet_counter = 0
        
    def register_computational_node(
        self, 
        node_id: str, 
        position: np.ndarray,
        capacity: float = 1.0,
        processing_speed: float = 1.0,
        specializations: Optional[Dict[str, float]] = None
    ):
        """Register a new computational node"""
        
        node = ComputationalNode(
            node_id=node_id,
            position=position,
            capacity=capacity,
            current_load=0.0,
            processing_speed=processing_speed,
            specialization_scores=specializations or {},
            connection_strength=1.0,
            last_used=0,
            performance_history=[],
            is_available=True
        )
        
        self.spatial_index.add_node(node)
        
        # Initialize connections to existing nodes (simplified)
        for existing_id in list(self.spatial_index.nodes.keys())[:-1]:  # Exclude the just-added node
            self.connection_manager.add_connection(existing_id, node_id)
            self.connection_manager.add_connection(node_id, existing_id)
    
    def unregister_computational_node(self, node_id: str):
        """Unregister a computational node"""
        self.spatial_index.remove_node(node_id)
    
    def calculate_node_suitability(
        self, 
        node: ComputationalNode,
        task_embedding: torch.Tensor,
        task_type: Optional[str] = None
    ) -> float:
        """Calculate how suitable a node is for a given task"""
        
        # Base suitability factors
        load_factor = 1.0 - (node.current_load / node.capacity)
        speed_factor = node.processing_speed
        availability_factor = 1.0 if node.is_available else 0.0
        
        # Specialization factor
        specialization_factor = 1.0
        if task_type and task_type in node.specialization_scores:
            specialization_factor = node.specialization_scores[task_type]
        
        # Performance history factor
        performance_factor = 1.0
        if node.performance_history:
            performance_factor = np.mean(node.performance_history)
        
        # Neural prediction of suitability
        node_features = torch.tensor([
            node.current_load,
            node.capacity,
            node.processing_speed,
            node.connection_strength,
            specialization_factor,
            performance_factor,
            len(node.performance_history),
            float(node.is_available),
            0.0,  # Reserved
            0.0   # Reserved
        ], dtype=torch.float32)
        
        combined_input = torch.cat([task_embedding.flatten(), node_features])
        neural_suitability = self.load_predictor(combined_input).item()
        
        # Weighted combination
        total_suitability = (
            0.25 * load_factor +
            0.20 * speed_factor +
            0.20 * availability_factor +
            0.15 * specialization_factor +
            0.10 * performance_factor +
            0.10 * neural_suitability
        )
        
        return total_suitability
    
    def select_computational_nodes(
        self,
        judge_position: np.ndarray,
        task_embedding: torch.Tensor,
        segment_outputs: Dict[str, Any],
        task_type: Optional[str] = None
    ) -> RoutingDecision:
        """Select the best computational nodes for processing"""
        
        if self.routing_strategy == RoutingStrategy.CLOSEST_FIRST:
            # Simple distance-based selection
            closest_nodes = self.spatial_index.find_closest_nodes(
                judge_position, 
                self.target_nodes_count * 2,  # Get more candidates
                self.distance_metric
            )
            
            # Filter by availability and capacity
            suitable_nodes = []
            for node_id, distance in closest_nodes:
                node = self.spatial_index.nodes[node_id]
                if node.is_available and node.current_load < node.capacity * 0.9:
                    suitability = 1.0 / (1.0 + distance)  # Inverse distance
                    suitable_nodes.append((node_id, suitability))
            
            # Select top nodes
            suitable_nodes.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node_id for node_id, _ in suitable_nodes[:self.target_nodes_count]]
            
        elif self.routing_strategy == RoutingStrategy.PERFORMANCE_WEIGHTED:
            # Performance and distance weighted selection
            closest_nodes = self.spatial_index.find_closest_nodes(
                judge_position, 
                self.target_nodes_count * 3,
                self.distance_metric
            )
            
            node_scores = []
            for node_id, distance in closest_nodes:
                node = self.spatial_index.nodes[node_id]
                suitability = self.calculate_node_suitability(node, task_embedding, task_type)
                
                # Combine distance and suitability
                distance_score = 1.0 / (1.0 + distance)
                combined_score = 0.6 * suitability + 0.4 * distance_score
                
                node_scores.append((node_id, combined_score))
            
            # Select top scoring nodes
            node_scores.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node_id for node_id, _ in node_scores[:self.target_nodes_count]]
            
        elif self.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            # Load balancing with spatial awareness
            all_nodes = list(self.spatial_index.nodes.items())
            
            # Calculate load-balanced scores
            node_scores = []
            for node_id, node in all_nodes:
                if not node.is_available:
                    continue
                
                distance = np.linalg.norm(judge_position - node.position)
                load_factor = 1.0 - (node.current_load / node.capacity)
                distance_factor = 1.0 / (1.0 + distance)
                
                # Balance load and distance
                balanced_score = 0.7 * load_factor + 0.3 * distance_factor
                node_scores.append((node_id, balanced_score))
            
            # Select top balanced nodes
            node_scores.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node_id for node_id, _ in node_scores[:self.target_nodes_count]]
            
        else:  # ADAPTIVE_CLUSTERING or DYNAMIC_ALLOCATION
            # Advanced selection using neural routing network
            routing_input = torch.cat([
                task_embedding.flatten(),
                torch.tensor(judge_position, dtype=torch.float32)
            ])
            
            routing_features = self.routing_network(routing_input)
            
            # Use routing features to score nodes
            all_nodes = list(self.spatial_index.nodes.items())
            node_scores = []
            
            for node_id, node in all_nodes:
                if not node.is_available:
                    continue
                
                suitability = self.calculate_node_suitability(node, task_embedding, task_type)
                distance = np.linalg.norm(judge_position - node.position)
                
                # Neural-guided scoring
                neural_affinity = torch.cosine_similarity(
                    routing_features.unsqueeze(0),
                    torch.tensor(node.position[:64], dtype=torch.float32).unsqueeze(0)
                ).item()
                
                combined_score = (
                    0.4 * suitability +
                    0.3 * (1.0 / (1.0 + distance)) +
                    0.3 * neural_affinity
                )
                
                node_scores.append((node_id, combined_score))
            
            # Select top scoring nodes
            node_scores.sort(key=lambda x: x[1], reverse=True)
            selected_nodes = [node_id for node_id, _ in node_scores[:self.target_nodes_count]]
        
        # Calculate allocation weights
        total_score = sum(score for _, score in node_scores[:len(selected_nodes)])
        allocation_weights = {}
        
        for i, node_id in enumerate(selected_nodes):
            if i < len(node_scores):
                weight = node_scores[i][1] / total_score if total_score > 0 else 1.0 / len(selected_nodes)
            else:
                weight = 1.0 / len(selected_nodes)
            allocation_weights[node_id] = weight
        
        # Calculate expected processing time
        selected_node_objs = [self.spatial_index.nodes[nid] for nid in selected_nodes]
        avg_speed = np.mean([n.processing_speed for n in selected_node_objs])
        data_size = sum(output.features.numel() for output in segment_outputs.values())
        expected_time = data_size / (avg_speed * 1000)  # Simplified calculation
        
        # Calculate routing confidence
        score_variance = np.var([score for _, score in node_scores[:len(selected_nodes)]])
        routing_confidence = 1.0 / (1.0 + score_variance)
        
        # Calculate load distribution
        load_distribution = {
            node_id: self.spatial_index.nodes[node_id].current_load 
            for node_id in selected_nodes
        }
        
        reasoning = f"Selected {len(selected_nodes)} nodes using {self.routing_strategy.value} strategy"
        
        return RoutingDecision(
            selected_nodes=selected_nodes,
            allocation_weights=allocation_weights,
            routing_confidence=routing_confidence,
            expected_processing_time=expected_time,
            load_distribution=load_distribution,
            reasoning=reasoning
        )
    
    def create_split_packets(
        self,
        judge_id: int,
        segment_outputs: Dict[str, Any],
        attention_masks: Dict[str, torch.Tensor],
        routing_decision: RoutingDecision,
        task_embedding: torch.Tensor
    ) -> List[SplitPacket]:
        """Create packets to send to computational nodes"""
        
        packets = []
        
        for segment_id, segment_output in segment_outputs.items():
            # Generate positional encoding for this segment
            positional_encoding = self.positional_encoder(task_embedding)
            
            # Calculate priority
            priority = self.priority_estimator(task_embedding).item()
            
            # Get attention mask for this segment
            attention_mask = attention_masks.get(segment_id, torch.ones_like(segment_output.features))
            
            # Create packet
            packet_id = f"{self.splitter_id}_packet_{self.packet_counter}"
            self.packet_counter += 1
            
            packet = SplitPacket(
                packet_id=packet_id,
                source_judge_id=judge_id,
                source_segment_id=segment_id,
                data=segment_output.features,
                attention_mask=attention_mask,
                positional_encoding=positional_encoding,
                metadata={
                    'segment_type': getattr(segment_output, 'segment_type', 'unknown'),
                    'confidence_score': getattr(segment_output, 'confidence_score', 0.5),
                    'processing_cost': getattr(segment_output, 'processing_cost', 0.0),
                    'routing_confidence': routing_decision.routing_confidence,
                    'selected_nodes': routing_decision.selected_nodes,
                    'allocation_weights': routing_decision.allocation_weights
                },
                priority=priority,
                expected_processing_time=routing_decision.expected_processing_time
            )
            
            packets.append(packet)
            self.active_packets[packet_id] = packet
        
        return packets
    
    def forward(
        self,
        judge_id: int,
        judge_position: np.ndarray,
        segment_outputs: Dict[str, Any],
        attention_masks: Dict[str, torch.Tensor],
        task_embedding: torch.Tensor,
        task_type: Optional[str] = None
    ) -> Tuple[List[SplitPacket], RoutingDecision]:
        """
        Main forward pass: select nodes and create packets for distribution
        """
        
        # Select computational nodes
        routing_decision = self.select_computational_nodes(
            judge_position, task_embedding, segment_outputs, task_type
        )
        
        # Create split packets
        packets = self.create_split_packets(
            judge_id, segment_outputs, attention_masks, routing_decision, task_embedding
        )
        
        # Update node loads (estimated)
        for node_id in routing_decision.selected_nodes:
            if node_id in self.spatial_index.nodes:
                estimated_load = routing_decision.allocation_weights[node_id] * 0.1  # Simplified
                current_load = self.spatial_index.nodes[node_id].current_load
                self.spatial_index.update_node_load(node_id, current_load + estimated_load)
        
        # Update performance metrics
        self.performance_metrics['total_routings'] += 1
        
        # Store routing decision in history
        self.routing_history.append({
            'judge_id': judge_id,
            'selected_nodes': routing_decision.selected_nodes,
            'confidence': routing_decision.routing_confidence,
            'packet_count': len(packets)
        })
        
        return packets, routing_decision
    
    def update_node_performance(
        self, 
        node_id: str, 
        processing_time: float, 
        success: bool,
        performance_score: float
    ):
        """Update computational node performance metrics"""
        
        if node_id in self.spatial_index.nodes:
            node = self.spatial_index.nodes[node_id]
            
            # Update performance history
            node.performance_history.append(performance_score)
            if len(node.performance_history) > 100:
                node.performance_history.pop(0)
            
            # Update load (assume processing is complete)
            node.current_load = max(0.0, node.current_load - 0.1)
            
            # Update success metrics
            if success:
                self.performance_metrics['successful_routings'] += 1
            
            # Update average latency
            alpha = 0.1
            if self.performance_metrics['average_latency'] == 0.0:
                self.performance_metrics['average_latency'] = processing_time
            else:
                current_avg = self.performance_metrics['average_latency']
                self.performance_metrics['average_latency'] = (
                    alpha * processing_time + (1 - alpha) * current_avg
                )
    
    def get_splitter_status(self) -> Dict[str, Any]:
        """Get current splitter status and metrics"""
        
        total_nodes = len(self.spatial_index.nodes)
        available_nodes = sum(1 for n in self.spatial_index.nodes.values() if n.is_available)
        average_load = np.mean([n.current_load for n in self.spatial_index.nodes.values()])
        
        return {
            'splitter_id': self.splitter_id,
            'total_computational_nodes': total_nodes,
            'available_nodes': available_nodes,
            'target_nodes_per_routing': self.target_nodes_count,
            'average_node_load': average_load,
            'active_packets': len(self.active_packets),
            'routing_strategy': self.routing_strategy.value,
            'distance_metric': self.distance_metric.value,
            'performance_metrics': self.performance_metrics.copy(),
            'recent_routings': len(self.routing_history)
        }


# Example usage and testing
if __name__ == "__main__":
    # Create splitter
    splitter = SplitterNode(
        splitter_id="splitter_001",
        embedding_dim=512,
        max_computational_nodes=1000,
        top_percentage=0.01
    )
    
    # Register some computational nodes
    for i in range(20):
        position = np.random.randn(32)  # Random position in 32D space
        splitter.register_computational_node(
            node_id=f"node_{i}",
            position=position,
            capacity=1.0,
            processing_speed=np.random.uniform(0.5, 2.0),
            specializations={'llm': np.random.uniform(0.3, 0.9)}
        )
    
    # Create sample data
    judge_position = np.random.randn(32)
    task_embedding = torch.randn(1, 512)
    
    # Mock segment outputs
    segment_outputs = {
        'segment_1': type('SegmentOutput', (), {
            'features': torch.randn(1, 512),
            'confidence_score': 0.8,
            'processing_cost': 0.1
        })(),
        'segment_2': type('SegmentOutput', (), {
            'features': torch.randn(1, 512),
            'confidence_score': 0.6,
            'processing_cost': 0.2
        })()
    }
    
    attention_masks = {
        'segment_1': torch.rand(1, 512),
        'segment_2': torch.rand(1, 512)
    }
    
    # Process through splitter
    packets, routing_decision = splitter(
        judge_id=42,
        judge_position=judge_position,
        segment_outputs=segment_outputs,
        attention_masks=attention_masks,
        task_embedding=task_embedding,
        task_type='llm'
    )
    
    # Print results
    print(f"Created {len(packets)} packets")
    print(f"Selected {len(routing_decision.selected_nodes)} computational nodes")
    print(f"Routing confidence: {routing_decision.routing_confidence:.3f}")
    print(f"Expected processing time: {routing_decision.expected_processing_time:.3f}")