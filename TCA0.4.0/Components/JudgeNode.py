import random
import math
import numpy as np
import Logger
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class JudgeNode:
    def __init__(self, ignored_features=None, logger=None, classification=None, target=None):
        self.segments = []
        self.segment_weights = {
            'segment': [],
            'clusters': [],
        }
        self.Logger = logger  if logger else Logger.Logger('JudgeNode.log', 4)
        self.classification = classification
        self.ignored_features = ignored_features if ignored_features else []
        self.features = []
        self.mode = "" # special modes can be assigned later
        self.target = target

    def display(self, message, Loud):
        message = f"[JudgeNode]: {message}"
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        self.Logger.log(message, self.classification, Loud)

    def filter_features(self, vector):
        """
        Supports:
        - list / tuple vectors → ignores by index
        - dict vectors        → ignores by feature name
        """

        # Dict-based vector (preprocessing / pandas style)
        if isinstance(vector, dict):
            return [
                v for k, v in vector.items()
                if k not in self.ignored_features
            ]

        # List / tuple vector (numeric pipeline)
        return [
            v for i, v in enumerate(vector)
            if i not in self.ignored_features
        ]
    def euclidean_distance(self, point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def generate_clusters(self, dataset, cluster_count):
        if len(dataset) < cluster_count:
            raise ValueError("Cluster count cannot exceed dataset size.")

        # Initialize centroids randomly
        centroids = random.sample(dataset, cluster_count)

        for _ in range(3):
            clusters = {i: [] for i in range(cluster_count)}

            for point in dataset:
                
                point_f = self.filter_features(point)
                distances = [
                    self.euclidean_distance(point_f, self.filter_features(c))
                    for c in centroids
                ]
                closest = distances.index(min(distances))
                clusters[closest].append(point)

            new_centroids = []
            for points in clusters.values():
                if not points:
                    new_centroids.append(random.choice(dataset))
                    continue
                dim = len(points[0])
                centroid = [
                    sum(p[i] for p in points) / len(points)
                    for i in range(dim)
                ]
                new_centroids.append(centroid)

            centroids = new_centroids

        self.segment_weights['clusters'] = [
            self.cluster_create(c, clusters[i])
            for i, c in enumerate(centroids)
        ]

    def cluster_create(self, centroid: list, points: list) -> dict:
        return {
            'centroid':   centroid,
            'points':     points,
            'segment_id': None,
        }
    
    def calculate_input_segment_relevance(self, input_vectorized, Loud: bool = False) -> dict:
        relevance_scores: dict[str, list] = {
            'clusters': [],
            'scores': []
        }
        input_filtered = self.filter_features(input_vectorized)
        if not self.segment_weights['clusters']:
            self.display("No clusters available for relevance calculation. Using default override with all segments of relevance 1", Loud)
            for sid in self.segment_weights['segment']:
                relevance_scores['clusters'].append({'segment_id': sid, 'centroid': [], 'points': []})
                relevance_scores['scores'].append(1.0)
            return relevance_scores
        for cluster in self.segment_weights['clusters']:
            centroid = cluster['centroid']
            distance = self.euclidean_distance(input_filtered, centroid)
            relevance = 1 / (1 + distance)
            relevance_scores['clusters'].append(cluster)
            relevance_scores['scores'].append(relevance)
        return relevance_scores
    
    def find_relevant_segments(self, relevance_scores: dict, selection_percentage: float = 0.5, Loud: bool = False) -> list[tuple[int, float]]:
        if not relevance_scores['scores']:
            self.display("No relevant segments found. Using all segments as fallback.", Loud=Loud)
            return [(sid, 1.0) for sid in self.segment_weights['segment']]

        amount_to_select = max(1, int(len(relevance_scores['scores']) * selection_percentage))
        scored_clusters = list(zip(relevance_scores['clusters'], relevance_scores['scores']))
        scored_clusters.sort(key=lambda x: x[1], reverse=True)
        selected_clusters = scored_clusters[:amount_to_select]

        # Aggregate by segment_id, keeping highest score per segment
        segment_scores: dict[int, float] = {}
        for cluster, score in selected_clusters:
            sid = cluster.get('segment_id')
            if sid is None:
                self.display("Cluster has no assigned segment_id. Skipping.", Loud=Loud)
                continue
            if sid not in segment_scores or score > segment_scores[sid]:
                segment_scores[sid] = score

        if not segment_scores:
            self.display("No valid segment assignments found. Using all segments as fallback.", Loud=Loud)
            return [(sid, 1.0) for sid in self.segment_weights['segment']]

        return list(segment_scores.items())
    
    def geometric_uniqueness(self, cluster, all_clusters, eps=1e-6):
        centroid = self.filter_features(cluster['centroid'])

        # Distance to nearest other centroid
        distances = []
        for other in all_clusters:
            if other is cluster:
                continue
            d = self.euclidean_distance(
                centroid,
                self.filter_features(other['centroid'])
            )
            distances.append(d)

        if not distances:
            return 0.0

        nearest = min(distances)

        # Cluster radius
        if not cluster['points']:
            return 0.0

        radius = sum(
            self.euclidean_distance(
                centroid,
                self.filter_features(p)
            )
            for p in cluster['points']
        ) / len(cluster['points'])

        return nearest / (radius + eps)

    def behavioral_diversity(self, cluster, all_clusters, dataset):
        def response(cluster, point):
            return 1 / (
                1 + self.euclidean_distance(
                    self.filter_features(point),
                    self.filter_features(cluster['centroid'])
                )
            )

        diffs = []

        for other in all_clusters:
            if other is cluster:
                continue

            total_diff = 0.0
            for p in dataset:
                total_diff += abs(
                    response(cluster, p) - response(other, p)
                )

            diffs.append(total_diff / len(dataset))

        if not diffs:
            return 0.0

        return sum(diffs) / len(diffs)
    
    def information_gain(self, cluster, dataset):
        if not dataset:
            return 0.0

        size_factor = len(cluster['points']) / len(dataset)

        relevance_sum = 0.0
        for p in dataset:
            relevance_sum += 1 / (
                1 + self.euclidean_distance(
                    self.filter_features(p),
                    self.filter_features(cluster['centroid'])
                )
            )

        mean_relevance = relevance_sum / len(dataset)

        return size_factor * mean_relevance



    def assign_clusters_to_segments(self, segments: list) -> None:
        if not segments:
            raise ValueError("Segments list is empty.")

        self.segment_weights['segment'] = [s.segment_id for s in segments]
        clusters = self.segment_weights['clusters']
        n_seg = len(segments)

        # Group similar clusters by running k-means on cluster centroids
        cluster_centroids = [self.filter_features(c['centroid']) for c in clusters]

        step = max(1, len(clusters) // n_seg)
        meta_centroids = [cluster_centroids[min(i * step, len(clusters) - 1)] for i in range(n_seg)]

        groups: dict[int, list[int]] = {i: [] for i in range(n_seg)}
        for _ in range(10):
            groups = {i: [] for i in range(n_seg)}
            for ci, cc in enumerate(cluster_centroids):
                distances = [self.euclidean_distance(cc, mc) for mc in meta_centroids]
                groups[distances.index(min(distances))].append(ci)

            for i in range(n_seg):
                if not groups[i]:
                    continue
                dim = len(cluster_centroids[groups[i][0]])
                meta_centroids[i] = [
                    sum(cluster_centroids[j][d] for j in groups[i]) / len(groups[i])
                    for d in range(dim)
                ]

        for i in range(n_seg):
            for ci in groups[i]:
                clusters[ci]['segment_id'] = segments[i].segment_id
    
    def calculate_cluster_scoring(
        self,
        dataset,
        w1=1.0, #weight for geometric uniqueness
        w2=1.0, #weight for behavioral diversity
        w3=1.0  #weight for information gain
    ):
        scores = []

        clusters = self.segment_weights['clusters']

        for cluster in clusters:
            g = self.geometric_uniqueness(cluster, clusters)
            b = self.behavioral_diversity(cluster, clusters, dataset)
            i = self.information_gain(cluster, dataset)

            score = w1 * g + w2 * b + w3 * i

            cluster['metrics'] = {
                'GeometricUniqueness': g,
                'BehavioralDiversity': b,
                'InformationGain': i,
                'ClusterScore': score
            }

            scores.append(cluster)

        return scores
    def summarize_clusters(self, clusters):
        summary = []
        for i, c in enumerate(clusters):
            m = c.get('metrics', {})
            summary.append({
                "cluster": i,
                "size": len(c['points']),
                "centroid": c['centroid'],
                "GeometricUniqueness": m.get('GeometricUniqueness', 0),
                "BehavioralDiversity": m.get('BehavioralDiversity', 0),
                "InformationGain": m.get('InformationGain', 0),
                "ClusterScore": m.get('ClusterScore', 0),
            })
        return summary

    def train(self, preprocessed_dataset, iterations: int, segments: list | None = None):
        dataset_vectors = [
            list(data_point.values())
            for data_point in preprocessed_dataset.to_dict(orient='records')
        ]
        scores = []
        seg_count = len(segments) if segments is not None else 1
        min_clusters = seg_count
        max_clusters = seg_count * 5
        for i in range(iterations):
            self.display(f"Training iteration {i+1}/{iterations}", True)
            cluster_count = min_clusters + (i * seg_count) # increase cluster count each iteration based on amount of segments
            if cluster_count > max_clusters:
                break
            self.generate_clusters(dataset_vectors, cluster_count)
            scored_clusters = self.calculate_cluster_scoring(dataset_vectors)

            scores.append({
                "cluster_count": cluster_count,
                "avg_score": sum(c['metrics']['ClusterScore'] for c in scored_clusters) / len(scored_clusters),
                "summary": self.summarize_clusters(scored_clusters)
            })
        scores.sort(key=lambda x: x["avg_score"], reverse=True)

        best = scores[0]
        self.display(
            f"Optimal clusters: {best['cluster_count']} | score={best['avg_score']:.4f}",
            True
        )
        self.display("Cluster summaries:", True)
        for c in best['summary']:
            self.display(
                f"Cluster {c['cluster']}: size={c['size']} | GeometricUniqueness={c['GeometricUniqueness']:.4f} | BehavioralDiversity={c['BehavioralDiversity']:.4f} | InformationGain={c['InformationGain']:.4f} | ClusterScore={c['ClusterScore']:.4f}",
                True
            )
        self.display("Summary for each cluster amount:", True)
        for s in scores:
            self.display(
                f"Clusters: {s['cluster_count']} | Avg ClusterScore: {s['avg_score']:.4f}",
                True
            )
            for c in s['summary']:
                self.display(
                    f"  Cluster {c['cluster']}: size={c['size']} | GeometricUniqueness={c['GeometricUniqueness']:.4f} | BehavioralDiversity={c['BehavioralDiversity']:.4f} | InformationGain={c['InformationGain']:.4f} | ClusterScore={c['ClusterScore']:.4f}",
                    True
                )
        self.display("Generating final clusters based on optimal cluster count...", True)
        self.generate_clusters(dataset_vectors, best['cluster_count'])
        self.calculate_cluster_scoring(dataset_vectors)
        if segments is not None:
            self.assign_clusters_to_segments(segments)
        

def plot_clusters_2d(clusters):
    # Collect all points
    all_points: list[list[float]] = []
    labels: list[int] = []

    for i, c in enumerate(clusters):
        for p in c['points']:
            all_points.append(p)
            labels.append(i)

    if not all_points:
        return

    # Reduce to 2D for visualization
    pca = PCA(n_components=2)

    X = np.asarray(all_points, dtype=float)
    reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for i in set(labels):
        xs = [reduced[j, 0] for j in range(len(labels)) if labels[j] == i]
        ys = [reduced[j, 1] for j in range(len(labels)) if labels[j] == i]
        plt.scatter(xs, ys, label=f"Cluster {i}", alpha=0.6)

    plt.legend()
    plt.title("Cluster Geometry (PCA projection)")
    plt.show()

def plot_cluster_scores(clusters):
    names = [f"C{i}" for i in range(len(clusters))]
    g = [c['metrics']['GeometricUniqueness'] for c in clusters]
    b = [c['metrics']['BehavioralDiversity'] for c in clusters]
    i = [c['metrics']['InformationGain'] for c in clusters]

    x = range(len(clusters))
    plt.figure(figsize=(10, 5))
    plt.bar(x, g, label="Geometric")
    plt.bar(x, b, bottom=g, label="Behavioral")
    plt.bar(x, i, bottom=[g[j]+b[j] for j in x], label="Info")

    plt.xticks(x, names)
    plt.legend()
    plt.title("ClusterScore Composition")
    plt.show()