
import random
import numpy as np
class NeuralNode:
    def __new__(cls, node_id, node_type, node_position, demo=False, config=None):
        # Dynamically import the node class from the Nodes directory
        import importlib
        import os
        # Always import from Nodes/{node}
        module_name = node_type if node_type != "Review" else "Reviewer"
        module_path = f"Nodes.{module_name}"
        try:
            node_module = importlib.import_module(module_path)
            node_class = getattr(node_module, module_name)
        except Exception as e:
            raise ImportError(f"Could not import node class '{module_name}' from '{module_path}': {e}")
        # Build positional args for each node type
        if node_type == "Controller":
            # Controller expects only keyword arguments, allow config override
            ctrl_args = dict(
                max_dimensions=10,
                max_judges_per_dimension=2,
                top_judge_percentage=0.5,
                hypercube_range=(-1000000, 1000000),
                learning_rate=0.01
            )
            if config and isinstance(config, dict):
                ctrl_args.update(config)
            return node_class(**ctrl_args)
        elif node_type == "Judge":
            # Judge expects specific keyword arguments, allow config override
            judge_args = dict(
                judge_id=str(node_id),
                specialization_domain="general",
                embedding_dim=512,
                num_attention_heads=8,
                max_sequence_length=1024,
                dropout_rate=0.1,
                dimensional_position=None,
                vocab_sizes=None
            )
            if config and isinstance(config, dict):
                judge_args.update(config)
            return node_class(**judge_args)
        elif node_type == "Splitter":
            return node_class(node_id, node_position, 1, demo=demo)
        elif node_type == "Computational":
            return node_class(node_id, node_position, 128, demo=demo)
        elif node_type == "Retainer":
            return node_class(node_id, node_position, 1, demo=demo)
        elif node_type == "Reviewer":
            return node_class(node_id, node_position, 1, demo=demo)
        elif node_type == "Handler":
            return node_class(node_id, node_position, 4, demo=demo)
        else:
            return node_class(node_id, node_position, demo=demo)

    def __init__(self, node_id, node_type, node_position, demo=False):
        # This will only be called for base NeuralNode, which should not happen
        self.node_id = node_id
        self.node_type = node_type
        self.node_position = node_position
        self.demo = demo
        # No setup_node_type needed

    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
    def update_node_weights(self, max_random=None, min_random=None, constant=None):
        if max_random is not None:
            self.weights['Max_random'] = max_random
        if min_random is not None:
            self.weights['Min_random'] = min_random
        if constant is not None:
            self.weights['constant'] = constant
    
    def setup_node_type(self, node_type):
        self.node_type = node_type
        import importlib
        module_name = node_type if node_type != "Review" else "Reviewer"
        module_path = f"DragonChild.v5.Nodes.{module_name}"
        try:
            node_module = importlib.import_module(module_path)
            self.node_class = getattr(node_module, module_name)
        except Exception as e:
            raise ValueError(f"Unknown node type or import failed: {node_type} ({e})")
    
    def process(self, token_embeddings):
        """
        Process the input token embeddings.
        This method should be overridden by subclasses to implement specific processing logic.
        """
        raise NotImplementedError("Subclasses must implement this method")
    


def test_node_imports(selected_type=None):
    node_types = [
        "Controller", "Judge", "Splitter", "Computational", "Repeater", "Retainer", "Reviewer", "Handler"
    ]
    if selected_type:
        node_types = [selected_type]
    results = []
    for node_type in node_types:
        try:
            node = NeuralNode(1, node_type, (0, 0, 0), demo=True)
            summary = {
                "type": node_type,
                "class": node.__class__.__name__,
                "module": node.__class__.__module__,
                "attributes": list(vars(node).keys()),
                "success": True,
                "error": None
            }
        except Exception as e:
            summary = {
                "type": node_type,
                "class": None,
                "module": None,
                "attributes": [],
                "success": False,
                "error": str(e)
            }
        results.append(summary)
    print("\n=== Node Import Test Summary ===")
    for res in results:
        print(f"Node Type: {res['type']}")
        print(f"  Success: {res['success']}")
        if res['success']:
            print(f"  Class: {res['class']}")
            print(f"  Module: {res['module']}")
            print(f"  Attributes: {res['attributes']}")
        else:
            print(f"  Error: {res['error']}")
        print("-----------------------------")


if __name__ == "__main__":
    node_types = [
        "Controller", "Judge", "Splitter", "Computational", "Repeater", "Retainer", "Reviewer", "Handler"
    ]
    print("\nSelect a node type to test:")
    for idx, ntype in enumerate(node_types, 1):
        print(f"  {idx}. {ntype}")
    print(f"  {len(node_types)+1}. All")
    selection = input(f"Enter number (1-{len(node_types)+1}): ").strip()
    try:
        sel_idx = int(selection)
        if sel_idx == len(node_types)+1:
            test_node_imports()
        elif 1 <= sel_idx <= len(node_types):
            test_node_imports(selected_type=node_types[sel_idx-1])
        else:
            print("Invalid selection.")
    except Exception:
        print("Invalid input. Please enter a number.")