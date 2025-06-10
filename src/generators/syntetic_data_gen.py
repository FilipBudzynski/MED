import numpy as np
import random
from typing import Set, List
from math import exp
import json
import csv


# NUM_ITEMS = 100000
# NUM_ROOTS = 250
# NUM_LEVELS = 4
# FANOUT = 5
# DEPTH_RATIO = 1.0
#
# NUM_TRANSACTIONS = 1000000
# AVG_TRANSACTION_SIZE = 10
# AVG_ITEMSET_SIZE = 4
# NUM_MAX_ITEMSETS = 10000
# CORRELATION_LEVEL = 0.5


NUM_ITEMS = 100
NUM_ROOTS = 50
NUM_LEVELS = 2
FANOUT = 2
DEPTH_RATIO = 1.0

NUM_TRANSACTIONS = 1000
AVG_TRANSACTION_SIZE = 2
AVG_ITEMSET_SIZE = 2
NUM_MAX_ITEMSETS = 8
CORRELATION_LEVEL = 0.5

np.random.seed(42)
random.seed(42)


# TAXONOMY GENERATION (Forest of Trees)
class TaxonomyNode:
    def __init__(self, item_id, level):
        self.item_id = item_id
        self.children = []
        self.level = level
        self.weight = 0.0


class SynteticGenerator:
    def __init__(self) -> None:
        pass

    def _save_transactions_to_csv(self, transactions: List[Set[int]], path: str):
        with open(path+".csv", "w", newline="") as f:
            writer = csv.writer(f)
            for t in transactions:
                writer.writerow(list(t))

    def _save_taxonomy_to_json(self, taxonomy_forest, out_path: str):
        taxonomy_dict = self.taxonomy_to_nested_dict(taxonomy_forest)

        with open(out_path+".json", "w") as f:
            json.dump(taxonomy_dict, f, indent=2)

    def generate_taxonomy(self, num_items, num_roots, num_levels, fanout):
        item_counter = 0
        forest = []
        all_nodes = {}

        def create_subtree(level):
            nonlocal item_counter
            if item_counter >= num_items:
                return None
            node_id = item_counter
            item_counter += 1
            node = TaxonomyNode(node_id, level)
            all_nodes[node_id] = node
            if level < num_levels:
                num_children = np.random.poisson(fanout)
                for _ in range(num_children):
                    child = create_subtree(level + 1)
                    if child:
                        node.children.append(child)
            return node

        for _ in range(num_roots):
            root = create_subtree(0)
            if root:
                forest.append(root)

        return forest, all_nodes

    # assign weights (bottom-up)
    def assign_weights(self, forest, depth_ratio):
        def compute_weight(node):
            if not node.children:
                node.weight = np.random.uniform(0.01, 1.0)
            else:
                for child in node.children:
                    compute_weight(child)
                node.weight = sum(child.weight for child in node.children) / depth_ratio

        for root in forest:
            compute_weight(root)

    # taxonomy to nested dict
    def taxonomy_to_nested_dict(self, forest):
        def node_to_dict(node):
            if not node.children:
                return {}
            return {child.item_id: node_to_dict(child) for child in node.children}

        taxonomy_dict = {}
        for root in forest:
            taxonomy_dict[root.item_id] = node_to_dict(root)
        return taxonomy_dict

    # flatten taxonomy nodes to list of leaves for sampling
    def get_leaves(self, nodes_dict):
        return [node for node in nodes_dict.values() if not node.children]

    # prepare weighted coin toss helper
    def weighted_choice(self, items, weights):
        total = sum(weights)
        rnd = random.uniform(0, total)
        upto = 0
        for item, w in zip(items, weights):
            if upto + w >= rnd:
                return item
            upto += w
        return items[-1]

    # generate weighted probabilities for all items (including non-leaves)
    def build_weight_distribution(self, all_nodes):
        items = list(all_nodes.keys())
        weights = [all_nodes[i].weight for i in items]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        return items, probs

    # specialize an item (internal node) to a leaf by descending weighted random choice
    def specialize_to_leaf(self, node: TaxonomyNode):
        current = node
        while current.children:
            children = current.children
            weights = [child.weight for child in children]
            current = self.weighted_choice(children, weights)
        return current.item_id

    # generate z, the set of potentially frequent itemsets
    def generate_Z(
        self, all_nodes, num_max_itemsets, avg_itemset_size, correlation_level
    ):
        items, item_probs = self.build_weight_distribution(all_nodes)
        Z = []
        weights_Z = []
        prev_itemset = set()

        for i in range(num_max_itemsets):
            size = max(1, np.random.poisson(avg_itemset_size))
            print(f"from genering Z: {i}")

            itemset = set()
            if i == 0:
                while len(itemset) < size:
                    chosen = self.weighted_choice(items, item_probs)
                    itemset.add(chosen)
            else:
                frac = min(1.0, np.random.exponential(correlation_level))
                num_from_prev = int(frac * size)
                num_new = size - num_from_prev

                prev_items_list = list(prev_itemset)
                available_prev = list(set(prev_items_list) - itemset)
                if len(available_prev) >= num_from_prev:
                    chosen_prev = random.sample(available_prev, num_from_prev)
                else:
                    chosen_prev = available_prev  # take all if not enough
                itemset.update(chosen_prev)

                while len(itemset) < size:
                    chosen = self.weighted_choice(items, item_probs)
                    itemset.add(chosen)

            prev_itemset = itemset

            exp_weight = np.random.exponential(1.0)
            geom_mean = exp(
                sum(np.log(item_probs[items.index(it)]) for it in itemset)
                / len(itemset)
            )
            w = exp_weight * geom_mean
            Z.append(itemset)
            weights_Z.append(w)

        total_w = sum(weights_Z)
        weights_Z = [w / total_w for w in weights_Z]

        return Z, weights_Z

    # corruption: drop items from itemset probabilistically by corruption level c
    def corrupt_itemset(self, itemset: Set[int], corruption_level: float) -> Set[int]:
        items = list(itemset)
        retained = []
        for item in items:
            if random.random() >= corruption_level:
                retained.append(item)
            else:
                pass
        if not retained and items:
            retained = [random.choice(items)]
        return set(retained)

    # generate transactions
    def generate_transactions(
        self, Z, weights_Z, all_nodes, num_transactions, avg_transaction_size
    ):
        transactions: List[Set[int]] = []
        num_Z = len(Z)

        corruption_levels = np.clip(np.random.normal(0.5, 0.1, num_Z), 0.0, 1.0)

        for num_transa in range(num_transactions):
            print(num_transa)
            transaction = set()
            target_size = max(1, np.random.poisson(avg_transaction_size))

            while len(transaction) < target_size:
                chosen_index = self.weighted_choice(range(num_Z), weights_Z)
                chosen_itemset = Z[chosen_index]

                specialized_items = set()
                for it_id in chosen_itemset:
                    node = all_nodes[it_id]
                    if node.children:
                        leaf_id = self.specialize_to_leaf(node)
                        specialized_items.add(leaf_id)
                    else:
                        specialized_items.add(it_id)

                corrupted_items = self.corrupt_itemset(
                    specialized_items, corruption_levels[chosen_index]
                )

                if len(transaction) + len(corrupted_items) <= target_size:
                    transaction.update(corrupted_items)
                else:
                    if random.random() < 0.5:
                        transaction.update(corrupted_items)
                    else:
                        break

            transactions.append(transaction)

        return transactions

    def run(self, save: bool = False, path: str = "") -> tuple[List[Set], dict]:
        """
        :returns: transactions, taxonomy
        """
        taxonomy_forest, taxonomy_nodes = self.generate_taxonomy(
            NUM_ITEMS, NUM_ROOTS, NUM_LEVELS, FANOUT
        )
        self.assign_weights(taxonomy_forest, DEPTH_RATIO)

        print("Generate set Z of potentially frequent itemsets")
        Z, weights_Z = self.generate_Z(
            taxonomy_nodes, NUM_MAX_ITEMSETS, AVG_ITEMSET_SIZE, CORRELATION_LEVEL
        )

        print("Generate transactions")
        transactions = self.generate_transactions(
            Z, weights_Z, taxonomy_nodes, NUM_TRANSACTIONS, AVG_TRANSACTION_SIZE
        )
        if save:
            self._save_transactions_to_csv(
                transactions=transactions, path=path + "_transactions"
            )
            self._save_taxonomy_to_json(
                taxonomy_forest=taxonomy_forest, out_path=path + "_taxonomy"
            )
        return transactions, self.taxonomy_to_nested_dict(taxonomy_forest)


gen = SynteticGenerator()
gen.run(True, "data/syntetic_data_test")
