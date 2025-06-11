import numpy as np
import random
from typing import Set, List
from math import exp
import json
import csv

np.random.seed(124)
random.seed(124)


class TaxonomyNode:
    def __init__(self, item_id, level):
        self.item_id = item_id
        self.children = []
        self.level = level
        self.weight = 0.0


class SynteticGenerator:
    def __init__(
        self,
        num_items=1000,
        num_roots=25,
        num_levels=4,
        fanout=5,
        depth_ratio=1.0,
        num_transactions=10000,
        avg_transaction_size=7,
        avg_itemset_size=3,
        num_max_itemsets=100,
        correlation_level=0.5,
    ) -> None:
        self.num_items = num_items
        self.num_roots = num_roots
        self.num_levels = num_levels
        self.fanout = fanout
        self.depth_ratio = depth_ratio

        self.num_transactions = num_transactions
        self.avg_transaction_size = avg_transaction_size
        self.avg_itemset_size = avg_itemset_size
        self.num_max_itemsets = num_max_itemsets
        self.correlation_level = correlation_level

    def _save_transactions_to_csv(self, transactions: List[Set[int]], path: str):
        with open(path + ".csv", "w", newline="") as f:
            writer = csv.writer(f)
            for t in transactions:
                writer.writerow(list(t))

    def _save_taxonomy_to_json(self, taxonomy_forest, out_path: str):
        taxonomy_dict = self.taxonomy_to_nested_dict(taxonomy_forest)

        with open(out_path + ".json", "w") as f:
            json.dump(taxonomy_dict, f, indent=2)

    def generate_taxonomy(self, N, R, L, F):
        item_counter = 0
        forest = []

        def create_subtree(level):
            nonlocal item_counter
            if item_counter >= N or level >= L:
                return None
            node_id = item_counter
            item_counter += 1
            node = TaxonomyNode(node_id, level)
            # Poisson-distributed children
            num_children = np.random.poisson(F)
            for _ in range(num_children):
                if item_counter < N:
                    child = create_subtree(level + 1)
                    if child:
                        node.children.append(child)
            return node

        for _ in range(R):
            if item_counter < N:
                root = create_subtree(0)
                if root:
                    forest.append(root)
        return forest

    def build_all_nodes_dict(self, forest):
        all_nodes = {}

        def traverse(node):
            all_nodes[node.item_id] = node
            for child in node.children:
                traverse(child)

        for root in forest:
            traverse(root)
        return all_nodes

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

    def taxonomy_to_nested_dict(self, forest):
        def node_to_dict(node):
            if not node.children:
                return {}
            return {child.item_id: node_to_dict(child) for child in node.children}

        return {root.item_id: node_to_dict(root) for root in forest}

    def get_leaves(self, nodes_dict):
        return [node for node in nodes_dict.values() if not node.children]

    def flatten_taxonomy(self, forest):
        all_nodes = {}

        def dfs(node):
            all_nodes[node.item_id] = node
            for c in node.children:
                dfs(c)

        for root in forest:
            dfs(root)
        return all_nodes

    def weighted_choice(self, items, weights):
        total = sum(weights)
        rnd = random.uniform(0, total)
        upto = 0
        for item, w in zip(items, weights):
            if upto + w >= rnd:
                return item
            upto += w
        return items[-1]

    def build_weight_distribution(self, all_nodes):
        items = list(all_nodes.keys())
        weights = [all_nodes[i].weight for i in items]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        return items, probs

    def specialize_to_leaf(self, node: TaxonomyNode):
        current = node
        while current.children:
            children = current.children
            weights = [child.weight for child in children]
            current = self.weighted_choice(children, weights)
        return current.item_id

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

    def corrupt_itemset(self, itemset: Set[int], corruption_level: float) -> Set[int]:
        retained = [item for item in itemset if random.random() >= corruption_level]
        if not retained and len(itemset) > 0:
            retained = [random.choice(list(itemset))]
        return set(retained)

    def generate_transactions(
        self, Z, weights_Z, all_nodes, num_transactions, avg_transaction_size
    ):
        num_Z = len(Z)
        corruption_levels = np.clip(np.random.normal(0.5, 0.1, num_Z), 0.0, 1.0)

        transactions = []

        for t in range(num_transactions):
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
        taxonomy_forest = self.generate_taxonomy(
            self.num_items, self.num_roots, self.num_levels, self.fanout
        )
        all_nodes = self.build_all_nodes_dict(taxonomy_forest)

        self.assign_weights(taxonomy_forest, self.depth_ratio)

        Z, weights_Z = self.generate_Z(
            all_nodes, self.num_max_itemsets, self.avg_itemset_size, self.correlation_level
        )

        transactions = self.generate_transactions(
            Z, weights_Z, all_nodes, self.num_transactions, self.avg_transaction_size
        )
        if save:
            self._save_transactions_to_csv(
                transactions=transactions, path=path + "_transactions"
            )
            self._save_taxonomy_to_json(
                taxonomy_forest=taxonomy_forest, out_path=path + "_taxonomy"
            )
        return transactions, self.taxonomy_to_nested_dict(taxonomy_forest)

