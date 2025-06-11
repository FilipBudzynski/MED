import json
import random
from collections import Counter, defaultdict
from itertools import combinations


class FastTaxonomyBuilder:
    def __init__(self, data_loader, num_roots=10, fanout=4):
        self.data_loader = data_loader
        self.num_roots = num_roots
        self.fanout = fanout

    def run(self, output_path="taxonomy.json"):
        transactions = self.data_loader.load()
        all_items = list(set(item for t in transactions for item in t))
        random.shuffle(all_items)

        split_items = [all_items[i :: self.num_roots] for i in range(self.num_roots)]
        forest = {}

        for i, item_chunk in enumerate(split_items):
            root_name = f"root_{i}"
            forest[root_name] = self._build_random_tree(item_chunk, prefix=root_name)

        with open(output_path, "w") as f:
            json.dump(forest, f, indent=2)

        return forest

    def _build_random_tree(self, items, prefix):
        """
        Recursively build a tree with only leaves as items.
        Internal nodes are synthetic and created randomly.
        """
        if len(items) <= self.fanout:
            return {f"{prefix}_leaf_{i}": item for i, item in enumerate(items)}

        num_groups = random.randint(2, self.fanout)
        random.shuffle(items)
        groups = [items[i::num_groups] for i in range(num_groups)]

        node = {}
        for i, group in enumerate(groups):
            child_prefix = f"{prefix}_{i}"
            node[child_prefix] = self._build_random_tree(group, child_prefix)
        return node


class FastHeuristicTaxonomyBuilder:
    def __init__(self, data_loader, num_roots=10, max_fanout=5):
        self.data_loader = data_loader
        self.num_roots = num_roots
        self.max_fanout = max_fanout
        self.transactions = self.data_loader.load()
        self.item_cooccur = defaultdict(lambda: defaultdict(int))
        self.taxonomy = {}
        self.items = set(item for t in self.transactions for item in t)

    def compute_cooccurrence(self):
        for t in self.transactions:
            for item in t:
                for other in t:
                    if item != other:
                        self.item_cooccur[item][other] += 1

    def group_items(self):
        unused_items = set(self.items)
        groups = []

        while unused_items and len(groups) < self.num_roots:
            root = f"root_{len(groups)}"
            group = [unused_items.pop()]
            candidates = list(unused_items)
            scores = defaultdict(int)

            for g_item in group:
                for c_item in candidates:
                    scores[c_item] += self.item_cooccur[g_item].get(c_item, 0)

            sorted_candidates = sorted(candidates, key=lambda x: -scores[x])
            for c in sorted_candidates[: self.max_fanout - 1]:
                group.append(c)
                unused_items.discard(c)

            groups.append((root, group))

        remaining_items = list(unused_items)
        for item in remaining_items:
            target_group = random.choice(groups)
            target_group[1].append(item)

        return groups

    def build_random_tree(self, parent_name, items):
        if len(items) <= self.max_fanout:
            return {f"{parent_name}_{i}": item for i, item in enumerate(items)}

        random.shuffle(items)
        children = {}
        fanout = random.randint(2, self.max_fanout)
        chunk_size = max(1, len(items) // fanout)

        for i in range(fanout):
            chunk = items[i * chunk_size : (i + 1) * chunk_size]
            if not chunk:
                continue
            node_name = f"{parent_name}_{i}"
            subtree = self.build_random_tree(node_name, chunk)
            children[node_name] = subtree

        return children

    def run(self, output_path="taxonomy.json"):
        print("Computing co-occurrence...")
        self.compute_cooccurrence()

        print("Grouping items...")
        groups = self.group_items()

        print("Building taxonomy tree...")
        for root_name, group_items in groups:
            self.taxonomy[root_name] = self.build_random_tree(root_name, group_items)

        print(f"Saving taxonomy to {output_path}")
        with open(output_path, "w") as f:
            json.dump(self.taxonomy, f, indent=2)

        return self.taxonomy
