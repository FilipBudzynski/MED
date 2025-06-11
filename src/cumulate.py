from typing import Dict, List, Set, Union
from itertools import combinations, chain
from collections import defaultdict

Nested = Dict[str, Union[Set[str], "Nested"]]


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end = False


def build_trie(itemsets):
    root = TrieNode()
    for itemset in itemsets:
        current = root
        for item in sorted(itemset):
            current = current.children[item]
        current.is_end = True
    return root


class Cumulate:
    def __init__(
        self,
        transactions: List[Set[str]],
        taxonomy: Dict[str, Set[str]],
        min_support: int,
        min_confidence: float,
        min_interest: float = 1.0,
    ):
        self.transactions = transactions
        self.taxonomy = taxonomy
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_interest = min_interest

        self.T_star = {}
        self._precompute_ancestors()
        self.frequent_itemsets = []
        self.support_counts = {}
        self.rules = []

    def _precompute_ancestors(self):  # optymalizacja 2
        for item in self._all_items():
            self.T_star[item] = self._get_ancestors(item)

    def _all_items(self) -> Set[str]:
        items = set(self.taxonomy.keys())
        for parents in self.taxonomy.values():
            items.update(parents)
        return items

    def _get_ancestors(self, item: str) -> Set[str]:
        visited = set()
        stack = list(self.taxonomy.get(item, []))
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(self.taxonomy.get(current, []))
        return visited

    def _generate_candidates(
        self, prev_frequent_itemsets: List[frozenset], k: int
    ) -> Set[frozenset]:
        candidates = set()
        prev_frequent_itemsets = sorted(
            [sorted(list(itemset)) for itemset in prev_frequent_itemsets]
        )
        prev_frequent_set = set(
            frozenset(itemset) for itemset in prev_frequent_itemsets
        )

        for i in range(len(prev_frequent_itemsets)):
            for j in range(i + 1, len(prev_frequent_itemsets)):
                # Join step: check if first k-2 items are the same
                if (
                    prev_frequent_itemsets[i][: k - 2]
                    == prev_frequent_itemsets[j][: k - 2]
                ):
                    # Candidate is union of two sets
                    candidate = frozenset(
                        sorted(
                            set(prev_frequent_itemsets[i])
                            | set(prev_frequent_itemsets[j])
                        )
                    )
                    if len(candidate) == k:
                        # Prune step: all (k-1) subsets must be in prev_frequent_set
                        all_subsets_frequent = True
                        for subset in combinations(candidate, k - 1):
                            if frozenset(subset) not in prev_frequent_set:
                                all_subsets_frequent = False
                                break
                        if all_subsets_frequent:
                            candidates.add(candidate)
                else:
                    # Because sorted, no need to continue inner loop if prefix mismatch
                    break
        return candidates

    def _filter_candidates_with_ancestors(
        self, candidates: Set[frozenset]
    ) -> Set[frozenset]:
        fitlered = set()
        for cand in candidates:
            contains_ancestors = False
            for item in cand:
                ancestors = self.T_star.get(item, set())
                if ancestors & cand:
                    contains_ancestors = True
                    break
            if not contains_ancestors:
                fitlered.add(cand)
        return fitlered

    def _prune_T_star(self, Ck: Set[frozenset]):
        candidate_items = set()
        for c in Ck:
            candidate_items.update(c)

        new_T_star = {}
        for item, ancestors in self.T_star.items():
            pruned_ancestors = {a for a in ancestors if a in candidate_items}
            new_T_star[item] = pruned_ancestors
        return new_T_star

    def run(self) -> List[Set[frozenset]]:
        item_counts = defaultdict(int)
        for t in self.transactions:
            extended_t = set()
            for item in t:
                extended_t.add(item)
                extended_t.update(self.T_star.get(item, set()))
            for item in extended_t:
                item_counts[frozenset([item])] += 1

        L = []
        L1 = {item for item, count in item_counts.items() if count >= self.min_support}
        L.append(L1)

        for item in L1:
            self.support_counts[item] = item_counts[item]

        k = 2
        while True:
            prev_L = list(L[-1])
            Ck = self._generate_candidates(prev_L, k)

            if not Ck:
                break

            if k == 2:
                Ck = self._filter_candidates_with_ancestors(Ck)

            self.T_star = self._prune_T_star(Ck)

            counts = defaultdict(int)
            for t in self.transactions:
                extended_t = set()
                for item in t:
                    ancestors = self.T_star.get(item, set())
                    extended_t.add(item)
                    extended_t.update(ancestors)

                for cand in Ck:
                    if cand.issubset(extended_t):
                        counts[cand] += 1

            Lk = {cand for cand, count in counts.items() if count >= self.min_support}

            for cand in Lk:
                self.support_counts[cand] = counts[cand]

            if not Lk:
                break

            L.append(Lk)
            k += 1

        self.frequent_itemsets = L
        return L

    def _filter_by_interest(self):
        filtered_rules = []
        rule_index = {
            (frozenset(r["antecedent"]), frozenset(r["consequent"])): r
            for r in self.rules
        }

        for rule in self.rules:
            A, B = rule["antecedent"], rule["consequent"]
            A_parents = set(chain.from_iterable(self.T_star.get(a, []) for a in A))
            B_parents = set(chain.from_iterable(self.T_star.get(b, []) for b in B))

            found_better_ancestor = False
            for Ap in _powerset(A_parents):
                for Bp in _powerset(B_parents):
                    if not Ap or not Bp:
                        continue
                    key = (frozenset(Ap), frozenset(Bp))
                    if key in rule_index:
                        ancestor_rule = rule_index[key]
                        expected_conf = ancestor_rule["confidence"]
                        if expected_conf > 0:
                            ratio = rule["confidence"] / expected_conf
                            if ratio < self.min_interest:
                                found_better_ancestor = True
            if not found_better_ancestor:
                filtered_rules.append(rule)

        self.rules = filtered_rules


def _powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def compute_ancestors_from_tree(tree) -> Dict[str, Set[str]]:
    """
    Given a nested‐dict taxonomy:
      {
        "clothes": {
          "outerwear": {"jacket", "ski_pants"},
          "shirt": {}
        },
        "footwear": {"shoes", "hiking_boots"}
      }
    Returns T_star: each item → set of *all* its ancestors.
    """
    T_star: Dict[str, Set[str]] = {}

    def visit(node: str, subtree: Union[Set[str], Nested], ancestors: Set[str]):
        T_star.setdefault(node, set()).update(ancestors)

        if isinstance(subtree, dict):
            for child, child_sub in subtree.items():
                visit(child, child_sub, ancestors | {node})
        else:
            for child in subtree:
                visit(child, {}, ancestors | {node})

    for root, sub in tree.items():
        visit(root, sub, set())

    return T_star
