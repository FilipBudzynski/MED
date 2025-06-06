from typing import Dict, List, Set
from itertools import combinations, chain
from collections import defaultdict


class Cumulate:
    def __init__(
        self,
        transactions: List[Set[str]],
        taxonomy: Dict[str, Set[str]],
        min_support: int,
        min_confidence: float,
        min_interest: float,
    ):
        self.transactions = transactions
        self.taxonomy = taxonomy
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_interest = min_interest

        self.T_star = {}
        self._precompute_ancestors()
        self.frequent_itemsets = []
        self.rules = []

    def _precompute_ancestors(self):  # optymalizacja 2
        for item in self._all_items():
            self.T_star[item] = self._get_ancestors(item)

        for item, ancestors in self.T_star.items():
            print(f"{item} -> {ancestors}")

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
        visited.add(item)
        return visited

    def run(self) -> List[Set[frozenset]]:
        item_counts = defaultdict(int)
        for t in self.transactions:
            extended_t = set()
            for item in t:
                extended_t.update(self.T_star.get(item, {item}))
            for item in extended_t:
                item_counts[frozenset([item])] += 1

        L = []
        L1 = {item for item, count in item_counts.items() if count >= self.min_support}
        L.append(L1)

        k = 2
        while True:
            prev_L = list(L[-1])
            Ck = set()

            for i in range(len(prev_L)):
                for j in range(i + 1, len(prev_L)):
                    union = prev_L[i] | prev_L[j]
                    if len(union) == k:
                        if self._contains_item_and_ancestor(union):
                            continue
                        Ck.add(union)

            if not Ck:
                break

            # Optymalizacja: usuwamy przodków, którzy nie mają żadnych kandydatów
            candidate_items = set()
            for cand in Ck:
                candidate_items.update(cand)

            filtered_T_star = {
                item: set(ancestors) & (candidate_items)
                for item, ancestors in self.T_star.items()
            }

            # Obliczamy wsparcie (support)
            counts = defaultdict(int)
            for t in self.transactions:
                extended_t = set()
                for item in t:
                    ancestors = filtered_T_star.get(item, set())
                    needed = ancestors & candidate_items
                    if item in candidate_items:
                        needed.add(item)
                    extended_t.update(needed)
                for cand in Ck:
                    if cand.issubset(extended_t):
                        counts[cand] += 1

            Lk = {cand for cand, count in counts.items() if count >= self.min_support}
            if not Lk:
                break

            L.append(Lk)
            k += 1

        self.frequent_itemsets = L
        return L

    def _contains_item_and_ancestor(self, itemset: frozenset) -> bool:
        for item in itemset:
            ancestors = self.T_star.get(item, set())
            if set(ancestors) & (itemset - {item}):
                return True
        return False
        # items = list(itemset)
        # for i in range(len(items)):
        #     for j in range(len(items)):
        #         if i != j and items[i] in self.T_star.get(items[j], set()):
        #             return True
        # return False

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

    def _generate_rules(self):
        support_counts = defaultdict(int)

        # Build extended transactions once
        extended_transactions = []
        for t in self.transactions:
            extended = set()
            for item in t:
                extended.update(self.T_star.get(item, [item]))
            extended_transactions.append(extended)

        # Count support for all frequent itemsets
        for level in self.frequent_itemsets:
            for itemset in level:
                for t in extended_transactions:
                    if itemset.issubset(t):
                        support_counts[itemset] += 1

        num_transactions = len(self.transactions)

        for level in self.frequent_itemsets:
            for itemset in level:
                items = list(itemset)
                for i in range(1, len(items)):
                    for antecedent in combinations(items, i):
                        antecedent = frozenset(antecedent)
                        consequent = frozenset(itemset - antecedent)
                        if not consequent:
                            continue
                        support_itemset = support_counts[itemset]
                        support_antecedent = support_counts.get(antecedent, 0)
                        if support_antecedent == 0:
                            continue
                        confidence = support_itemset / support_antecedent
                        if confidence >= self.min_confidence:
                            self.rules.append(
                                {
                                    "antecedent": antecedent,
                                    "consequent": consequent,
                                    "confidence": confidence,
                                    "support": support_itemset / num_transactions,
                                }
                            )


def _powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
