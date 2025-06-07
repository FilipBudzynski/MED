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

    def generate_candidates(
        self, prev_frequent_itemsets: List[frozenset], k: int
    ) -> Set[frozenset]:
        candidates = set()
        L_prev = list(prev_frequent_itemsets)
        for i in range(len(L_prev)):
            for j in range(i + 1, len(L_prev)):
                l1 = sorted(L_prev[i])
                l2 = sorted(L_prev[j])
                if l1[:-1] == l2[:-1]:
                    union = frozenset(set(l1) | set(l2))
                    if len(union) == k:
                        all_subsets_frequent = all(
                            frozenset(subset) in prev_frequent_itemsets
                            for subset in combinations(union, k - 1)
                        )
                        if all_subsets_frequent:
                            candidates.add(union)
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
            Ck = self.generate_candidates(prev_L, k)

            if not Ck:
                break

            if k == 2:
                Ck = self._filter_candidates_with_ancestors(Ck)

            # Optymalizacja: usuwamy przodków z T* którzy nie są obecni w C_k
            self.T_star = self._prune_T_star(Ck)

            # Obliczamy wsparcie (support)
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

    def _generate_rules(self):
        support_counts = defaultdict(int)

        # Build extended transactions once
        extended_transactions = []
        for t in self.transactions:
            extended = set()
            for item in t:
                extended.add(item)
                extended.update(self.T_star.get(item, set()))
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
