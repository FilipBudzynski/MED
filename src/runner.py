from loaders.loader import DataLoader
from collections import defaultdict
from itertools import combinations
from typing import Optional


class Runner:
    """
    Constructs new runner object to load transactions and initiate data mining algorithms

    :param min_support: [0 - 1] float -- minimal support
    :param min_confidence: [0 - 1] float -- minimal confidence
    """

    def __init__(
        self,
        taxonomy,
        miner,
        data_loader: Optional[DataLoader],
        transactions,
        min_support: float,
        min_confidence: float,
    ):
        self.taxonomy = taxonomy
        self.data_loader = data_loader
        self.transactions = transactions or self.load()
        self.min_support = min_support * len(self.transactions)
        self.miner = miner(
            self.transactions,
            taxonomy,
            min_support * len(self.transactions),
            min_confidence,
        )

    def load(self):
        transactions = self.data_loader.load()
        return transactions

    def mine_frequent_itemsets(self, log: bool = False):
        frequent_itemsets = self.miner.run()
        self.frequent_itemsets = frequent_itemsets

        if log:
            self._log_frequent_items()
        return frequent_itemsets

    def mine_assosiation_rules(self, log: bool = False):
        rules = self._mine_generalized_rules()
        self.rules = rules

        if log:
            self._log_generalized_rules()
        return rules

    def _mine_generalized_rules(self):
        rules = []
        support_counts = defaultdict(int)

        extended_transactions = []
        for t in self.transactions:
            extended = set()
            for item in t:
                extended.add(item)
                extended.update(self.miner.T_star.get(item, set()))
            extended_transactions.append(extended)

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
                        if confidence >= self.miner.min_confidence:
                            rules.append(
                                {
                                    "antecedent": sorted(antecedent),
                                    "consequent": sorted(consequent),
                                    "confidence": confidence,
                                    "support": support_itemset / num_transactions,
                                }
                            )
        return rules

    def _log_frequent_items(self):
        print("Frequent Itemsets:")
        for level in self.frequent_itemsets:
            for itemset in level:
                support = self.miner.support_counts.get(itemset, 0)
                print(f"{set(itemset)} (support: {support})")

    def _log_generalized_rules(self):
        print("\nGenerated Rules:")
        for rule in self.rules:
            print(
                f"{set(rule['antecedent'])} => {set(rule['consequent'])}, "
                f"support: {rule['support']:.2f}, conf: {rule['confidence']:.2f}"
            )
