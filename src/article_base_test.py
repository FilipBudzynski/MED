from cumulate import Cumulate

transactions = [
    {"shirt"},
    {"jacket", "hiking_boots"},
    {"ski_pants", "hiking_boots"},
    {"shoes"},
    {"shoes"},
    {"jacket"},
]

taxonomy = {
    "jacket": {"outerwear"},
    "ski_pants": {"outerwear"},
    "outerwear": {"clothes"},
    "shirt": {"clothes"},
    "hiking_boots": {"footwear"},
    "shoes": {"footwear"},
}

if __name__ == "__main__":

    cumulate = Cumulate(
        transactions=transactions,
        taxonomy=taxonomy,
        min_support=2,
        min_confidence=0.66,
        min_interest=1.0,
    )

    frequent_itemsets = cumulate.run()
    cumulate._generate_rules()

    print("Frequent Itemsets:")
    for level in frequent_itemsets:
        for itemset in level:
            support = cumulate.support_counts.get(itemset, 0)
            print(f"{set(itemset)} (support: {support})")

    print("\nGenerated Rules:")
    for rule in cumulate.rules:
        print(
            f"{set(rule['antecedent'])} => {set(rule['consequent'])}, "
            f"support: {rule['support']:.2f}, conf: {rule['confidence']:.2f}"
        )
