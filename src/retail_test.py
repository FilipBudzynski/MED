from generators.taxonomy_gen import FastTaxonomyBuilder, FastHeuristicTaxonomyBuilder
from data import data
from loaders.loader import SimpleSeparatedDataLoader
from cumulate import Cumulate
from runner import Runner
import time
import matplotlib.pyplot as plt

retail_path = data["retail"]["file_path"]

data_loader = SimpleSeparatedDataLoader(retail_path, separator=" ")
transactions = data_loader.load()

taxonomy_builder = FastHeuristicTaxonomyBuilder(
    data_loader, max_fanout=10, num_roots=100
)
taxonomy = taxonomy_builder.run(output_path="data/retail_taxonomy.json")


runner = Runner(
    taxonomy=taxonomy,
    miner=Cumulate,
    transactions=transactions,
    min_support=0.1,
    min_confidence=0.05,
)
runner.mine_frequent_itemsets(log=True)
runner.mine_assosiation_rules(log=True)


def retail_minsupp_experiment():
    minsups = [0.1, 0.075, 0.05, 0.025, 0.01]
    execution_times = []

    for minsup in minsups:
        runner = Runner(
            taxonomy=taxonomy,
            miner=Cumulate,
            transactions=transactions,
            min_support=minsup,
            min_confidence=0.05,
        )

        start_time = time.time()
        runner.mine_frequent_itemsets()
        runner.mine_assosiation_rules()
        end_time = time.time()
        duration = end_time - start_time
        execution_times.append(duration)

    plt.figure(figsize=(10, 6))
    plt.plot(minsups, execution_times, marker="o", color="teal")
    plt.xlabel("Minsup Value")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs. MIN SUPPORT in Retail Real Dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.show()

