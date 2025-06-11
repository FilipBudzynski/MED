from runner import Runner
from data import data
from cumulate import Cumulate
from loaders.loader import BinaryDataLoader, SimpleSeparatedDataLoader
from collections import Counter
import time
import matplotlib.pyplot as plt

market_data = data["market"]

runner = Runner(
    taxonomy=market_data["taxonomy"],
    miner=Cumulate,
    data_loader=BinaryDataLoader(market_data["file_path"]),
    min_support=0.64,
    min_confidence=0.90,
)
runner.mine_frequent_itemsets(log=True)
runner.mine_assosiation_rules(log=True)

def market_minsupp_experiment():
    minsups = [0.75, 0.5, 0.25, 0.1, 0.075]
    execution_times = []

    for minsup in minsups:
        runner = Runner(
            miner=Cumulate,
            data_loader=BinaryDataLoader(market_data["file_path"]),
            taxonomy=market_data["taxonomy"],
            min_support=minsup,
            min_confidence=0.01,
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
    plt.title("Execution Time vs. MIN SUPPORT in Market Real Dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.gca().invert_xaxis()  
    plt.show()



