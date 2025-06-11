from runner import Runner
from cumulate import Cumulate
from loaders.loader import SimpleSeparatedDataLoader
import json
import time
import matplotlib.pyplot as plt
from generators.syntetic_data_gen_two import SynteticGenerator
import random
import numpy as np

np.random.seed(42)
random.seed(42)


def fanout_experiments():
    fanouts = [2.5, 5, 7.5, 10, 15, 20, 25]
    execution_times = []

    for fanout in fanouts:
        print(f"\nRunning experiment for FANOUT={fanout}")
        generator = SynteticGenerator(
            fanout=fanout,
        )

        transactions, taxonomy = generator.run(save=False)
        runner = Runner(
            miner=Cumulate,
            transactions=transactions,
            taxonomy=taxonomy,
            min_support=0.01,
            min_confidence=0.01,
        )

        start_time = time.time()
        runner.mine_frequent_itemsets()
        runner.mine_assosiation_rules()
        end_time = time.time()
        duration = end_time - start_time
        execution_times.append(duration)

    plt.figure(figsize=(10, 6))
    plt.plot(fanouts, execution_times, marker="o", color="teal")
    plt.xlabel("Fanout Value")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs. FANOUT in Synthetic Dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def roots_experiments():
    roots = [25, 33, 50, 75]
    execution_times = []

    for n_root in roots:
        print(f"\nRunning experiment for Rots={n_root}")
        generator = SynteticGenerator(
            num_roots=n_root,
        )

        transactions, taxonomy = generator.run(save=False)
        runner = Runner(
            miner=Cumulate,
            transactions=transactions,
            taxonomy=taxonomy,
            min_support=0.01,
            min_confidence=0.01,
        )

        start_time = time.time()
        runner.mine_frequent_itemsets()
        runner.mine_assosiation_rules()
        end_time = time.time()
        duration = end_time - start_time
        execution_times.append(duration)

    plt.figure(figsize=(10, 6))
    plt.plot(roots, execution_times, marker="o", color="teal")
    plt.xlabel("Num roots Value")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs. ROOTS NUM in Synthetic Dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def num_items_experiments():
    items = [500, 1000, 1500, 2000, 2500, 5000]
    execution_times = []

    for item in items:
        print(f"\nRunning experiment for items={item}")
        generator = SynteticGenerator(
            num_items=item,
        )

        transactions, taxonomy = generator.run(save=False)
        runner = Runner(
            miner=Cumulate,
            transactions=transactions,
            taxonomy=taxonomy,
            min_support=0.01,
            min_confidence=0.01,
        )

        start_time = time.time()
        runner.mine_frequent_itemsets()
        runner.mine_assosiation_rules()
        end_time = time.time()
        duration = end_time - start_time
        execution_times.append(duration)

    plt.figure(figsize=(10, 6))
    plt.plot(items, execution_times, marker="o", color="teal")
    plt.xlabel("Num items Value")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs. ITEMS NUM in Synthetic Dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def depth_experiments():
    depths = [0.5, 0.75, 1, 1.5, 2]
    execution_times = []

    for depth in depths:
        print(f"\nRunning experiment for depth={depth}")
        generator = SynteticGenerator(
            depth_ratio=depth
        )

        transactions, taxonomy = generator.run(save=False)
        runner = Runner(
            miner=Cumulate,
            transactions=transactions,
            taxonomy=taxonomy,
            min_support=0.0005,
            min_confidence=0.001,
        )

        start_time = time.time()
        runner.mine_frequent_itemsets()
        runner.mine_assosiation_rules()
        end_time = time.time()
        duration = end_time - start_time
        execution_times.append(duration)

    plt.figure(figsize=(10, 6))
    plt.plot(depths, execution_times, marker="o", color="teal")
    plt.xlabel("Num items Value")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs. DEPTH RATIO in Synthetic Dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def minsupport_experiments():
    supports = [0.002, 0.0015, 0.001, 0.00075, 0.0005, 0.00033]
    execution_times = []

    for min_sup in supports:
        print(f"\nRunning experiment for min_sup = {min_sup}")
        generator = SynteticGenerator()

        transactions, taxonomy = generator.run(save=False)
        runner = Runner(
            miner=Cumulate,
            transactions=transactions,
            taxonomy=taxonomy,
            min_support=min_sup,
            min_confidence=0.001,
        )

        start_time = time.time()
        runner.mine_frequent_itemsets()
        runner.mine_assosiation_rules()
        end_time = time.time()
        duration = end_time - start_time
        execution_times.append(duration)

    plt.figure(figsize=(10, 6))
    plt.plot(supports, execution_times, marker="o", color="teal")
    plt.xlabel("Num items Value")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time and MIN_SUPPORT in Synthetic Dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # fanout_experiments()
    #roots_experiments()
    # num_items_experiments()
    # depth_experiments()
    minsupport_experiments()
