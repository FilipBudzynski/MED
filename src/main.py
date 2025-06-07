from runner import Runner
from data import data
from cumulate import Cumulate
from loaders.loader import BinaryDataLoader

if __name__ == "__main__":
    market_data = data["market"]

    runner = Runner(
        market_data["taxonomy"],
        Cumulate,
        BinaryDataLoader(market_data["file_path"]),
        300,
        0.90,
    )

    runner.mine_frequent_itemsets(log=True)
    runner.mine_assosiation_rules(log=True)
