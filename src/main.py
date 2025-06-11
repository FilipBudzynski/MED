from runner import Runner
from data import data
from cumulate import Cumulate
from loaders.loader import BinaryDataLoader

if __name__ == "__main__":

    market_data = data["market"]
    article_data = data["article"]

    runner = Runner(
        taxonomy=market_data["taxonomy"],
        miner=Cumulate,
        data_loader=BinaryDataLoader(market_data["file_path"]),
        min_support=0.64,
        min_confidence=0.90,
    )
    runner.mine_frequent_itemsets(log=True)
    runner.mine_assosiation_rules(log=True)
