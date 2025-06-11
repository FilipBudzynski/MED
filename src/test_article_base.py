from runner import Runner
from data import data
from cumulate import Cumulate
from loaders.loader import BinaryDataLoader
import unittest

if __name__ == "__main__":
    article_data = data["article"]

    runner = Runner(
        taxonomy=article_data["taxonomy"],
        miner=Cumulate,
        data_loader=BinaryDataLoader(article_data["file_path"]),
        min_support=0.33,
        min_confidence=0.60,
    )

    frequent_itemsets = runner.mine_frequent_itemsets(log=True)
    rules = runner.mine_assosiation_rules(log=True)

