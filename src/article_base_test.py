from runner import Runner
from data import data
from cumulate import Cumulate
from loaders.loader import BinaryDataLoader

if __name__ == "__main__":

    article_data = data["article"]

    runner = Runner(
        article_data["taxonomy"],
        Cumulate,
        BinaryDataLoader(article_data["file_path"]),
        2,
        0.60,
    )

    runner.mine_frequent_itemsets(log=True)
    runner.mine_assosiation_rules(log=True)
