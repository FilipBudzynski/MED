from runner import Runner
from data import data
from cumulate import Cumulate
from loaders.loader import BinaryDataLoader

market_data = data["market"]

runner = Runner(
    market_data["taxonomy"],
    Cumulate,
    BinaryDataLoader(market_data["file_path"]),
    200,
    0.90,
)

runner.mine_frequent_itemsets()
runner.mine_assosiation_rules()
