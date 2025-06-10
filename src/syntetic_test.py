from runner import Runner
from cumulate import Cumulate
from generators.syntetic_data_gen import SynteticGenerator

if __name__ == "__main__":

    transaction_gen = SynteticGenerator()
    transactions, taxonomy = transaction_gen.run()

    runner = Runner(
        taxonomy=taxonomy,
        miner=Cumulate,
        transactions=transactions,
        data_loader=None,
        min_support=0.05,
        min_confidence=0.01,
    )
    runner.mine_frequent_itemsets(log=True)
    runner.mine_assosiation_rules(log=True)
