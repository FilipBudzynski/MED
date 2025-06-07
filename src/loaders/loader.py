from abc import ABC, abstractmethod
from typing import List, Set


class DataLoader(ABC):
    def __init__(self, data_path):
        self.data_path = data_path

    @abstractmethod
    def load(self) -> List[Set]:
        """
        Load data from the data_path.
        Must be implemented by subclasses.
        """
        pass


class BinaryDataLoader(DataLoader):
    """
    Concrete DataLoader class to load transaction data in binary form
    data form example:

    Shirt;Jacket;Shoes
    0;1;1
    1;0;0
    1;1;1
    """

    def __init__(self, data_path):
        super().__init__(data_path)

    def load(self):
        transactions = []
        with open(self.data_path, "r") as f:
            lines = f.readlines()
            items = lines[0].strip().split(";")

            for line in lines[1:]:
                values = line.strip().split(";")
                transaction = {items[i] for i, val in enumerate(values) if val == "1"}
                transactions.append(transaction)

        return transactions
