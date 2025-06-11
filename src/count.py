from loaders.loader import SimpleSeparatedDataLoader

transactions = SimpleSeparatedDataLoader("data/syntetic_data_test_article_transactions.csv").load()

num_transactions = len(transactions)

all_products = set().union(*transactions)
num_products = len(all_products)

avg_transaction_length = sum(len(t) for t in transactions) / num_transactions if num_transactions else 0

total_products_bought = sum(len(t) for t in transactions)
density = total_products_bought / (num_transactions * num_products) if num_transactions and num_products else 0

print(f"Ilość transakcji: {num_transactions}")
print(f"Ilość produktów: {num_products}")
print(f"Średnia długość transakcji: {avg_transaction_length:.2f}")
print(f"Gęstość: {density:.4f}")
