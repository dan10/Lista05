import pandas as pd
from apyori import apriori


# Questao 1,2 e 3
def who_takes(transac):
    # Apply the Apriori algorithm to find frequent itemsets
    results = list(apriori(transac, min_support=0.3, min_confidence=0.8))

    for r in results:
        itemset = ', '.join(list(r.items))
        label = f"Frequent itemset with support {r.support:.2f}:"
        print(f"{label} {itemset}")

    # Convert the results to a pandas DataFrame
    rules = []
    for record in results:
        for ordered_statistic in record.ordered_statistics:
            rule = {'antecedents': list(ordered_statistic.items_base),
                    'consequents': list(ordered_statistic.items_add),
                    'support': record.support,
                    'confidence': ordered_statistic.confidence,
                    'lift': ordered_statistic.lift}
            rules.append(rule)

    rules_df = pd.DataFrame(rules, columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

    # Print the association rules DataFrame
    print(rules_df)


# Questao 4
def who_not_takes(transac):
    # Create a set of all items
    items = set(item for tran in transac for item in tran)

    # Create new transactions marking each item as present or not present
    new_transactions = []
    for tran in transac:
        new_transaction = [item if item in tran else f"not_{item}" for item in items]
        new_transactions.append(new_transaction)

    # Apply the Apriori algorithm to find association rules
    rules = list(apriori(new_transactions, min_confidence=0.1))

    # Filter and print the right rules where the items_base contains at least one not_item
    for rule in rules:
        for ordered_stat in rule.ordered_statistics:
            if any(item.startswith("not_") for item in ordered_stat.items_base):
                print(f"{ordered_stat.items_base} => {ordered_stat.items_add} "
                      f"(support={rule.support:.3f}, confidence={ordered_stat.confidence:.3f}, "
                      f"lift={ordered_stat.lift:.3f})")


if __name__ == '__main__':
    df = pd.read_csv('MercadoSim.csv', sep=';', header=None)

    # Convert the DataFrame into a list of transactions
    transactions = []
    for i in range(len(df)):
        transaction = [str(df.values[i, j]) for j in range(len(df.columns)) if
                       str(df.values[i, j]) != 'nan']
        transactions.append(transaction)

    who_takes(transactions)
    print("Questao 4")
    who_not_takes(transactions)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
