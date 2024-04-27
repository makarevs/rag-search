import pandas as pd

"""
Check the correctness of read of the CSV with complex text
Assume that Title should be of limited length
"""

DATA_FILE = "data/medium.csv"
data = pd.read_csv(DATA_FILE)

data["Title len"] = data["Title"].apply(lambda x: len(x))

longest_titles = data.sort_values(by="Title len", ascending=False, axis=0)  # [["Title len", "Title"]]
print(longest_titles[["Title len", "Title"]].head())

print(data.head)

print(f"Maximum title length is {max(data['Title len'])}")

print()