import pandas as pd

labelled = pd.read_csv(
    "processed/final_training_corpus.csv"
)

unlabelled = pd.read_csv(
    "processed/final_unlabelled_corpus.csv"
)

print("Labelled:", labelled.shape)
print("Unlabelled:", unlabelled.shape)

total_rows = len(labelled) + len(unlabelled)

print("\nTotal Records:", total_rows)