from datasets import load_dataset

boolq = load_dataset("google/boolq")

print(boolq)
print(boolq["train"].num_rows)
print(boolq["train"].column_names)