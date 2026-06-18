import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

print("Loading model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Domain descriptions
categories = {
    "coding":
        "computer programming software development coding python java sql algorithms web development",

    "science":
        "physics chemistry biology scientific research experiments natural science",

    "history":
        "historical events civilizations wars empires ancient history world history",

    "mathematics":
        "mathematics algebra geometry calculus statistics equations mathematical reasoning",

    "business":
        "finance economics accounting marketing management entrepreneurship business strategy",

    "health":
        "medicine healthcare diseases treatment doctors hospitals wellness nutrition",

    "technology":
        "technology artificial intelligence machine learning computers digital systems innovation",

    "education":
        "learning teaching schools universities students education academic concepts",

    "creative_writing":
        "stories fiction novels characters storytelling creative writing literature",

    "humor":
        "jokes funny comedy humor entertainment",

    "general_assistance":
        "general questions explanations advice information assistance"
}

print("Creating category embeddings...")

category_names = list(categories.keys())

category_embeddings = model.encode(
    list(categories.values()),
    convert_to_tensor=True,
    normalize_embeddings=True
)

print("Loading dataset...")

df = pd.read_csv(
    "processed/final_training_corpus.csv"
)

print("Rows:", len(df))

# Preserve categories already known
df["final_category"] = None

df.loc[
    df["category"] == "conversation",
    "final_category"
] = "conversation"

df.loc[
    df["category"] == "summarization",
    "final_category"
] = "summarization"

# Only classify remaining rows
mask = df["final_category"].isna()

texts = (
    df.loc[mask, "input"]
    .fillna("")
    .astype(str)
    .tolist()
)

indices = df.loc[mask].index

print(
    f"Semantic classification needed for {len(texts):,} rows"
)

BATCH_SIZE = 2048

predictions = []

for start in tqdm(
    range(0, len(texts), BATCH_SIZE),
    desc="Processing batches"
):

    batch = texts[start:start+BATCH_SIZE]

    embeddings = model.encode(
        batch,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarities = util.cos_sim(
        embeddings,
        category_embeddings
    )

    best_matches = similarities.argmax(
        dim=1
    ).cpu().numpy()

    batch_categories = [
        category_names[i]
        for i in best_matches
    ]

    predictions.extend(
        batch_categories
    )

# Insert predictions back
df.loc[
    indices,
    "final_category"
] = predictions

print("\nCategory Distribution:")
print(
    df["final_category"]
    .value_counts()
)

# Save master file
df.to_csv(
    "processed/final_training_corpus_semantic.csv",
    index=False
)

print(
    "\nSaved final_training_corpus_semantic.csv"
)

# Create category files
os.makedirs(
    "categorized_semantic",
    exist_ok=True
)

for category in df["final_category"].unique():

    subset = df[
        df["final_category"] == category
    ]

    filepath = (
        f"categorized_semantic/{category}.csv"
    )

    subset.to_csv(
        filepath,
        index=False
    )

    print(
        f"Saved {filepath} "
        f"({len(subset):,} rows)"
    )

print("\nDone.")