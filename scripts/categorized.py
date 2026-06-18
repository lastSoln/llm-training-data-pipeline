import pandas as pd
import os


def classify(text):
    text = str(text).lower()

    if any(k in text for k in [
        "python", "java", "javascript",
        "sql", "programming", "code",
        "algorithm", "software"
    ]):
        return "coding"

    elif any(k in text for k in [
        "physics", "chemistry",
        "biology", "science"
    ]):
        return "science"

    elif any(k in text for k in [
        "history", "war",
        "empire", "ancient",
        "kingdom"
    ]):
        return "history"

    elif any(k in text for k in [
        "math", "algebra",
        "geometry", "calculus"
    ]):
        return "mathematics"

    elif any(k in text for k in [
        "business", "finance",
        "marketing", "economics"
    ]):
        return "business"

    elif any(k in text for k in [
        "health", "medical",
        "doctor", "disease"
    ]):
        return "health"

    elif any(k in text for k in [
        "technology",
        "computer",
        "artificial intelligence",
        "machine learning",
        "ai"
    ]):
        return "technology"

    elif any(k in text for k in [
        "education",
        "student",
        "teacher",
        "learning"
    ]):
        return "education"

    elif any(k in text for k in [
        "story",
        "character",
        "fiction",
        "novel"
    ]):
        return "creative_writing"

    elif any(k in text for k in [
        "joke",
        "funny",
        "humor"
    ]):
        return "humor"

    return "general_assistance"


print("Loading training corpus...")

df = pd.read_csv(
    "processed/final_training_corpus.csv"
)

print("Rows loaded:", len(df))

# Create category column
df["final_category"] = df["input"].apply(classify)

# Preserve known dataset categories
df.loc[
    df["category"] == "conversation",
    "final_category"
] = "conversation"

df.loc[
    df["category"] == "summarization",
    "final_category"
] = "summarization"

df.loc[
    df["category"] == "qa",
    "final_category"
] = "general_assistance"

print("\nCategory Distribution:")
print(df["final_category"].value_counts())

# Save updated master file
df.to_csv(
    "processed/final_training_corpus.csv",
    index=False
)

print("\nUpdated master corpus saved.")

# Create categorized folder
os.makedirs(
    "categorized",
    exist_ok=True
)

# Export category files
for category in df["final_category"].unique():

    subset = df[
        df["final_category"] == category
    ]

    filename = (
        f"categorized/{category}.csv"
    )

    subset.to_csv(
        filename,
        index=False
    )

    print(
        f"Saved {filename} "
        f"({len(subset)} rows)"
    )

print("\nAll category files created successfully.")