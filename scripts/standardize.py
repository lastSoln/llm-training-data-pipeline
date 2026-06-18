from datasets import load_dataset
import pandas as pd


def clean_text(text):
    if pd.isna(text):
        return ""

    return str(text).strip()


def build_alpaca():
    print("Loading Alpaca...")

    alpaca = load_dataset("tatsu-lab/alpaca")

    train = alpaca["train"]

    rows = []

    for instruction, extra_input, output in zip(
        train["instruction"],
        train["input"],
        train["output"]
    ):

        instruction = clean_text(instruction)
        extra_input = clean_text(extra_input)
        output = clean_text(output)

        if extra_input:
            final_input = (
                instruction +
                "\n\nContext:\n" +
                extra_input
            )
        else:
            final_input = instruction

        rows.append({
            "source": "alpaca",
            "category": "instruction",
            "input": final_input,
            "output": output
        })
        
    df = pd.DataFrame(rows)

    df.dropna(inplace=True)

    df["input"] = (
        df["input"]
        .astype(str)
        .str.strip()
    )

    df["output"] = (
        df["output"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["input"] != "")
        &
        (df["output"] != "")
    ]

    df.drop_duplicates(
        subset=["input", "output"],
        inplace=True
    )

    return df
    return pd.DataFrame(rows)


alpaca_df = build_alpaca()

print(alpaca_df.head())
print(alpaca_df.shape)
alpaca_df.to_csv(
    "processed/alpaca_clean.csv",
    index=False
)

print("Saved alpaca_clean.csv")
def build_cnn():
    print("Loading CNN/DailyMail...")

    cnn = load_dataset(
        "abisee/cnn_dailymail",
        "3.0.0"
    )

    train = cnn["train"]

    rows = []

    for article, summary in zip(
        train["article"],
        train["highlights"]
    ):

        article = clean_text(article)
        summary = clean_text(summary)

        rows.append({
            "source": "cnn_dailymail",
            "category": "summarization",
            "input": article,
            "output": summary
        })

    df = pd.DataFrame(rows)

    df.dropna(inplace=True)

    df = df[
        (df["input"] != "")
        &
        (df["output"] != "")
    ]

    df.drop_duplicates(
        subset=["input", "output"],
        inplace=True
    )

    return df
alpaca_df = build_alpaca()
cnn_df = build_cnn()

print("Alpaca:", alpaca_df.shape)
print("CNN:", cnn_df.shape)

merged = pd.concat(
    [alpaca_df, cnn_df],
    ignore_index=True
)

print("Merged:", merged.shape)

merged.to_csv(
    "processed/labelled_v1.csv",
    index=False
)

print("Saved labelled_v1.csv")
print(merged["category"].value_counts())
print(merged.isnull().sum())
def build_boolq():
    print("Loading BoolQ...")

    boolq = load_dataset("google/boolq")

    train = boolq["train"]

    rows = []

    for question, passage, answer in zip(
        train["question"],
        train["passage"],
        train["answer"]
    ):

        question = clean_text(question)
        passage = clean_text(passage)

        final_input = (
            f"Question: {question}\n\n"
            f"Passage: {passage}"
        )

        rows.append({
            "source": "boolq",
            "category": "qa",
            "input": final_input,
            "output": str(answer)
        })

    df = pd.DataFrame(rows)

    df.drop_duplicates(
        subset=["input", "output"],
        inplace=True
    )

    return df
boolq_df = build_boolq()

merged = pd.concat(
    [alpaca_df, cnn_df, boolq_df],
    ignore_index=True
)
merged.to_csv(
    "processed/labelled_v2.csv",
    index=False
)
merged["input_len"] = merged["input"].str.len()
merged["output_len"] = merged["output"].str.len()

print(merged["input_len"].describe())
print(merged["output_len"].describe())
merged[
    merged["input"].str.len() < 10
]
print(merged.shape)

print("\nCategory Counts:")
print(merged["category"].value_counts())

print("\nNull Values:")
print(merged.isnull().sum())
merged.to_csv(
    "processed/labelled_v2.csv",
    index=False
)
def build_wikitext():
    print("Loading WikiText...")

    wiki = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1"
    )

    train = wiki["train"]

    rows = []

    for text in train["text"]:

        text = clean_text(text)

        rows.append({
            "source": "wikitext",
            "category": "knowledge",
            "text": text
        })

    df = pd.DataFrame(rows)
    df.dropna(inplace=True)

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"] != ""
    ]

    df.drop_duplicates(
        subset=["text"],
        inplace=True
    )

    df = df[
        df["text"].str.len() > 20
    ]

    return df
wiki_df = build_wikitext()

print(wiki_df.head())
print(wiki_df.shape)
wiki_df.to_csv(
    "processed/unlabelled_v1.csv",
    index=False
)

print("Saved unlabelled_v1.csv")
print(wiki_df.shape)
def build_oasst():
    print("Loading OpenAssistant...")

    oasst = load_dataset("OpenAssistant/oasst1")

    train = pd.DataFrame(oasst["train"])

    train["message_id"] = train["message_id"].astype(str)
    train["parent_id"] = train["parent_id"].astype(str)

    id_lookup = (
        train
        .set_index("message_id")
        .to_dict("index")
    )

    rows = []

    for _, row in train.iterrows():

        if row["role"] != "assistant":
            continue

        parent_id = row["parent_id"]

        if parent_id not in id_lookup:
            continue

        parent = id_lookup[parent_id]

        if parent["role"] != "prompter":
            continue

        prompt = clean_text(parent["text"])
        response = clean_text(row["text"])

        if not prompt or not response:
            continue

        rows.append({
            "source": "openassistant",
            "category": "conversation",
            "input": prompt,
            "output": response
        })

    df = pd.DataFrame(rows)

    df.drop_duplicates(
        subset=["input", "output"],
        inplace=True
    )

    return df
oasst_df = build_oasst()

print(oasst_df.head())
print(oasst_df.shape)
merged = pd.concat(
    [
        alpaca_df,
        cnn_df,
        boolq_df,
        oasst_df
    ],
    ignore_index=True
)
merged.to_csv(
    "processed/final_labelled.csv",
    index=False
)
def build_dolly():
    print("Loading Dolly...")

    dolly = load_dataset(
        "databricks/databricks-dolly-15k"
    )

    train = dolly["train"]

    rows = []

    for instruction, context, response in zip(
        train["instruction"],
        train["context"],
        train["response"]
    ):

        instruction = clean_text(instruction)
        context = clean_text(context)
        response = clean_text(response)

        if context:
            final_input = (
                instruction +
                "\n\nContext:\n" +
                context
            )
        else:
            final_input = instruction

        rows.append({
            "source": "dolly",
            "category": "instruction",
            "input": final_input,
            "output": response
        })

    df = pd.DataFrame(rows)

    df.drop_duplicates(
        subset=["input", "output"],
        inplace=True
    )

    return df
dolly_df = build_dolly()

final_labelled = pd.concat(
    [
        dolly_df,
        alpaca_df,
        cnn_df,
        boolq_df,
        oasst_df
    ],
    ignore_index=True
)
print(final_labelled.shape)
print(final_labelled["category"].value_counts())
final_labelled.to_csv(
    "processed/final_labelled.csv",
    index=False
)

final_labelled = pd.concat(
    [
        dolly_df,
        alpaca_df,
        boolq_df,
        cnn_df,
        oasst_df
    ],
    ignore_index=True
)

final_labelled.to_csv(
    "processed/final_training_corpus.csv",
    index=False
)
final_labelled["final_category"] = ""
