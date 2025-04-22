import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
SCORES_FILE = "data/Debate_Judge_Scores.csv"
TOPICS_FILE = "data/Debate Topics.csv"
PROP_MODEL = "mistral"
OPP_MODEL = "llama2"

# === Load Scores and Topics ===
scores_df = pd.read_csv(SCORES_FILE)
topics_df = pd.read_csv(TOPICS_FILE)

# === Clean and Aggregate ===
scores_df["Side Winner"] = scores_df["Winner"].apply(lambda x: x.split(": ")[1])

# Compute average score per side per topic
avg_scores = scores_df.groupby("Debate Topic").agg({
    "Prop_Total": "mean",
    "Opp_Total": "mean"
}).reset_index()

# Decide winner based on average scores
avg_scores["Winner"] = avg_scores.apply(
    lambda row: PROP_MODEL if row["Prop_Total"] > row["Opp_Total"] else OPP_MODEL if row["Opp_Total"] > row["Prop_Total"] else "Tie",
    axis=1
)

# Merge topic metadata
if "Category" in topics_df.columns:
    avg_scores = avg_scores.merge(topics_df[["Debate Topic", "Category"]], on="Debate Topic", how="left")

# === Save Summary ===
summary_path = "data/Model_Comparison_Summary.csv"
avg_scores.to_csv(summary_path, index=False)
print(f"✅ Saved model comparison summary to {summary_path}")

# === Bar Chart: Wins per Model ===
win_counts = avg_scores["Winner"].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(win_counts.index, win_counts.values, color=["royalblue", "darkred", "gray"][:len(win_counts)])
plt.title("🏆 Debate Wins by Model")
plt.xlabel("Model")
plt.ylabel("# Topics Won")
plt.tight_layout()
plt.savefig("data/model_win_comparison.png")
plt.show()

# === Optional: Wins by Topic Category ===
if "Category" in avg_scores.columns:
    cat_wins = avg_scores.groupby(["Category", "Winner"]).size().unstack(fill_value=0)
    cat_wins.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="coolwarm")
    plt.title("🏆 Wins by Topic Category")
    plt.ylabel("# Wins")
    plt.xlabel("Category")
    plt.tight_layout()
    plt.savefig("data/wins_by_category.png")
    plt.show()