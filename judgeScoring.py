import pandas as pd
import ollama
import json
import time
import matplotlib.pyplot as plt

# === CONFIG ===
CSV_INPUT = "data/Debate_Responses.csv"
CSV_OUTPUT = "data/Debate_Judge_Scores.csv"
JUDGE_MODELS = ["phi", "gemma", "neural-chat"]

# === Load Full Debate ===
df = pd.read_csv(CSV_INPUT)
debate_topic = df["Debate Topic"].iloc[0]

# Group responses by side
prop_text = ""
opp_text = ""

for _, row in df.iterrows():
    round_num = row["Round"]
    prop_text += f"Round {round_num}:\n{row['Proposition Response']}\n\n"
    opp_text += f"Round {round_num}:\n{row['Opposition Response']}\n\n"

# === Build evaluation prompt ===
def build_prompt(debate_topic, prop_text, opp_text):
    return f"""
You are an impartial debate judge. You have just read a full 4-round debate on the motion:

\"{debate_topic}\"

Here are the complete arguments from each side:

--- PROPOSITION ---
{prop_text}

--- OPPOSITION ---
{opp_text}

Evaluate both sides on the following:
- Content (40%): Are arguments logical, relevant, and well-supported?
- Style (40%): Is the writing clear, formal, and persuasive?
- Strategy (20%): Did they structure their case well, clash effectively, and close strong?

Return a JSON object like:
{{
  "Proposition": {{"Content": x, "Style": y, "Strategy": z}},
  "Opposition": {{"Content": a, "Style": b, "Strategy": c}}
}}

Scores should be between 0 and 10. Do not include any commentary or explanation.
"""

# === Run scoring ===
results = []
for model in JUDGE_MODELS:
    print(f"\n🧑‍⚖️ Scoring debate with judge model: {model}")
    prompt = build_prompt(debate_topic, prop_text, opp_text)
    time.sleep(2)
    try:
        result = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        content = result["message"]["content"]
        scores = json.loads(content)

        prop_score = 0.4 * scores["Proposition"]["Content"] + 0.4 * scores["Proposition"]["Style"] + 0.2 * scores["Proposition"]["Strategy"]
        opp_score = 0.4 * scores["Opposition"]["Content"] + 0.4 * scores["Opposition"]["Style"] + 0.2 * scores["Opposition"]["Strategy"]

        winner = "Proposition" if prop_score > opp_score else "Opposition" if opp_score > prop_score else "Tie"

        row = {
            "Judge Model": model,
            "Debate Topic": debate_topic,
            "Prop_Content": scores["Proposition"]["Content"],
            "Prop_Style": scores["Proposition"]["Style"],
            "Prop_Strategy": scores["Proposition"]["Strategy"],
            "Opp_Content": scores["Opposition"]["Content"],
            "Opp_Style": scores["Opposition"]["Style"],
            "Opp_Strategy": scores["Opposition"]["Strategy"],
            "Prop_Total": round(prop_score, 2),
            "Opp_Total": round(opp_score, 2),
            "Winner": f"{model}: {winner}"
        }
        results.append(row)
        print(f"✅ {winner} wins — judged by {model}")
    except Exception as e:
        print(f"❌ Failed scoring with {model}: {e}")

# === Save results ===
score_df = pd.DataFrame(results)
score_df.to_csv(CSV_OUTPUT, index=False)
print(f"\n✅ Judge scores with winner info saved to: {CSV_OUTPUT}")

# === Summary Report ===
summary = score_df["Winner"].apply(lambda x: x.split(": ")[1]).value_counts().to_dict()
print("\n📊 Summary Report:")
for side, count in summary.items():
    print(f"- {side}: {count} vote(s)")

# === Declare Overall Winner ===
if "Proposition" not in summary:
    summary["Proposition"] = 0
if "Opposition" not in summary:
    summary["Opposition"] = 0

if summary["Proposition"] > summary["Opposition"]:
    print("\n🥇 Overall Winner: Proposition")
elif summary["Opposition"] > summary["Proposition"]:
    print("\n🥇 Overall Winner: Opposition")
else:
    print("\n🤝 It's a Tie!")

# === Visualization ===
plt.figure(figsize=(6, 4))
labels = list(summary.keys())
votes = list(summary.values())
plt.bar(labels, votes, color=["steelblue", "indianred", "gray"][:len(votes)])
plt.title(f"🧑‍⚖️ Judge Votes for Debate: '{debate_topic}'")
plt.ylabel("Votes")
plt.xlabel("Side")
plt.tight_layout()
plt.savefig("data/judge_vote_chart.png")
plt.show()