import pandas as pd
import ollama
import time
import json
import csv
import os
import matplotlib.pyplot as plt

# === CONFIG ===
PROP_MODEL = "mistral"
OPP_MODEL = "llama2"
JUDGE_MODELS = ["phi", "gemma", "neural-chat"]
TOPIC_FILE = "data/Debate Topics.csv"
STRUCTURE_FILE = "data/Debator Prompts.csv"
RESPONSES_OUTPUT = "data/Debate_Responses.csv"
SCORES_OUTPUT = "data/Debate_Judge_Scores.csv"
TRANSCRIPT_FOLDER = "data/transcripts"

# === Round Templates ===
rounds = {
    1: {"name": "Opening", "suffix": "FIRST RESPONSE (Opening)"},
    2: {"name": "Rebuttal & Case Extension", "suffix": "SECOND RESPONSE (Rebuttal & Case Extension)"},
    3: {"name": "Summary & Clash", "suffix": "THIRD RESPONSE (Summary & Clash)"},
    4: {"name": "Closing Summary", "suffix": "FINAL RESPONSE (Closing Summary)"}
}

# === Setup Output Files ===
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

if not os.path.exists(RESPONSES_OUTPUT):
    with open(RESPONSES_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Debate Topic", "Round", "Proposition Response", "Opposition Response"])

if not os.path.exists(SCORES_OUTPUT):
    with open(SCORES_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Debate Topic", "Judge Model", "Prop_Total", "Opp_Total", "Winner"])

# === Load Topics and Prompt Structures ===
# Load Topics
topics_df = pd.read_csv(TOPIC_FILE, header=None)
topics_df = topics_df[[0]]  # keep only the first column
topics_df.columns = ["Debate Topic"]


template_df = pd.read_csv(STRUCTURE_FILE)
prop_structure = template_df.columns[0]
opp_structure = template_df.columns[1]

# === Process Each Topic ===
for _, row in topics_df.iterrows():
    topic = row["Debate Topic"]
    base_prop_prompt = f'You are debating in favor of the motion: "{topic}"\n\n{prop_structure}'
    base_opp_prompt = f'You are debating against the motion: "{topic}"\n\n{opp_structure}'
    history = []
    transcript = []

    print(f"\n🔥 Running debate on topic: {topic}")

    for round_num, info in rounds.items():
        prop_prompt = f"{base_prop_prompt}\n\nIMPORTANT: Generate only the {info['suffix']}."
        opp_prompt = f"{base_opp_prompt}\n\nIMPORTANT: Generate only the {info['suffix']}."

        print(f"\n🌀 Round {round_num}: {info['name']}")
        prop_messages = history + [{"role": "user", "content": prop_prompt}]
        opp_messages = history + [{"role": "user", "content": opp_prompt}]

        prop_response = ollama.chat(model=PROP_MODEL, messages=prop_messages)["message"]["content"]
        opp_response = ollama.chat(model=OPP_MODEL, messages=opp_messages)["message"]["content"]

        history.extend([
            {"role": "user", "content": prop_prompt},
            {"role": "assistant", "content": prop_response},
            {"role": "user", "content": opp_prompt},
            {"role": "assistant", "content": opp_response},
        ])

        with open(RESPONSES_OUTPUT, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([topic, round_num, prop_response, opp_response])

        transcript.append(f"Round {round_num} - {info['name']}\n\nProposition:\n{prop_response}\n\nOpposition:\n{opp_response}\n\n")

    # Save individual transcript
    with open(os.path.join(TRANSCRIPT_FOLDER, f"{topic[:50].replace('/', '-')}.txt"), "w", encoding="utf-8") as tf:
        tf.write("\n".join(transcript))

    time.sleep(2)  # Auto pause between debates

# === JUDGING ===
df = pd.read_csv(RESPONSES_OUTPUT)
all_topics = df["Debate Topic"].unique()

for topic in all_topics:
    topic_df = df[df["Debate Topic"] == topic].sort_values("Round")
    prop_text = ""
    opp_text = ""
    for _, r in topic_df.iterrows():
        prop_text += f"Round {r['Round']}:\n{r['Proposition Response']}\n\n"
        opp_text += f"Round {r['Round']}:\n{r['Opposition Response']}\n\n"

    def build_judge_prompt():
        return f"""
You are an impartial debate judge. Evaluate the full 4-round debate on:
"{topic}"

--- PROPOSITION ---
{prop_text}
--- OPPOSITION ---
{opp_text}

Score each side from 0–10 for:
- Content (40%)
- Style (40%)
- Strategy (20%)

Return a JSON like:
{{
  "Proposition": {{"Content": x, "Style": y, "Strategy": z}},
  "Opposition": {{"Content": a, "Style": b, "Strategy": c}}
}}
"""

    for model in JUDGE_MODELS:
        print(f"\n🧑‍⚖️ Judging topic '{topic}' with {model}...")
        prompt = build_judge_prompt()
        try:
            result = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            content = result["message"]["content"]
            scores = json.loads(content)

            prop_total = round(0.4 * scores["Proposition"]["Content"] + 0.4 * scores["Proposition"]["Style"] + 0.2 * scores["Proposition"]["Strategy"], 2)
            opp_total = round(0.4 * scores["Opposition"]["Content"] + 0.4 * scores["Opposition"]["Style"] + 0.2 * scores["Opposition"]["Strategy"], 2)
            winner = f"{model}: " + ("Proposition" if prop_total > opp_total else "Opposition" if opp_total > prop_total else "Tie")

            with open(SCORES_OUTPUT, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([topic, model, prop_total, opp_total, winner])

            print(f"✅ Judged by {model} — Winner: {winner}")
        except Exception as e:
            print(f"❌ Error judging with {model}: {e}")

# === Optional Summary Chart ===
score_df = pd.read_csv(SCORES_OUTPUT)
winner_counts = score_df["Winner"].apply(lambda x: x.split(": ")[1]).value_counts()
plt.figure(figsize=(6, 4))
plt.bar(winner_counts.index, winner_counts.values, color=["steelblue", "indianred", "gray"][:len(winner_counts)])
plt.title("Judge Vote Summary Across All Debates")
plt.xlabel("Side")
plt.ylabel("Total Votes")
plt.tight_layout()
plt.savefig("data/all_judges_vote_summary.png")
plt.show()
