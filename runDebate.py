import pandas as pd
import time
import ollama
import csv
import os

# === CONFIG ===
PROP_MODEL = "mistral"
OPP_MODEL = "llama2"
PROMPTS_FILE = "data/Final_Debator_Prompts.csv"
CSV_OUTPUT = "data/Debate_Responses.csv"

# === Load Prompts ===
print("📥 Loading prompts...")
prompts_df = pd.read_csv(PROMPTS_FILE)
debate_topic = prompts_df.loc[0, "Debate Topic"]
proposition_prompt_r1 = prompts_df.loc[0, "Proposition Final Prompt"]
opposition_prompt_r1 = prompts_df.loc[0, "Opposition Final Prompt"]

print(f"🎯 Debate Topic: {debate_topic}")

# === Ensure Output CSV Exists ===
if not os.path.exists(CSV_OUTPUT):
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Debate Topic", "Round", "Proposition Response", "Opposition Response"])
    print("📄 Created new CSV for responses.")
else:
    print("📄 Appending to existing CSV for responses.")

# === Round definitions ===
rounds = {
    1: {
        "name": "Opening",
        "prop_prompt": proposition_prompt_r1,
        "opp_prompt": opposition_prompt_r1
    },
    2: {
        "name": "Rebuttal & Case Extension",
        "prop_prompt": (
            "You are now delivering your second-round rebuttal and case extension as the PROPOSITION.\n\n"
            "Please:\n"
            "- Rebut 2–3 key arguments from the Opposition's opening.\n"
            "- Defend your own opening arguments.\n"
            "- Introduce 1 new supporting argument.\n"
            "- Emphasize where the Opposition failed to address your key points.\n"
            "Length: 500–800 words.\n\n"
            "IMPORTANT: Generate only the SECOND RESPONSE (Rebuttal & Case Extension). Do NOT generate the other rounds."
        ),
        "opp_prompt": (
            "You are now delivering your second-round rebuttal and case extension as the OPPOSITION.\n\n"
            "Please:\n"
            "- Rebut 2–3 key arguments from the Proposition's opening.\n"
            "- Defend your own opening arguments.\n"
            "- Introduce 1 new supporting argument.\n"
            "- Emphasize where the Proposition failed to address your key points.\n"
            "Length: 500–800 words.\n\n"
            "IMPORTANT: Generate only the SECOND RESPONSE (Rebuttal & Case Extension). Do NOT generate the other rounds."
        )
    },
    3: {
        "name": "Summary & Clash",
        "prop_prompt": (
            "You are summarizing the debate so far as the PROPOSITION.\n\n"
            "Please:\n"
            "- Identify the key areas of clash in the debate.\n"
            "- Demonstrate why your side has won these crucial points.\n"
            "- Strengthen your most compelling arguments with additional evidence.\n"
            "- Explain why Opposition rebuttals were insufficient.\n"
            "Length: 400–600 words.\n\n"
            "IMPORTANT: Generate only the THIRD RESPONSE (Summary & Clash). Do NOT generate the final round."
        ),
        "opp_prompt": (
            "You are summarizing the debate so far as the OPPOSITION.\n\n"
            "Please:\n"
            "- Identify the key areas of clash in the debate.\n"
            "- Demonstrate why your side has won these crucial points.\n"
            "- Strengthen your most compelling arguments with additional evidence.\n"
            "- Explain why Proposition rebuttals were insufficient.\n"
            "Length: 400–600 words.\n\n"
            "IMPORTANT: Generate only the THIRD RESPONSE (Summary & Clash). Do NOT generate the final round."
        )
    },
    4: {
        "name": "Closing Summary",
        "prop_prompt": (
            "You are delivering your final closing summary as the PROPOSITION.\n\n"
            "Please:\n"
            "- Provide a concise, powerful overview of the debate.\n"
            "- Remind judges why the Proposition presented stronger arguments.\n"
            "- Emphasize your strongest points and the Opposition's key weaknesses.\n"
            "- Do NOT introduce new arguments.\n"
            "Length: 300–400 words.\n\n"
            "IMPORTANT: Generate only the FINAL RESPONSE (Closing Summary)."
        ),
        "opp_prompt": (
            "You are delivering your final closing summary as the OPPOSITION.\n\n"
            "Please:\n"
            "- Provide a concise, powerful overview of the debate.\n"
            "- Remind judges why the Opposition presented stronger arguments.\n"
            "- Emphasize your strongest points and the Proposition's key weaknesses.\n"
            "- Do NOT introduce new arguments.\n"
            "Length: 300–400 words.\n\n"
            "IMPORTANT: Generate only the FINAL RESPONSE (Closing Summary)."
        )
    }
}

# === Generate all rounds ===
previous_responses = []

for round_number, round_info in rounds.items():
    print(f"\n🌀 Round {round_number}: {round_info['name']}")

    print("🧠 Sending to Proposition model...")
    prop_messages = previous_responses + [
        {"role": "user", "content": round_info["prop_prompt"]}
    ]
    prop_response = ollama.chat(model=PROP_MODEL, messages=prop_messages)["message"]["content"]
    print("✅ Proposition response generated.")

    print("🧠 Sending to Opposition model...")
    opp_messages = previous_responses + [
        {"role": "user", "content": round_info["opp_prompt"]}
    ]
    opp_response = ollama.chat(model=OPP_MODEL, messages=opp_messages)["message"]["content"]
    print("✅ Opposition response generated.")

    # Update conversation history
    previous_responses.extend([
        {"role": "user", "content": round_info["prop_prompt"]},
        {"role": "assistant", "content": prop_response},
        {"role": "user", "content": round_info["opp_prompt"]},
        {"role": "assistant", "content": opp_response}
    ])

    # Save to CSV
    print(f"💾 Saving Round {round_number} to CSV...")
    with open(CSV_OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([debate_topic, round_number, prop_response, opp_response])

    print(f"✅ Round {round_number} saved to CSV successfully.")
