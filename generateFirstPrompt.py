import pandas as pd

# === Load topic and prompt template ===
topics_df = pd.read_csv("data/Debate Topics.csv")
template_df = pd.read_csv("data/Debator Prompts.csv")

# === Extract structure from header names ===
prop_structure = template_df.columns[0]
opp_structure = template_df.columns[1]

# === Get the first topic only ===
topic = topics_df.iloc[0, 0]

# === Explicit instruction to limit model generation to first round ===
explicit_instruction = (
    "\n\nIMPORTANT: You must only generate the FIRST RESPONSE (Opening). \n"
    "Do NOT generate the Second, Third, or Final Response.\n"
    "Only produce the Opening Statement and then stop."
)

# === Build final prompts ===
final_row = {
    "Debate Topic": topic,
    "Proposition Final Prompt": f'You are debating in favor of the motion: "{topic}"\n\n{prop_structure}{explicit_instruction}',
    "Opposition Final Prompt": f'You are debating against the motion: "{topic}"\n\n{opp_structure}{explicit_instruction}'
}

# === Create DataFrame and save ===
final_prompts_df = pd.DataFrame([final_row])
output_path = "data/Final_Debator_Prompts.csv"
final_prompts_df.to_csv(output_path, index=False)

print("✅ Final prompt (Opening only) saved to:", output_path)
print(final_prompts_df.head())
