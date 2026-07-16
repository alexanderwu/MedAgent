import re

path = r"E:\MedAgent\notebooks\mimic_merged_eda.pkl"

with open(path, 'rb') as f:
    raw = f.read()

# AWS Access Key IDs: AKIA or ASIA followed by 16 alphanumeric chars (20 total)
matches = re.findall(rb'(AKIA|ASIA)[A-Z0-9]{16}', raw)

print(f"Found {len(matches)} potential AWS key matches")
for m in set(matches):
    print(m)