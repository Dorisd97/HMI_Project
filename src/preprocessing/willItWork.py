import json
import re

# Load JSON data
with open('d:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_5data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def extract_chain(email):
    # Split body using common reply/forward separators
    body = email.get("Body", "")
    parts = re.split(r'(?:^|\n)(?:-+ ?Original Message ?-+|From:|----- Forwarded by|On .+ wrote:|[\w\s]+@[\w\s]+ on .+)', body, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]  # Remove blanks

    chain = []
    for i, part in enumerate(parts):
        entry = {
            "From": email.get("From", ""),
            "To": email.get("To", ""),
            "Subject": email.get("Subject", ""),
            "Body": part
        }
        if "cc" in email and email["cc"]:
            entry["cc"] = email["cc"]
        chain.append(entry)
    return chain

# Inject BodyChain into each email
for email in data:
    body_chain = extract_chain(email)
    if len(body_chain) > 1:  # Only add if there's a chain
        email["BodyChain"] = body_chain

# Save the updated data
with open('d:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_5data.json', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 90 and i <= 100:
            print(f"{i+1}: {line.strip()}")

print("Updated emails saved with embedded BodyChain.")
