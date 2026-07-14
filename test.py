import requests

try:
    # Try to reach a public website
    response = requests.get("https://httpbin.org/get", timeout=5)
    print("✅ Python CAN access the internet. Status:", response.status_code)
except Exception as e:
    print("❌ Python is BLOCKED from the internet:", e)