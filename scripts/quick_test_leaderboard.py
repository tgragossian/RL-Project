"""Quick test to see what the leaderboard response looks like."""
import requests
import json

API_KEY = "RGAPI-c6aee0b2-4405-465a-a1fb-507e754538d6"
platform = "na1"

url = f"https://{platform}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
headers = {"X-Riot-Token": API_KEY}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("Keys in response:", list(data.keys()))
    if 'entries' in data and data['entries']:
        print("\nFirst entry keys:", list(data['entries'][0].keys()))
        print("\nFirst entry:", json.dumps(data['entries'][0], indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
