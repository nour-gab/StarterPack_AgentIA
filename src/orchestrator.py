from sql_analyzer import analyze_client
from rag_recommender import recommend
from pitching_bot import generate_pitch

def run_pipeline(client_id):
    profile_raw = analyze_client(client_id)
    if not profile_raw:
        return "Client not found."
    profile = dict(profile_raw[0])  # Simplify to dict
    recs = recommend(profile)
    pitch = generate_pitch(profile['RAISON_SOCIALE'], recs)
    # Log to file/DB for feedback
    with open('logs/interactions.log', 'a') as f:
        f.write(f"Client {client_id}: {pitch}\n")
    return pitch

# Run example
print(run_pipeline(12122))