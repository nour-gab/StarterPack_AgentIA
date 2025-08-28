import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
import json

# Load data
personnes = pd.read_excel('data/Données_Assurance_S2.1.xlsx')
colonnes_desc = pd.read_excel('data/Description des colonnes-thématique 2.xlsx', sheet_name='Sheet1')
mapping_produits = pd.read_excel('data/Mapping produits vs profils_cibles.xlsx', sheet_name='Sheet1')
garanties = pd.read_excel('data/Description_garanties.xlsx', sheet_name='Sheet1')

with open('data/api_next.postman_collection.json') as f:
    api_data = json.load(f)

# Create DB connection
engine = create_engine('sqlite:///db/insurance.db', echo=True)
with engine.begin() as conn:  # begin() ensures auto-commit
    # Ingest tables
    personnes.to_sql('personnes', conn, if_exists='replace', index=False)
    colonnes_desc.to_sql('colonnes_desc', conn, if_exists='replace', index=False)
    mapping_produits.to_sql('mapping_produits', conn, if_exists='replace', index=False)
    garanties.to_sql('garanties', conn, if_exists='replace', index=False)

    # Drop old view if it exists
    conn.execute(text("DROP VIEW IF EXISTS client_profiles"))

    # Recreate fixed view
    conn.execute(text("""
    CREATE VIEW client_profiles AS
    SELECT p.REF_PERSONNE,
           p.RAISON_SOCIALE,
           p.LIB_SECTEUR_ACTIVITE,
           m.LIB_PRODUIT,
           m."Profils cibles" AS Profils_cibles
    FROM personnes p
    LEFT JOIN mapping_produits m
        ON p.LIB_SECTEUR_ACTIVITE = m.LIB_BRANCHE
        OR p.LIB_ACTIVITE = m.LIB_PRODUIT
    """))

print("✅ DB ingested successfully with client_profiles view.")
