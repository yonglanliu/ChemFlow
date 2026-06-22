import sqlite3

conn = sqlite3.connect("chem.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS compounds(
               compound_id INTEGER PRIMARY KEY,
               chemble_id TEXT,
               uniprot_id TEXT,
               smiles TEXT)""")