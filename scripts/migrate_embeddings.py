import os
import sys
from sqlalchemy import create_engine, text

# Load environment variables manually to ensure we get the correct DB URI
def load_env_file(env_file_path):
    if os.path.exists(env_file_path):
        print(f"Loading environment from {env_file_path}")
        with open(env_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

# Adjust path to find mirix package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
load_env_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from mirix.settings import settings

def migrate_embeddings():
    db_uri = settings.pg_uri
    print(f"Starting database migration using DB: {db_uri}")
    
    engine = create_engine(db_uri)
    
    tables_columns = {
        "raw_memory": ["ocr_text_embedding"],
        "resource_memory": ["summary_embedding"],
        "episodic_memory": ["details_embedding", "summary_embedding"],
        "semantic_memory": ["details_embedding", "summary_embedding", "name_embedding"],
        "procedural_memory": ["steps_embedding", "summary_embedding"],
        "knowledge_vault": ["caption_embedding"]
    }
    
    with engine.connect() as connection:
        for table, columns in tables_columns.items():
            print(f"Processing table: {table}")
            for column in columns:
                print(f"  Migrating column: {column} to VECTOR(1536)")
                try:
                    # 1. Set existing embeddings to NULL to avoid type cast errors
                    connection.execute(text(f"UPDATE {table} SET {column} = NULL"))
                    
                    # 2. Alter the column type to VECTOR(1536)
                    # Note: We use USING NULL to be explicit, although the update above handles it
                    connection.execute(text(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE vector(1536) USING NULL"))
                    
                    print(f"  Successfully migrated {table}.{column}")
                except Exception as e:
                    print(f"  Error migrating {table}.{column}: {e}")
        
        connection.commit()
        print("Database migration completed.")

if __name__ == "__main__":
    migrate_embeddings()
