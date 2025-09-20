#!/usr/bin/env python3
"""
Run PostgreSQL Migration Script
Connects to PostgreSQL database and executes the migration SQL
"""

import os
import sys
from urllib.parse import urlparse
import psycopg2
from psycopg2 import sql


def parse_database_uri(uri):
    """Parse database URI and return connection parameters"""
    parsed = urlparse(uri)
    
    # Remove the driver part (postgresql+pg8000 -> postgresql)
    scheme = parsed.scheme.split('+')[0]
    
    return {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }


def run_migration(db_uri, sql_file_path):
    """Run the migration SQL script"""
    
    print(f"🔄 Starting PostgreSQL migration")
    print(f"Database URI: {db_uri}")
    print(f"SQL file: {sql_file_path}")
    
    # Parse database URI
    conn_params = parse_database_uri(db_uri)
    print(f"Connecting to database: {conn_params['database']} on {conn_params['host']}:{conn_params['port']}")
    
    # Read SQL file
    if not os.path.exists(sql_file_path):
        print(f"❌ SQL file not found: {sql_file_path}")
        return False
    
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        migration_sql = f.read()
    
    # Connect to database
    try:
        conn = psycopg2.connect(**conn_params)
        conn.set_session(autocommit=True)  # Enable autocommit for notices
        
        print("✅ Connected to PostgreSQL database")
        
        # Execute migration SQL
        cursor = conn.cursor()
        
        print("\n🚀 Executing migration SQL...")
        cursor.execute(migration_sql)
        
        # Get any notices/messages from the migration
        for notice in conn.notices:
            print(notice.strip())
        
        print("\n✅ Migration completed successfully!")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    # Get database URI from environment or command line
    db_uri = os.getenv('MIRIX_PG_URI')
    
    if len(sys.argv) > 1:
        db_uri = sys.argv[1]
    
    if not db_uri:
        print("❌ Database URI not provided")
        print("Usage:")
        print("  export MIRIX_PG_URI='postgresql+pg8000://user@host:port/database'")
        print("  python run_postgresql_migration.py")
        print("Or:")
        print("  python run_postgresql_migration.py 'postgresql+pg8000://user@host:port/database'")
        sys.exit(1)
    
    # SQL file path
    sql_file_path = os.path.join(os.path.dirname(__file__), 'migrate_database_postgresql.sql')
    
    # Run migration
    success = run_migration(db_uri, sql_file_path)
    
    if success:
        print("\n🎉 Database migration completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Database migration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
