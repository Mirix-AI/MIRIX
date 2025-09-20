#!/usr/bin/env python3
"""
Database Migration Script for Mirix
Migrates old database format to new format with added user_id columns and other schema changes.
"""

import sqlite3
import sys
import os
from pathlib import Path
import shutil
from datetime import datetime
import uuid


def backup_database(db_path):
    """Create a backup of the database before migration"""
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path


def get_default_user_id(conn):
    """Get or create a default user ID for migration"""
    cursor = conn.cursor()
    
    # Try to find an existing user
    cursor.execute("SELECT id FROM users LIMIT 1")
    result = cursor.fetchone()
    
    if result:
        user_id = result[0]
        print(f"Using existing user ID: {user_id}")
        return user_id
    
    # Create a default user if none exists
    default_org_id = get_default_organization_id(conn)
    user_id = str(uuid.uuid4())
    
    cursor.execute("""
        INSERT INTO users (id, name, timezone, organization_id, status)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, "Default User", "UTC", default_org_id, "active"))
    
    print(f"Created default user with ID: {user_id}")
    return user_id


def get_default_organization_id(conn):
    """Get or create a default organization ID"""
    cursor = conn.cursor()
    
    # Try to find an existing organization
    cursor.execute("SELECT id FROM organizations LIMIT 1")
    result = cursor.fetchone()
    
    if result:
        org_id = result[0]
        print(f"Using existing organization ID: {org_id}")
        return org_id
    
    # Create a default organization if none exists
    org_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO organizations (id, name)
        VALUES (?, ?)
    """, (org_id, "Default Organization"))
    
    print(f"Created default organization with ID: {org_id}")
    return org_id


def check_column_exists(conn, table_name, column_name):
    """Check if a column exists in a table"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def migrate_database(old_db_path, new_db_path):
    """Migrate database from old format to new format"""
    
    print(f"Starting migration from {old_db_path} to {new_db_path}")
    
    # Create backup
    backup_path = backup_database(old_db_path)
    
    # Copy old database to new location
    shutil.copy2(old_db_path, new_db_path)
    
    # Connect to the database
    conn = sqlite3.connect(new_db_path)
    conn.execute("PRAGMA foreign_keys = OFF")  # Disable foreign keys during migration
    
    try:
        # Get default user and organization IDs
        default_user_id = get_default_user_id(conn)
        
        # Migration steps
        migrations = [
            # Add status column to users table if it doesn't exist
            {
                'name': 'Add status column to users table',
                'check': lambda: check_column_exists(conn, 'users', 'status'),
                'execute': lambda: conn.execute("ALTER TABLE users ADD COLUMN status VARCHAR NOT NULL DEFAULT 'active'")
            },
            
            # Add mcp_tools column to agents table if it doesn't exist
            {
                'name': 'Add mcp_tools column to agents table',
                'check': lambda: check_column_exists(conn, 'agents', 'mcp_tools'),
                'execute': lambda: conn.execute("ALTER TABLE agents ADD COLUMN mcp_tools JSON")
            },
            
            # Add user_id column to block table if it doesn't exist
            {
                'name': 'Add user_id column to block table',
                'check': lambda: check_column_exists(conn, 'block', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE block ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE block SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                    # Note: We can't easily add NOT NULL constraint to existing column in SQLite
                ]
            },
            
            # Add user_id column to files table if it doesn't exist
            {
                'name': 'Add user_id column to files table',
                'check': lambda: check_column_exists(conn, 'files', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE files ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE files SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to cloud_file_mapping table if it doesn't exist
            {
                'name': 'Add user_id column to cloud_file_mapping table',
                'check': lambda: check_column_exists(conn, 'cloud_file_mapping', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE cloud_file_mapping ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE cloud_file_mapping SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to episodic_memory table if it doesn't exist
            {
                'name': 'Add user_id column to episodic_memory table',
                'check': lambda: check_column_exists(conn, 'episodic_memory', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE episodic_memory ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE episodic_memory SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to knowledge_vault table if it doesn't exist
            {
                'name': 'Add user_id column to knowledge_vault table',
                'check': lambda: check_column_exists(conn, 'knowledge_vault', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE knowledge_vault ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE knowledge_vault SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to procedural_memory table if it doesn't exist
            {
                'name': 'Add user_id column to procedural_memory table',
                'check': lambda: check_column_exists(conn, 'procedural_memory', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE procedural_memory ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE procedural_memory SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to resource_memory table if it doesn't exist
            {
                'name': 'Add user_id column to resource_memory table',
                'check': lambda: check_column_exists(conn, 'resource_memory', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE resource_memory ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE resource_memory SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to semantic_memory table if it doesn't exist
            {
                'name': 'Add user_id column to semantic_memory table',
                'check': lambda: check_column_exists(conn, 'semantic_memory', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE semantic_memory ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE semantic_memory SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to messages table if it doesn't exist
            {
                'name': 'Add user_id column to messages table',
                'check': lambda: check_column_exists(conn, 'messages', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE messages ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE messages SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
        ]
        
        # Execute migrations
        for migration in migrations:
            print(f"Checking: {migration['name']}")
            if not migration['check']():
                print(f"Executing: {migration['name']}")
                result = migration['execute']()
                if isinstance(result, list):
                    # Handle multiple statements
                    pass
                conn.commit()
                print(f"✓ Completed: {migration['name']}")
            else:
                print(f"✓ Skipped (already exists): {migration['name']}")
        
        # Re-enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
        
        print(f"\n✅ Migration completed successfully!")
        print(f"New database: {new_db_path}")
        print(f"Backup created: {backup_path}")
        
        # Verify the migration
        verify_migration(conn)
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def migrate_database_inplace(db_path):
    """Migrate database in-place"""
    
    print(f"Starting in-place migration of {db_path}")
    
    # Create backup
    backup_path = backup_database(db_path)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")  # Disable foreign keys during migration
    
    try:
        # Get default user and organization IDs
        default_user_id = get_default_user_id(conn)
        
        # Migration steps (same as migrate_database)
        migrations = [
            # Add status column to users table if it doesn't exist
            {
                'name': 'Add status column to users table',
                'check': lambda: check_column_exists(conn, 'users', 'status'),
                'execute': lambda: conn.execute("ALTER TABLE users ADD COLUMN status VARCHAR NOT NULL DEFAULT 'active'")
            },
            
            # Add mcp_tools column to agents table if it doesn't exist
            {
                'name': 'Add mcp_tools column to agents table',
                'check': lambda: check_column_exists(conn, 'agents', 'mcp_tools'),
                'execute': lambda: conn.execute("ALTER TABLE agents ADD COLUMN mcp_tools JSON")
            },
            
            # Add user_id column to block table if it doesn't exist
            {
                'name': 'Add user_id column to block table',
                'check': lambda: check_column_exists(conn, 'block', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE block ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE block SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to files table if it doesn't exist
            {
                'name': 'Add user_id column to files table',
                'check': lambda: check_column_exists(conn, 'files', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE files ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE files SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to cloud_file_mapping table if it doesn't exist
            {
                'name': 'Add user_id column to cloud_file_mapping table',
                'check': lambda: check_column_exists(conn, 'cloud_file_mapping', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE cloud_file_mapping ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE cloud_file_mapping SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to episodic_memory table if it doesn't exist
            {
                'name': 'Add user_id column to episodic_memory table',
                'check': lambda: check_column_exists(conn, 'episodic_memory', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE episodic_memory ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE episodic_memory SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to knowledge_vault table if it doesn't exist
            {
                'name': 'Add user_id column to knowledge_vault table',
                'check': lambda: check_column_exists(conn, 'knowledge_vault', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE knowledge_vault ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE knowledge_vault SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to procedural_memory table if it doesn't exist
            {
                'name': 'Add user_id column to procedural_memory table',
                'check': lambda: check_column_exists(conn, 'procedural_memory', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE procedural_memory ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE procedural_memory SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to resource_memory table if it doesn't exist
            {
                'name': 'Add user_id column to resource_memory table',
                'check': lambda: check_column_exists(conn, 'resource_memory', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE resource_memory ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE resource_memory SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to semantic_memory table if it doesn't exist
            {
                'name': 'Add user_id column to semantic_memory table',
                'check': lambda: check_column_exists(conn, 'semantic_memory', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE semantic_memory ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE semantic_memory SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
            
            # Add user_id column to messages table if it doesn't exist
            {
                'name': 'Add user_id column to messages table',
                'check': lambda: check_column_exists(conn, 'messages', 'user_id'),
                'execute': lambda: [
                    conn.execute("ALTER TABLE messages ADD COLUMN user_id VARCHAR"),
                    conn.execute("UPDATE messages SET user_id = ? WHERE user_id IS NULL", (default_user_id,)),
                ]
            },
        ]
        
        # Execute migrations
        for migration in migrations:
            print(f"Checking: {migration['name']}")
            if not migration['check']():
                print(f"Executing: {migration['name']}")
                result = migration['execute']()
                if isinstance(result, list):
                    # Handle multiple statements
                    pass
                conn.commit()
                print(f"✓ Completed: {migration['name']}")
            else:
                print(f"✓ Skipped (already exists): {migration['name']}")
        
        # Re-enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
        
        print(f"\n✅ In-place migration completed successfully!")
        print(f"Database: {db_path}")
        print(f"Backup created: {backup_path}")
        
        # Verify the migration
        verify_migration(conn)
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def verify_migration(conn):
    """Verify that the migration was successful"""
    print("\n🔍 Verifying migration...")
    
    cursor = conn.cursor()
    
    # Check that required columns exist
    required_columns = [
        ('users', 'status'),
        ('agents', 'mcp_tools'),
        ('block', 'user_id'),
        ('files', 'user_id'),
        ('cloud_file_mapping', 'user_id'),
        ('episodic_memory', 'user_id'),
        ('knowledge_vault', 'user_id'),
        ('procedural_memory', 'user_id'),
        ('resource_memory', 'user_id'),
        ('semantic_memory', 'user_id'),
        ('messages', 'user_id'),
    ]
    
    for table, column in required_columns:
        if check_column_exists(conn, table, column):
            print(f"✓ {table}.{column} exists")
        else:
            print(f"❌ {table}.{column} missing")
    
    # Check that user_id columns have been populated
    tables_with_user_id = [
        'block', 'files', 'cloud_file_mapping', 'episodic_memory',
        'knowledge_vault', 'procedural_memory', 'resource_memory',
        'semantic_memory', 'messages'
    ]
    
    for table in tables_with_user_id:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id IS NULL")
            null_count = cursor.fetchone()[0]
            if null_count == 0:
                print(f"✓ {table}.user_id populated")
            else:
                print(f"⚠️  {table}.user_id has {null_count} NULL values")
        except sqlite3.OperationalError:
            # Table might not exist or have data
            print(f"⚠️  Could not verify {table}.user_id")


def main():
    # Default to migrating ~/.mirix/sqlite.db in-place
    mirix_db_path = os.path.expanduser("~/.mirix/sqlite.db")
    
    if len(sys.argv) == 1:
        # No arguments - migrate ~/.mirix/sqlite.db in-place
        db_path = mirix_db_path
        if not os.path.exists(db_path):
            print(f"❌ Mirix database not found: {db_path}")
            print("Make sure Mirix has been run at least once to create the database.")
            sys.exit(1)
        
        print(f"🔄 In-place migration of Mirix database")
        print(f"  Database: {db_path}")
        print(f"  Backup will be created automatically")
        
        response = input("\nProceed with in-place migration? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            sys.exit(0)
        
        try:
            migrate_database_inplace(db_path)
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            sys.exit(1)
    
    elif len(sys.argv) == 3:
        # Two arguments - migrate from old to new location
        old_db_path = sys.argv[1]
        new_db_path = sys.argv[2]
        
        # Validate input
        if not os.path.exists(old_db_path):
            print(f"❌ Old database not found: {old_db_path}")
            sys.exit(1)
        
        # Create directory for new database if it doesn't exist
        new_db_dir = os.path.dirname(new_db_path)
        if new_db_dir and not os.path.exists(new_db_dir):
            os.makedirs(new_db_dir, exist_ok=True)
            print(f"Created directory: {new_db_dir}")
        
        # Confirm before proceeding
        print(f"Migration plan:")
        print(f"  Source: {old_db_path}")
        print(f"  Target: {new_db_path}")
        print(f"  Backup will be created automatically")
        
        response = input("\nProceed with migration? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            sys.exit(0)
        
        try:
            migrate_database(old_db_path, new_db_path)
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            sys.exit(1)
    
    else:
        print("Usage:")
        print("  python migrate_database.py                    # Migrate ~/.mirix/sqlite.db in-place")
        print("  python migrate_database.py <old> <new>        # Migrate from old to new location")
        print("")
        print("Examples:")
        print("  python migrate_database.py                              # In-place migration")
        print("  python migrate_database.py tmp/sqlite.db ~/.mirix/sqlite.db  # Copy and migrate")
        sys.exit(1)


if __name__ == "__main__":
    main()
