#!/usr/bin/env python3
"""
Mirix Database Reset Script (Python version)
Usage: python reset_database.py [postgresql|sqlite] [connection_params...]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Don't override existing environment variables
                    if key not in os.environ:
                        os.environ[key] = value

# Load .env file at module import
load_env_file()

# Add the project root to Python path so we can import settings
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mirix.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    settings = None

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def log_info(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def log_success(message: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")

def log_warning(message: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")

def log_error(message: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def parse_pg_uri(uri: str) -> Dict[str, str]:
    """Parse PostgreSQL URI and extract connection details."""
    parsed = urlparse(uri)
    return {
        'host': parsed.hostname or 'localhost',
        'port': str(parsed.port or 5432),
        'user': parsed.username or 'mirix',
        'database': parsed.path.lstrip('/') or 'mirix'
    }

def get_pg_connection_details() -> Dict[str, str]:
    """Get PostgreSQL connection details from settings or environment variables."""
    # First try to get from MIRIX_PG_URI environment variable
    pg_uri = os.getenv('MIRIX_PG_URI')
    if pg_uri:
        return parse_pg_uri(pg_uri)
    
    if SETTINGS_AVAILABLE and settings:
        # Try to get from mirix settings first
        if hasattr(settings, 'pg_uri') and settings.pg_uri:
            return parse_pg_uri(settings.pg_uri)
        else:
            return {
                'host': getattr(settings, 'pg_host', None) or 'localhost',
                'port': str(getattr(settings, 'pg_port', None) or 5432),
                'user': getattr(settings, 'pg_user', None) or 'mirix',
                'database': getattr(settings, 'pg_db', None) or 'mirix'
            }
    
    # Fallback to individual environment variables
    return {
        'host': os.getenv('MIRIX_PG_HOST', 'localhost'),
        'port': os.getenv('MIRIX_PG_PORT', '5432'),
        'user': os.getenv('MIRIX_PG_USER', 'mirix'),
        'database': os.getenv('MIRIX_PG_DB', 'mirix')
    }

def detect_database_type() -> str:
    """Auto-detect database type from settings or environment."""
    # Check for MIRIX_PG_URI first
    if os.getenv('MIRIX_PG_URI'):
        return 'postgresql'
    
    if SETTINGS_AVAILABLE and settings:
        if hasattr(settings, 'mirix_pg_uri_no_default') and settings.mirix_pg_uri_no_default:
            return 'postgresql'
    
    # Check other environment variables
    if os.getenv('MIRIX_PG_HOST'):
        return 'postgresql'
    
    # Check for SQLite file
    sqlite_path = Path.home() / '.mirix' / 'sqlite.db'
    if sqlite_path.exists():
        return 'sqlite'
    
    raise ValueError("Could not auto-detect database type. Please specify 'postgresql' or 'sqlite'")

def run_command(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command with error handling."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        if check:
            log_error(f"Command failed: {' '.join(cmd)}")
            log_error(f"Error: {e.stderr.strip() if e.stderr else str(e)}")
            raise
        return e

def reset_postgresql(host: str = None, port: str = None, user: str = None, database: str = None):
    """Reset PostgreSQL database."""
    conn = get_pg_connection_details()
    
    # Override with provided parameters
    if host: conn['host'] = host
    if port: conn['port'] = port
    if user: conn['user'] = user
    if database: conn['database'] = database
    
    log_info(f"Resetting PostgreSQL database: {conn['database']} on {conn['host']}:{conn['port']}")
    
    # Check if database exists
    check_cmd = ['psql', '-h', conn['host'], '-p', conn['port'], '-U', conn['user'], 
                 '-lqt']
    try:
        result = run_command(check_cmd, check=False)
        if conn['database'] not in result.stdout:
            log_warning(f"Database {conn['database']} does not exist")
            return
    except Exception as e:
        log_warning(f"Could not check if database exists: {e}")
    
    # Try to drop and recreate database (cleanest approach)
    log_info("Attempting to drop and recreate database...")
    
    # First, terminate any active connections to the database
    log_info("Terminating active database connections...")
    terminate_sql = f"""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '{conn['database']}' AND pid <> pg_backend_pid();
    """
    terminate_cmd = ['psql', '-h', conn['host'], '-p', conn['port'], '-U', conn['user'], 
                    '-d', 'postgres', '-c', terminate_sql]
    run_command(terminate_cmd, check=False)
    
    # Add a small delay to ensure connections are terminated
    import time
    time.sleep(1)
    
    drop_cmd = ['dropdb', '-h', conn['host'], '-p', conn['port'], '-U', conn['user'], conn['database']]
    
    drop_result = run_command(drop_cmd, check=False)
    if drop_result.returncode == 0:
        log_success(f"Dropped database: {conn['database']}")
    else:
        log_warning(f"Database drop had issues: {drop_result.stderr.strip() if drop_result.stderr else 'Unknown error'}")
    
    # Always try to create the database, regardless of drop success
    log_info(f"Creating database: {conn['database']}")
    create_cmd = ['createdb', '-h', conn['host'], '-p', conn['port'], '-U', conn['user'], conn['database']]
    create_result = run_command(create_cmd, check=False)
    
    if create_result.returncode == 0:
        log_success(f"Created fresh database: {conn['database']}")
    else:
        # If creation failed, try to drop again and recreate
        if "already exists" in create_result.stderr:
            log_info("Database still exists, attempting force drop and recreate...")
            
            # Force drop with more aggressive connection termination
            force_terminate_sql = f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{conn['database']}';
            """
            force_terminate_cmd = ['psql', '-h', conn['host'], '-p', conn['port'], '-U', conn['user'], 
                                 '-d', 'postgres', '-c', force_terminate_sql]
            run_command(force_terminate_cmd, check=False)
            
            time.sleep(2)  # Longer delay
            
            # Try drop again
            drop_result2 = run_command(drop_cmd, check=False)
            if drop_result2.returncode == 0:
                log_success(f"Force dropped database: {conn['database']}")
            
            time.sleep(1)
            
            # Try create again
            create_result2 = run_command(create_cmd, check=False)
            if create_result2.returncode == 0:
                log_success(f"Created fresh database: {conn['database']}")
            else:
                log_warning(f"Database creation still failed, falling back to table truncation...")
                # Continue to table truncation logic below
                pass
        else:
            log_error(f"Database creation failed: {create_result.stderr.strip() if create_result.stderr else 'Unknown error'}")
            log_warning("Falling back to table truncation...")
    
    # If we successfully created the database, enable extensions and return
    if create_result.returncode == 0 or (create_result.returncode != 0 and "create_result2" in locals() and create_result2.returncode == 0):
        # Enable pgvector extension (requires superuser)
        import getpass
        superuser = os.getenv('SUPERUSER', getpass.getuser())
        extension_cmd = ['psql', '-h', conn['host'], '-p', conn['port'], '-U', superuser, 
                        '-d', conn['database'], '-c', 'CREATE EXTENSION IF NOT EXISTS vector;']
        try:
            run_command(extension_cmd, check=False)
            log_success("Enabled pgvector extension")
        except:
            log_warning("Could not enable pgvector extension (may need superuser privileges)")
            log_info(f"You can manually run: psql -U {superuser} -d {conn['database']} -c 'CREATE EXTENSION IF NOT EXISTS vector;'")
        
        return
    
    # If we reach here, both drop/create attempts failed, fall back to table truncation
    log_warning("Database drop/create failed, trying table truncation...")
    
    # Database creation failed, now truncate existing tables
    log_info("Preparing to truncate existing database tables...")
    
    # Truncate tables
    log_info("Truncating database tables...")
    truncate_sql = """
        DO $$
        DECLARE
            table_name text;
            tables text[] := ARRAY[
                'message', 'blocks_agents', 'tools_agents', 'agents_tags',
                'episodic_memory', 'semantic_memory', 'procedural_memory',
                'knowledge_vault', 'resource_memory', 'cloud_file_mapping',
                'step', 'block', 'tool', 'agent', 'sandbox_config',
                'sandbox_environment_variables', 'agent_environment_variables',
                'provider', 'organization', 'user'
            ];
        BEGIN
            FOREACH table_name IN ARRAY tables
            LOOP
                BEGIN
                    EXECUTE 'TRUNCATE TABLE ' || quote_ident(table_name) || ' CASCADE';
                    RAISE NOTICE 'Truncated table: %', table_name;
                EXCEPTION
                    WHEN undefined_table THEN
                        RAISE NOTICE 'Table % does not exist, skipping', table_name;
                END;
            END LOOP;
        END $$;
    """
    
    truncate_cmd = ['psql', '-h', conn['host'], '-p', conn['port'], '-U', conn['user'], 
                   '-d', conn['database'], '-c', truncate_sql]
    run_command(truncate_cmd)
    log_success("Successfully truncated all tables")

def reset_sqlite(database_path: str = None):
    """Reset SQLite database."""
    if not database_path:
        database_path = str(Path.home() / '.mirix' / 'sqlite.db')
    
    log_info(f"Resetting SQLite database: {database_path}")
    
    sqlite_file = Path(database_path)
    if sqlite_file.exists():
        sqlite_file.unlink()
        log_success(f"Deleted SQLite database: {database_path}")
    else:
        log_warning(f"SQLite database does not exist: {database_path}")

def main():
    parser = argparse.ArgumentParser(description='Mirix Database Reset Script')
    parser.add_argument('db_type', nargs='?', default='auto', 
                       choices=['auto', 'postgresql', 'postgres', 'pg', 'sqlite', 'sqlite3'],
                       help='Database type (default: auto-detect)')
    parser.add_argument('--host', help='PostgreSQL host')
    parser.add_argument('--port', help='PostgreSQL port')
    parser.add_argument('--user', help='PostgreSQL user')
    parser.add_argument('--database', help='PostgreSQL database name')
    parser.add_argument('--sqlite-path', help='SQLite database path')
    
    args = parser.parse_args()
    
    try:
        log_info("Starting Mirix database reset...")
        
        # Determine database type
        if args.db_type == 'auto':
            db_type = detect_database_type()
        else:
            db_type = args.db_type
        
        log_info(f"Database type: {db_type}")
        
        # Reset based on database type
        if db_type in ['postgresql', 'postgres', 'pg']:
            reset_postgresql(args.host, args.port, args.user, args.database)
        elif db_type in ['sqlite', 'sqlite3']:
            reset_sqlite(args.sqlite_path)
        else:
            log_error(f"Unsupported database type: {db_type}")
            sys.exit(1)
        
        log_success("Database reset completed!")
        print()
        log_info("Next steps:")
        print("  1. Restart your Mirix application")
        print("  2. The application will automatically recreate the database schema")
        
    except Exception as e:
        log_error(f"Database reset failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 