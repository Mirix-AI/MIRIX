-- PostgreSQL Database Migration Script for Mirix
-- Migrates database schema to add user_id columns and other schema changes
-- This script is equivalent to the SQLite migration in migrate_database.py
-- Updated to work with VARCHAR IDs instead of UUID types

-- Start transaction to ensure atomicity
BEGIN;

-- Create a function to check if a column exists
CREATE OR REPLACE FUNCTION column_exists(tbl_name text, col_name text) 
RETURNS boolean AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = tbl_name 
        AND column_name = col_name
    );
END;
$$ LANGUAGE plpgsql;

-- Create default organization if it doesn't exist
DO $$
DECLARE
    default_org_id VARCHAR;
BEGIN
    -- Check if any organization exists
    SELECT id INTO default_org_id FROM organizations LIMIT 1;
    
    IF default_org_id IS NULL THEN
        -- Create default organization with string ID format
        default_org_id := 'org-' || replace(gen_random_uuid()::text, '-', '-');
        INSERT INTO organizations (id, name, is_deleted) 
        VALUES (default_org_id, 'Default Organization', false);
        RAISE NOTICE 'Created default organization with ID: %', default_org_id;
    ELSE
        RAISE NOTICE 'Using existing organization ID: %', default_org_id;
    END IF;
END;
$$;

-- Create default user if it doesn't exist and get user ID for migrations
DO $$
DECLARE
    default_user_id VARCHAR;
    default_org_id VARCHAR;
BEGIN
    -- Check if any user exists
    SELECT id INTO default_user_id FROM users LIMIT 1;
    
    IF default_user_id IS NULL THEN
        -- Get the organization ID
        SELECT id INTO default_org_id FROM organizations LIMIT 1;
        
        -- Create default user with string ID format
        default_user_id := 'user-' || replace(gen_random_uuid()::text, '-', '-');
        INSERT INTO users (id, name, timezone, organization_id, is_deleted) 
        VALUES (default_user_id, 'Default User', 'UTC', default_org_id, false);
        RAISE NOTICE 'Created default user with ID: %', default_user_id;
    ELSE
        RAISE NOTICE 'Using existing user ID: %', default_user_id;
    END IF;
    
    -- Store the default user ID in a temporary table for use in subsequent migrations
    CREATE TEMP TABLE IF NOT EXISTS migration_vars (
        default_user_id VARCHAR
    );
    DELETE FROM migration_vars;
    INSERT INTO migration_vars (default_user_id) VALUES (default_user_id);
END;
$$;

-- Migration 1: Add status column to users table if it doesn't exist
DO $$
BEGIN
    IF NOT column_exists('users', 'status') THEN
        ALTER TABLE users ADD COLUMN status VARCHAR NOT NULL DEFAULT 'active';
        RAISE NOTICE '✓ Added status column to users table';
    ELSE
        RAISE NOTICE '✓ Skipped: status column already exists in users table';
    END IF;
END;
$$;

-- Migration 2: Add mcp_tools column to agents table if it doesn't exist
DO $$
BEGIN
    IF NOT column_exists('agents', 'mcp_tools') THEN
        ALTER TABLE agents ADD COLUMN mcp_tools JSONB;
        RAISE NOTICE '✓ Added mcp_tools column to agents table';
    ELSE
        RAISE NOTICE '✓ Skipped: mcp_tools column already exists in agents table';
    END IF;
END;
$$;

-- Migration 3: Add user_id column to block table if it doesn't exist
DO $$
DECLARE
    default_user_id VARCHAR;
BEGIN
    SELECT migration_vars.default_user_id INTO default_user_id FROM migration_vars;
    
    IF NOT column_exists('block', 'user_id') THEN
        ALTER TABLE block ADD COLUMN user_id VARCHAR;
        UPDATE block SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Added and populated user_id column in block table';
    ELSE
        -- Update any NULL values
        UPDATE block SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Skipped: user_id column already exists in block table (updated NULL values)';
    END IF;
END;
$$;

-- Migration 4: Add user_id column to files table if it doesn't exist
DO $$
DECLARE
    default_user_id VARCHAR;
BEGIN
    SELECT migration_vars.default_user_id INTO default_user_id FROM migration_vars;
    
    IF NOT column_exists('files', 'user_id') THEN
        ALTER TABLE files ADD COLUMN user_id VARCHAR;
        UPDATE files SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Added and populated user_id column in files table';
    ELSE
        -- Update any NULL values
        UPDATE files SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Skipped: user_id column already exists in files table (updated NULL values)';
    END IF;
END;
$$;

-- Migration 5: Add user_id column to cloud_file_mapping table if it doesn't exist
DO $$
DECLARE
    default_user_id VARCHAR;
BEGIN
    SELECT migration_vars.default_user_id INTO default_user_id FROM migration_vars;
    
    IF NOT column_exists('cloud_file_mapping', 'user_id') THEN
        ALTER TABLE cloud_file_mapping ADD COLUMN user_id VARCHAR;
        UPDATE cloud_file_mapping SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Added and populated user_id column in cloud_file_mapping table';
    ELSE
        -- Update any NULL values
        UPDATE cloud_file_mapping SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Skipped: user_id column already exists in cloud_file_mapping table (updated NULL values)';
    END IF;
END;
$$;

-- Migration 6: Add user_id column to episodic_memory table if it doesn't exist
DO $$
DECLARE
    default_user_id VARCHAR;
BEGIN
    SELECT migration_vars.default_user_id INTO default_user_id FROM migration_vars;
    
    IF NOT column_exists('episodic_memory', 'user_id') THEN
        ALTER TABLE episodic_memory ADD COLUMN user_id VARCHAR;
        UPDATE episodic_memory SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Added and populated user_id column in episodic_memory table';
    ELSE
        -- Update any NULL values
        UPDATE episodic_memory SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Skipped: user_id column already exists in episodic_memory table (updated NULL values)';
    END IF;
END;
$$;

-- Migration 7: Add user_id column to knowledge_vault table if it doesn't exist
DO $$
DECLARE
    default_user_id VARCHAR;
BEGIN
    SELECT migration_vars.default_user_id INTO default_user_id FROM migration_vars;
    
    IF NOT column_exists('knowledge_vault', 'user_id') THEN
        ALTER TABLE knowledge_vault ADD COLUMN user_id VARCHAR;
        UPDATE knowledge_vault SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Added and populated user_id column in knowledge_vault table';
    ELSE
        -- Update any NULL values
        UPDATE knowledge_vault SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Skipped: user_id column already exists in knowledge_vault table (updated NULL values)';
    END IF;
END;
$$;

-- Migration 8: Add user_id column to procedural_memory table if it doesn't exist
DO $$
DECLARE
    default_user_id VARCHAR;
BEGIN
    SELECT migration_vars.default_user_id INTO default_user_id FROM migration_vars;
    
    IF NOT column_exists('procedural_memory', 'user_id') THEN
        ALTER TABLE procedural_memory ADD COLUMN user_id VARCHAR;
        UPDATE procedural_memory SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Added and populated user_id column in procedural_memory table';
    ELSE
        -- Update any NULL values
        UPDATE procedural_memory SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Skipped: user_id column already exists in procedural_memory table (updated NULL values)';
    END IF;
END;
$$;

-- Migration 9: Add user_id column to resource_memory table if it doesn't exist
DO $$
DECLARE
    default_user_id VARCHAR;
BEGIN
    SELECT migration_vars.default_user_id INTO default_user_id FROM migration_vars;
    
    IF NOT column_exists('resource_memory', 'user_id') THEN
        ALTER TABLE resource_memory ADD COLUMN user_id VARCHAR;
        UPDATE resource_memory SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Added and populated user_id column in resource_memory table';
    ELSE
        -- Update any NULL values
        UPDATE resource_memory SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Skipped: user_id column already exists in resource_memory table (updated NULL values)';
    END IF;
END;
$$;

-- Migration 10: Add user_id column to semantic_memory table if it doesn't exist
DO $$
DECLARE
    default_user_id VARCHAR;
BEGIN
    SELECT migration_vars.default_user_id INTO default_user_id FROM migration_vars;
    
    IF NOT column_exists('semantic_memory', 'user_id') THEN
        ALTER TABLE semantic_memory ADD COLUMN user_id VARCHAR;
        UPDATE semantic_memory SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Added and populated user_id column in semantic_memory table';
    ELSE
        -- Update any NULL values
        UPDATE semantic_memory SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Skipped: user_id column already exists in semantic_memory table (updated NULL values)';
    END IF;
END;
$$;

-- Migration 11: Add user_id column to messages table if it doesn't exist
DO $$
DECLARE
    default_user_id VARCHAR;
BEGIN
    SELECT migration_vars.default_user_id INTO default_user_id FROM migration_vars;
    
    IF NOT column_exists('messages', 'user_id') THEN
        ALTER TABLE messages ADD COLUMN user_id VARCHAR;
        UPDATE messages SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Added and populated user_id column in messages table';
    ELSE
        -- Update any NULL values
        UPDATE messages SET user_id = default_user_id WHERE user_id IS NULL;
        RAISE NOTICE '✓ Skipped: user_id column already exists in messages table (updated NULL values)';
    END IF;
END;
$$;

-- Verification: Check that all required columns exist and are populated
DO $$
DECLARE
    table_name text;
    column_name text;
    null_count integer;
    tables_with_user_id text[] := ARRAY['block', 'files', 'cloud_file_mapping', 'episodic_memory', 
                                       'knowledge_vault', 'procedural_memory', 'resource_memory', 
                                       'semantic_memory', 'messages'];
    required_columns text[][] := ARRAY[
        ARRAY['users', 'status'],
        ARRAY['agents', 'mcp_tools'],
        ARRAY['block', 'user_id'],
        ARRAY['files', 'user_id'],
        ARRAY['cloud_file_mapping', 'user_id'],
        ARRAY['episodic_memory', 'user_id'],
        ARRAY['knowledge_vault', 'user_id'],
        ARRAY['procedural_memory', 'user_id'],
        ARRAY['resource_memory', 'user_id'],
        ARRAY['semantic_memory', 'user_id'],
        ARRAY['messages', 'user_id']
    ];
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '🔍 Verifying migration...';
    
    -- Check that required columns exist
    FOR i IN 1..array_length(required_columns, 1) LOOP
        table_name := required_columns[i][1];
        column_name := required_columns[i][2];
        
        IF column_exists(table_name, column_name) THEN
            RAISE NOTICE '✓ %.% exists', table_name, column_name;
        ELSE
            RAISE NOTICE '❌ %.% missing', table_name, column_name;
        END IF;
    END LOOP;
    
    -- Check that user_id columns have been populated
    FOREACH table_name IN ARRAY tables_with_user_id LOOP
        BEGIN
            EXECUTE format('SELECT COUNT(*) FROM %I WHERE user_id IS NULL', table_name) INTO null_count;
            IF null_count = 0 THEN
                RAISE NOTICE '✓ %.user_id populated', table_name;
            ELSE
                RAISE NOTICE '⚠️  %.user_id has % NULL values', table_name, null_count;
            END IF;
        EXCEPTION
            WHEN undefined_table THEN
                RAISE NOTICE '⚠️  Could not verify %.user_id (table does not exist)', table_name;
            WHEN OTHERS THEN
                RAISE NOTICE '⚠️  Could not verify %.user_id (error: %)', table_name, SQLERRM;
        END;
    END LOOP;
END;
$$;

-- Clean up temporary objects
DROP FUNCTION IF EXISTS column_exists(text, text);
DROP TABLE IF EXISTS migration_vars;

-- Commit the transaction
COMMIT;

-- Final success message
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '✅ PostgreSQL migration completed successfully!';
    RAISE NOTICE 'All schema changes have been applied and verified.';
END;
$$;
