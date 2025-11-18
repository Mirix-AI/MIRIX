-- PostgreSQL Database Migration Script for Mirix
-- Adds raw_memory table and raw_memory_references columns to memory tables
-- Part of Phase 1: Raw Memory Implementation

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

-- Create a function to check if a table exists
CREATE OR REPLACE FUNCTION table_exists(tbl_name text)
RETURNS boolean AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = tbl_name
    );
END;
$$ LANGUAGE plpgsql;

-- Migration 1: Create raw_memory table if it doesn't exist
DO $$
BEGIN
    IF NOT table_exists('raw_memory') THEN
        CREATE TABLE raw_memory (
            -- Primary key
            id VARCHAR PRIMARY KEY,

            -- Screenshot metadata
            screenshot_path VARCHAR NOT NULL,
            source_app VARCHAR NOT NULL,
            captured_at TIMESTAMP NOT NULL,

            -- OCR extracted data
            ocr_text TEXT,
            source_url VARCHAR,

            -- Cloud storage reference (for Gemini model)
            google_cloud_url VARCHAR,

            -- Additional metadata
            metadata_ JSONB DEFAULT '{}'::jsonb,

            -- Processing status
            processed BOOLEAN DEFAULT false NOT NULL,
            processing_count INTEGER DEFAULT 0 NOT NULL,

            -- Last modification tracking
            last_modify JSONB NOT NULL DEFAULT jsonb_build_object(
                'timestamp', to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'),
                'operation', 'created'
            ),

            -- Embedding configuration
            embedding_config JSONB,

            -- Vector embedding field (pgvector)
            ocr_text_embedding vector(1536),

            -- Foreign keys
            user_id VARCHAR NOT NULL,
            organization_id VARCHAR NOT NULL,

            -- Constraints
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE
        );

        -- Create indexes for common queries
        CREATE INDEX idx_raw_memory_user_id ON raw_memory(user_id);
        CREATE INDEX idx_raw_memory_organization_id ON raw_memory(organization_id);
        CREATE INDEX idx_raw_memory_source_app ON raw_memory(source_app);
        CREATE INDEX idx_raw_memory_captured_at ON raw_memory(captured_at DESC);
        CREATE INDEX idx_raw_memory_processed ON raw_memory(processed);

        RAISE NOTICE '✓ Created raw_memory table';
    ELSE
        RAISE NOTICE '✓ Skipped: raw_memory table already exists';
    END IF;
END;
$$;

-- Migration 2: Add raw_memory_references column to episodic_memory table
DO $$
BEGIN
    IF NOT column_exists('episodic_memory', 'raw_memory_references') THEN
        ALTER TABLE episodic_memory ADD COLUMN raw_memory_references JSONB DEFAULT '[]'::jsonb NOT NULL;
        RAISE NOTICE '✓ Added raw_memory_references column to episodic_memory table';
    ELSE
        RAISE NOTICE '✓ Skipped: raw_memory_references column already exists in episodic_memory table';
    END IF;
END;
$$;

-- Migration 3: Add raw_memory_references column to semantic_memory table
DO $$
BEGIN
    IF NOT column_exists('semantic_memory', 'raw_memory_references') THEN
        ALTER TABLE semantic_memory ADD COLUMN raw_memory_references JSONB DEFAULT '[]'::jsonb NOT NULL;
        RAISE NOTICE '✓ Added raw_memory_references column to semantic_memory table';
    ELSE
        RAISE NOTICE '✓ Skipped: raw_memory_references column already exists in semantic_memory table';
    END IF;
END;
$$;

-- Migration 4: Add raw_memory_references column to procedural_memory table
DO $$
BEGIN
    IF NOT column_exists('procedural_memory', 'raw_memory_references') THEN
        ALTER TABLE procedural_memory ADD COLUMN raw_memory_references JSONB DEFAULT '[]'::jsonb NOT NULL;
        RAISE NOTICE '✓ Added raw_memory_references column to procedural_memory table';
    ELSE
        RAISE NOTICE '✓ Skipped: raw_memory_references column already exists in procedural_memory table';
    END IF;
END;
$$;

-- Migration 5: Add raw_memory_references column to resource_memory table
DO $$
BEGIN
    IF NOT column_exists('resource_memory', 'raw_memory_references') THEN
        ALTER TABLE resource_memory ADD COLUMN raw_memory_references JSONB DEFAULT '[]'::jsonb NOT NULL;
        RAISE NOTICE '✓ Added raw_memory_references column to resource_memory table';
    ELSE
        RAISE NOTICE '✓ Skipped: raw_memory_references column already exists in resource_memory table';
    END IF;
END;
$$;

-- Migration 6: Add raw_memory_references column to knowledge_vault table
DO $$
BEGIN
    IF NOT column_exists('knowledge_vault', 'raw_memory_references') THEN
        ALTER TABLE knowledge_vault ADD COLUMN raw_memory_references JSONB DEFAULT '[]'::jsonb NOT NULL;
        RAISE NOTICE '✓ Added raw_memory_references column to knowledge_vault table';
    ELSE
        RAISE NOTICE '✓ Skipped: raw_memory_references column already exists in knowledge_vault table';
    END IF;
END;
$$;

-- Verification: Check that all required tables and columns exist
DO $$
DECLARE
    table_name text;
    column_name text;
    required_columns text[][] := ARRAY[
        ARRAY['episodic_memory', 'raw_memory_references'],
        ARRAY['semantic_memory', 'raw_memory_references'],
        ARRAY['procedural_memory', 'raw_memory_references'],
        ARRAY['resource_memory', 'raw_memory_references'],
        ARRAY['knowledge_vault', 'raw_memory_references']
    ];
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '🔍 Verifying migration...';

    -- Check that raw_memory table exists
    IF table_exists('raw_memory') THEN
        RAISE NOTICE '✓ raw_memory table exists';
    ELSE
        RAISE NOTICE '❌ raw_memory table missing';
    END IF;

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
END;
$$;

-- Clean up temporary functions
DROP FUNCTION IF EXISTS column_exists(text, text);
DROP FUNCTION IF EXISTS table_exists(text);

-- Commit the transaction
COMMIT;

-- Final success message
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '✅ PostgreSQL migration for raw_memory completed successfully!';
    RAISE NOTICE 'All schema changes have been applied and verified.';
END;
$$;
