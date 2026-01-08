-- ============================================================================
-- Mirix PostgreSQL Initialization Script
-- ============================================================================
-- This script runs automatically on first database startup.
-- It sets up the required extensions and initial configuration.
-- ============================================================================

-- Enable pgvector extension for vector similarity search
-- This is required for storing and querying embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable additional useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";     -- UUID generation functions
CREATE EXTENSION IF NOT EXISTS "pg_trgm";       -- Trigram matching for fuzzy text search

-- Verify extensions are installed
DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Mirix PostgreSQL Initialization Complete';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Extensions enabled:';
    RAISE NOTICE '  ✓ vector (pgvector) - Vector similarity search';
    RAISE NOTICE '  ✓ uuid-ossp - UUID generation';
    RAISE NOTICE '  ✓ pg_trgm - Trigram text matching';
    RAISE NOTICE '============================================';
END;
$$;

-- Display installed extensions
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm');

