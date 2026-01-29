"""
Schema extraction utility for PGlite initialization
This module extracts the database schema from SQLAlchemy models and generates DDL
"""

import os
import sys

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from mirix.log import get_logger

logger = get_logger(__name__)

# Add the mirix directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def extract_schema_ddl():
    """Extract DDL statements from SQLAlchemy models for PGlite"""

    # Import all the ORM models to ensure they're registered
    from mirix.orm.sqlalchemy_base import SqlalchemyBase

    # Get all tables from the base metadata

    metadata = SqlalchemyBase.metadata

    ddl_statements = []

    # Generate CREATE TABLE statements
    for table in metadata.sorted_tables:
        # Skip tables that are for PostgreSQL-specific features we don't need
        if table.name in ["alembic_version"]:
            continue

        create_table_ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))

        # Clean up the DDL for PGlite compatibility
        create_table_ddl = clean_ddl_for_pglite(create_table_ddl)

        ddl_statements.append(create_table_ddl)

    # Add any additional setup statements
    setup_statements = [
        "-- PGlite Schema Initialization",
        "-- Generated from SQLAlchemy models",
        "",
        "-- Create basic sequences that might be needed",
        "-- (PGlite should handle SERIAL columns automatically)",
        "",
    ]

    return "\n".join(setup_statements + ddl_statements)


def clean_ddl_for_pglite(ddl):
    """Clean DDL statements for PGlite compatibility"""

    # Remove PostgreSQL-specific extensions and features that PGlite doesn't support
    replacements = [
        # Remove vector column types (PGlite doesn't support pgvector)
        ("vector(1536)", "TEXT"),  # Replace vector columns with TEXT for now
        ("vector(3072)", "TEXT"),
        ("vector(4096)", "TEXT"),
        # Remove GIN indexes (not supported in PGlite)
        ("USING gin", ""),
        # Simplify constraint names
        ("CONSTRAINT ", ""),
        # Remove some PostgreSQL-specific column constraints
        ("::text", ""),
        # Remove timezone from TIMESTAMP
        ("TIMESTAMP WITH TIME ZONE", "TIMESTAMP"),
        # Simplify SERIAL to INTEGER with autoincrement
        ("BIGSERIAL", "INTEGER"),
        ("SERIAL", "INTEGER"),
    ]

    for old, new in replacements:
        ddl = ddl.replace(old, new)

    # Remove any lines that contain unsupported PostgreSQL features
    lines = ddl.split("\n")
    filtered_lines = []

    for line in lines:
        # Skip lines with unsupported features
        if any(
            unsupported in line.lower()
            for unsupported in [
                "gin",
                "gist",
                "tsvector",
                "to_tsvector",
                "pg_trgm",
                "btree_gin",
                "btree_gist",
            ]
        ):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def get_basic_schema():
    """Get a basic schema for PGlite with essential tables"""

    return """
-- Basic PGlite Schema for Mirix
-- Essential tables for core functionality

-- Organizations table
CREATE TABLE organizations (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
);

-- Users table
CREATE TABLE users (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    timezone VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Agents table
CREATE TABLE agents (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description TEXT,
    memory TEXT,
    tools TEXT,
    agent_type VARCHAR DEFAULT 'mirix_agent',
    llm_config TEXT,
    embedding_config TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Messages table
CREATE TABLE messages (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    agent_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    text TEXT,
    content TEXT,
    model VARCHAR,
    name VARCHAR,
    tool_calls TEXT,
    tool_call_id VARCHAR,
    tool_return TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id),
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

-- Tools table
CREATE TABLE tools (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    tool_type VARCHAR DEFAULT 'custom',
    return_char_limit INTEGER,
    description TEXT,
    tags TEXT,
    source_type VARCHAR DEFAULT 'json',
    source_code TEXT,
    json_schema TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Blocks table (for core memory)
CREATE TABLE blocks (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    label VARCHAR NOT NULL,
    value TEXT NOT NULL,
    char_limit INTEGER DEFAULT 2000,
    metadata_ TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Providers table
CREATE TABLE providers (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    api_key VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Knowledge vault table
CREATE TABLE knowledge_vault (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    entry_type VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    sensitivity VARCHAR NOT NULL,
    secret_value TEXT NOT NULL,
    metadata_ TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Semantic memory table
CREATE TABLE semantic_memory (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    summary TEXT NOT NULL,
    details TEXT NOT NULL,
    source VARCHAR,
    metadata_ TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Episodic memory table
CREATE TABLE episodic_memory (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    occurred_at TIMESTAMP NOT NULL,
    last_modify TEXT NOT NULL,
    actor VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    summary VARCHAR NOT NULL,
    details TEXT NOT NULL,
    metadata_ TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Procedural memory table
CREATE TABLE procedural_memory (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    entry_type VARCHAR NOT NULL,
    summary VARCHAR NOT NULL,
    steps TEXT NOT NULL,
    last_modify TEXT NOT NULL,
    metadata_ TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Resource memory table
CREATE TABLE resource_memory (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    summary VARCHAR NOT NULL,
    resource_type VARCHAR NOT NULL,
    content TEXT NOT NULL,
    last_modify TEXT NOT NULL,
    metadata_ TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Files table
CREATE TABLE files (
    id VARCHAR PRIMARY KEY,
    organization_id VARCHAR NOT NULL,
    source_id VARCHAR,
    file_name VARCHAR,
    file_path VARCHAR,
    source_url VARCHAR,
    google_cloud_url VARCHAR,
    file_type VARCHAR,
    file_size INTEGER,
    file_creation_date VARCHAR,
    file_last_modified_date VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- Create indexes for better performance
CREATE INDEX idx_messages_agent_created_at ON messages(agent_id, created_at);
CREATE INDEX idx_messages_created_at ON messages(created_at, id);
CREATE INDEX idx_agents_organization ON agents(organization_id);
CREATE INDEX idx_tools_organization ON tools(organization_id);
CREATE INDEX idx_blocks_organization ON blocks(organization_id);
CREATE INDEX idx_semantic_memory_organization ON semantic_memory(organization_id);
CREATE INDEX idx_episodic_memory_organization ON episodic_memory(organization_id);
CREATE INDEX idx_procedural_memory_organization ON procedural_memory(organization_id);
CREATE INDEX idx_resource_memory_organization ON resource_memory(organization_id);
CREATE INDEX idx_knowledge_vault_organization ON knowledge_vault(organization_id);
CREATE INDEX idx_files_organization ON files(organization_id);

-- Insert default organization if not exists
INSERT INTO organizations (id, name) 
VALUES ('default-org', 'Default Organization') 
ON CONFLICT DO NOTHING;

-- Insert default user if not exists
INSERT INTO users (id, organization_id, name, timezone) 
VALUES ('default-user', 'default-org', 'Default User', 'UTC') 
ON CONFLICT DO NOTHING;
"""


if __name__ == "__main__":
    try:
        # Try to extract full schema from SQLAlchemy models
        schema = extract_schema_ddl()
        logger.debug("Full schema extracted successfully:")
        logger.debug(schema)
    except Exception as e:
        logger.debug("Failed to extract full schema: %s", e)
        logger.debug("Using basic schema instead:")
        logger.debug(get_basic_schema())
