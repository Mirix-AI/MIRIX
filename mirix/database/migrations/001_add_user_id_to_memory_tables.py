"""
Migration: Add user_id columns to memory tables for multi-user suppor

This migration adds nullable user_id columns to all memory tables to suppor
multi-user functionality while maintaining backwards compatibility.

Tables modified:
- messages
- episodic_memory
- semantic_memory
- procedural_memory
- resource_memory
- knowledge_vaul
- block

All user_id columns are nullable to ensure backwards compatibility with existing data.
"""

from sqlalchemy import text

from mirix.settings import settings


def upgrade_postgresql(connection):
    """Apply migration for PostgreSQL database."""

    # Default user ID for existing data
    DEFAULT_USER_ID = "user-00000000-0000-4000-8000-000000000000"

    # Add user_id columns to all memory tables
    tables_to_modify = [
        "messages",
        "episodic_memory",
        "semantic_memory",
        "procedural_memory",
        "resource_memory",
        "knowledge_vault",
        "block",
    ]

    for table in tables_to_modify:
        # Add user_id column (nullable initially)
        connection.execute(
            text(
                f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);
        """
            )
        )

        # Backfill existing data with default user
        connection.execute(
            text(
                f"""
            UPDATE {table}
            SET user_id = '{DEFAULT_USER_ID}'
            WHERE user_id IS NULL;
        """
            )
        )

        # Make column NOT NULL after backfill
        connection.execute(
            text(
                f"""
            ALTER TABLE {table}
            ALTER COLUMN user_id SET NOT NULL;
        """
            )
        )

        # Add foreign key constraint
        connection.execute(
            text(
                f"""
            ALTER TABLE {table}
            ADD CONSTRAINT IF NOT EXISTS fk_{table}_user_id
            FOREIGN KEY (user_id) REFERENCES users(id);
        """
            )
        )

        # Add composite index for performance
        connection.execute(
            text(
                f"""
            CREATE INDEX IF NOT EXISTS ix_{table}_organization_user
            ON {table} (organization_id, user_id);
        """
            )
        )

    print("PostgreSQL migration completed: Added user_id columns, backfilled with default user, added constraints")


def upgrade_sqlite(connection):
    """Apply migration for SQLite database."""

    # Default user ID for existing data
    DEFAULT_USER_ID = "user-00000000-0000-4000-8000-000000000000"

    # SQLite doesn't support ADD CONSTRAINT for foreign keys on existing tables
    # We'll add the columns without foreign key constraints for SQLite

    tables_to_modify = [
        "messages",
        "episodic_memory",
        "semantic_memory",
        "procedural_memory",
        "resource_memory",
        "knowledge_vault",
        "block",
    ]

    for table in tables_to_modify:
        try:
            # Add user_id column (SQLite will ignore if already exists)
            connection.execute(
                text(
                    f"""
                ALTER TABLE {table} 
                ADD COLUMN user_id VARCHAR(255);
            """
                )
            )

            # Backfill existing data with default user
            connection.execute(
                text(
                    f"""
                UPDATE {table} 
                SET user_id = '{DEFAULT_USER_ID}' 
                WHERE user_id IS NULL OR user_id = '';
            """
                )
            )

            # Add index for performance
            connection.execute(
                text(
                    f"""
                CREATE INDEX IF NOT EXISTS ix_{table}_organization_user
                ON {table} (organization_id, user_id);
            """
                )
            )

            print(f"Added user_id column, backfilled, and indexed {table}")

        except Exception as e:
            # Column might already exist, which is fine
            if "duplicate column name" in str(e).lower():
                print(f"Column user_id already exists in {table}")
                # Still try to backfill in case it was added but not populated
                try:
                    connection.execute(
                        text(
                            f"""
                        UPDATE {table} 
                        SET user_id = '{DEFAULT_USER_ID}' 
                        WHERE user_id IS NULL OR user_id = '';
                    """
                        )
                    )
                    print(f"Backfilled existing data in {table}")
                except Exception as backfill_error:
                    print(f"Warning: Could not backfill {table}: {backfill_error}")
            else:
                raise e

    print("SQLite migration completed: Added user_id columns, backfilled with default user, added indexes")


def upgrade():
    """
    Apply the migration based on the database type.
    """
    from mirix.server.server import db_context

    with db_context() as session:
        connection = session.connection()

        # Determine database type and apply appropriate migration
        if settings.mirix_pg_uri_no_default:
            upgrade_postgresql(connection)
        else:
            upgrade_sqlite(connection)

        # Commit the changes
        session.commit()
        print("Migration 001_add_user_id_to_memory_tables completed successfully")


def downgrade():
    """
    Rollback the migration (remove user_id columns).

    WARNING: This will permanently delete user_id data!
    """
    from mirix.server.server import db_context

    print("WARNING: Downgrade will permanently delete user_id data!")
    response = input("Are you sure you want to continue? (yes/no): ")

    if response.lower() != "yes":
        print("Downgrade cancelled")
        return

    with db_context() as session:
        connection = session.connection()

        tables_to_modify = [
            "messages",
            "episodic_memory",
            "semantic_memory",
            "procedural_memory",
            "resource_memory",
            "knowledge_vault",
            "block",
        ]

        if settings.mirix_pg_uri_no_default:
            # PostgreSQL downgrade
            for table in tables_to_modify:
                # Drop foreign key constraint firs
                connection.execute(
                    text(
                        f"""
                    ALTER TABLE {table}
                    DROP CONSTRAINT IF EXISTS fk_{table}_user_id;
                """
                    )
                )

                # Drop index
                connection.execute(
                    text(
                        f"""
                    DROP INDEX IF EXISTS ix_{table}_organization_user;
                """
                    )
                )

                # Drop column
                connection.execute(
                    text(
                        f"""
                    ALTER TABLE {table}
                    DROP COLUMN IF EXISTS user_id;
                """
                    )
                )
        else:
            # SQLite downgrade (more complex due to limited ALTER TABLE support)
            print("SQLite downgrade requires table recreation - not implemented for safety")
            print("Recommend backing up data and recreating database if downgrade needed")
            return

        session.commit()
        print("Downgrade completed: Removed user_id columns")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 001_add_user_id_to_memory_tables.py [upgrade|downgrade]")
        sys.exit(1)

    action = sys.argv[1].lower()

    if action == "upgrade":
        upgrade()
    elif action == "downgrade":
        downgrade()
    else:
        print("Invalid action. Use 'upgrade' or 'downgrade'")
        sys.exit(1)
