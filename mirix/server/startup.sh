#!/bin/sh
set -e  # Exit on any error

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8283}"

# Function to wait for PostgreSQL to be ready
wait_for_postgres() {
    until pg_isready -U "${POSTGRES_USER:-mirix}" -h localhost; do
        echo "Waiting for PostgreSQL to be ready..."
        sleep 2
    done
}

# Check if we're configured for external Postgres
if [ -n "${MIRIX_PG_URI:-}" ]; then
    echo "External Postgres configuration detected"
else
    : "${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD for the internal database}"
    echo "No external Postgres configuration detected, starting internal PostgreSQL..."
    # Start PostgreSQL using the base image's entrypoint script
    /usr/local/bin/docker-entrypoint.sh postgres -c listen_addresses=127.0.0.1 &

    # Wait for PostgreSQL to be ready
    wait_for_postgres

    # Set default connection URI for internal postgres
    MIRIX_PG_URI=$(python3 -c 'import os; from urllib.parse import quote; print("postgresql://{}:{}@127.0.0.1:5432/{}".format(quote(os.environ.get("POSTGRES_USER", "mirix"), safe=""), quote(os.environ["POSTGRES_PASSWORD"], safe=""), quote(os.environ.get("POSTGRES_DB", "mirix"), safe="")))')
    export MIRIX_PG_URI
    echo "Using internal PostgreSQL on loopback"
fi

# Attempt database migration
echo "Attempting to migrate database..."
if ! alembic upgrade head; then
    echo "ERROR: Database migration failed!"
    echo "Please check your database connection and try again."
    echo "If the problem persists, check the logs for more details."
    exit 1
fi
echo "Database migration completed successfully."

# If ADE is enabled, add the --ade flag to the command
set -- mirix server --host "$HOST" --port "$PORT"
if [ "${SECURE:-false}" = "true" ]; then
    set -- "$@" --secure
fi

echo "Starting Mirix server at http://$HOST:$PORT..."
exec "$@"
