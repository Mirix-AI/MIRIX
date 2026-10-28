#!/usr/bin/env bash
# Run MIRIX Postgres integration tests against a FRESH, throwaway database.
#
# MIRIX reads MIRIX_PG_URI once, at import time (mirix/settings.py), and the ORM
# picks Postgres-only columns/indexes from it. So every Postgres integration run
# needs its own pytest process with MIRIX_PG_URI exported before import, and —
# so runs never see each other's rows — its own database. This script does both:
#
#   1. creates database mirix_it_<random> on the test server
#      (default 127.0.0.1:5433, user test / password test);
#   2. enables the extensions MIRIX needs: vector, pg_trgm, "uuid-ossp";
#   3. runs `pytest -m integration <your args>` with
#        MIRIX_PG_URI=postgresql+pg8000://<user>:<password>@<host>:<port>/<db>,
#        a throwaway HOME (so ~/.mirix is never read or written), and every
#        model-provider API key unset (no paid calls, ever);
#   4. ALWAYS drops the database afterwards (EXIT trap), even if pytest fails
#      or you hit Ctrl-C, and exits with pytest's exit code.
#
# Usage:
#   scripts/run_pg_integration_tests.sh <pytest paths/args...>
#
# Examples:
#   scripts/run_pg_integration_tests.sh tests/test_pg_integration_smoke.py
#   scripts/run_pg_integration_tests.sh tests/test_skill_data_layer_pg.py -k upgrade -x
#
# Environment overrides:
#   MIRIX_IT_PG_HOST      (default 127.0.0.1)
#   MIRIX_IT_PG_PORT      (default 5433)
#   MIRIX_IT_PG_USER      (default test)      — needs CREATEDB (the docker
#   MIRIX_IT_PG_PASSWORD  (default test)        POSTGRES_USER is a superuser)
#   MIRIX_IT_PG_ADMIN_DB  (default postgres)  — database used to CREATE/DROP
#   PYTHON                (default <repo>/.venv/bin/python, else python3)
#   MIRIX_IT_KEEP_DB=1    keep the database for debugging (prints its URI)
#
# The test server is a pgvector container, e.g.:
#   docker run -d --name mirix_skillbench_test_db -p 127.0.0.1:5433:5432 \
#     -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=mirix_test \
#     pgvector/pgvector:pg16
#
# Database administration goes through the project's own Python + pg8000, so
# the host needs no psql client.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'
    echo "error: pass at least one pytest path or argument" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PG_HOST="${MIRIX_IT_PG_HOST:-127.0.0.1}"
PG_PORT="${MIRIX_IT_PG_PORT:-5433}"
PG_USER="${MIRIX_IT_PG_USER:-test}"
PG_PASSWORD="${MIRIX_IT_PG_PASSWORD:-test}"
PG_ADMIN_DB="${MIRIX_IT_PG_ADMIN_DB:-postgres}"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi

DB_NAME="mirix_it_$("$PYTHON" -c 'import uuid; print(uuid.uuid4().hex[:12])')"
TEST_HOME="$(mktemp -d "${TMPDIR:-/tmp}/mirix_it_home.XXXXXX")"
export MIRIX_IT_PG_HOST="$PG_HOST" MIRIX_IT_PG_PORT="$PG_PORT" \
    MIRIX_IT_PG_USER="$PG_USER" MIRIX_IT_PG_PASSWORD="$PG_PASSWORD"

# pg_admin <create|drop> <database> — CREATE/DROP DATABASE via pg8000. The name
# is generated above from [a-z0-9_] only, and re-validated here before it is
# interpolated into DDL (identifiers cannot be bound as parameters).
pg_admin() {
    "$PYTHON" - "$1" "$2" "$PG_ADMIN_DB" <<'PY'
import os
import re
import sys

import pg8000.native

action, database, admin_db = sys.argv[1], sys.argv[2], sys.argv[3]
if not re.fullmatch(r"mirix_it_[a-z0-9_]{1,50}", database):
    sys.exit(f"refusing to {action} unexpected database name {database!r}")


def connect(dbname):
    return pg8000.native.Connection(
        user=os.environ["MIRIX_IT_PG_USER"],
        password=os.environ["MIRIX_IT_PG_PASSWORD"],
        host=os.environ["MIRIX_IT_PG_HOST"],
        port=int(os.environ["MIRIX_IT_PG_PORT"]),
        database=dbname,
    )


admin = connect(admin_db)
try:
    if action == "create":
        admin.run(f'CREATE DATABASE "{database}"')
    elif action == "drop":
        admin.run(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = :db AND pid <> pg_backend_pid()",
            db=database,
        )
        admin.run(f'DROP DATABASE IF EXISTS "{database}"')
    else:
        sys.exit(f"unknown action {action!r}")
finally:
    admin.close()

if action == "create":
    db = connect(database)
    try:
        for extension in ("vector", "pg_trgm", "uuid-ossp"):
            db.run(f'CREATE EXTENSION IF NOT EXISTS "{extension}"')
    finally:
        db.close()
PY
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "${MIRIX_IT_KEEP_DB:-0}" == "1" ]]; then
        echo "[pg-it] keeping database $DB_NAME (MIRIX_IT_KEEP_DB=1): $PG_URI" >&2
    elif ! pg_admin drop "$DB_NAME"; then
        echo "[pg-it] WARNING: failed to drop database $DB_NAME" >&2
    else
        echo "[pg-it] dropped database $DB_NAME" >&2
    fi
    rm -rf "$TEST_HOME"
    exit "$status"
}

PG_URI="postgresql+pg8000://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:${PG_PORT}/${DB_NAME}"

# Arm the cleanup BEFORE creating, so a half-finished create (database made,
# extension failed) is still dropped. DROP ... IF EXISTS makes that safe.
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
pg_admin create "$DB_NAME"
echo "[pg-it] created database $DB_NAME on $PG_HOST:$PG_PORT" >&2

set +e
HOME="$TEST_HOME" env \
    -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY -u OPENROUTER_API_KEY -u GOOGLE_API_KEY \
    -u MIRIX_PG_DB -u MIRIX_PG_USER -u MIRIX_PG_PASSWORD -u MIRIX_PG_HOST -u MIRIX_PG_PORT \
    MIRIX_PG_URI="$PG_URI" \
    "$PYTHON" -m pytest -p no:cacheprovider -m integration "$@"
status=$?
set -e
exit "$status"
