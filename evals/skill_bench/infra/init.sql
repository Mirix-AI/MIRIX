-- skill_bench Postgres bootstrap. Runs once, via psql, when the volume is
-- initialised (docker-entrypoint-initdb.d). It builds skill_bench_template, which
-- carries every extension MIRIX needs; each cell attempt is then created with
--   CREATE DATABASE "<cell db>" TEMPLATE skill_bench_template
-- by infra/mirix_server.py:ensure_cell_database (one database per cell attempt).
CREATE DATABASE skill_bench_template;
\connect skill_bench_template
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
\connect postgres
-- No sessions on the template: a connected template makes CREATE DATABASE ... TEMPLATE fail.
ALTER DATABASE skill_bench_template WITH IS_TEMPLATE true ALLOW_CONNECTIONS false;
