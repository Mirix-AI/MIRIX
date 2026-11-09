#!/bin/sh
set -eu

: "${MIRIX_REDIS_PASSWORD:?Set MIRIX_REDIS_PASSWORD}"
# Redis configuration strings allow escaping quotes and backslashes, but a
# literal newline would start another directive. Never interpolate a password
# into REDIS_ARGS, which the upstream entrypoint splits as shell words.
case "$MIRIX_REDIS_PASSWORD" in
    *'
'*|*''*) echo "MIRIX_REDIS_PASSWORD must not contain line breaks" >&2; exit 1 ;;
esac
umask 077
escaped_password=$(printf '%s' "$MIRIX_REDIS_PASSWORD" | sed 's/\\/\\\\/g; s/"/\\"/g')
printf 'requirepass "%s"\n' "$escaped_password" > /redis-stack.conf
unset escaped_password
export REDIS_ARGS="--protected-mode yes --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru"
exec /entrypoint.sh
