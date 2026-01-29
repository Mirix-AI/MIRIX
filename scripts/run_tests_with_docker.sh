#!/bin/bash
# ============================================================================
# Run tests with Docker/Podman Compose test infrastructure
# ============================================================================
# This script:
#   1. Starts ephemeral PostgreSQL and Redis containers
#   2. Optionally starts test API server container (with --integration)
#   3. Runs pytest with test database configuration
#   4. Stops containers when done (even on failure)
#
# Usage:
#   ./scripts/run_tests_with_docker.sh                    # Run all tests (uses Docker)
#   ./scripts/run_tests_with_docker.sh --podman           # Use Podman instead
#   ./scripts/run_tests_with_docker.sh --integration      # Start server for integration tests
#   ./scripts/run_tests_with_docker.sh test_raw_memory.py  # Run specific test file
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.test.yml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
USE_PODMAN=false
START_SERVER=false
SERVER_PORT=8000
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --podman)
            USE_PODMAN=true
            shift
            ;;
        --integration|--with-server)
            START_SERVER=true
            shift
            ;;
        --server-port)
            START_SERVER=true
            SERVER_PORT="$2"
            shift 2
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# Detect compose command
if [ "$USE_PODMAN" = true ]; then
    if command -v podman-compose &> /dev/null; then
        COMPOSE_CMD="podman-compose"
    elif command -v podman &> /dev/null && podman compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="podman compose"
    else
        echo -e "${RED}Error: podman-compose not found. Install with: pip install podman-compose${NC}"
        exit 1
    fi
else
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    elif command -v docker &> /dev/null && docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        echo -e "${RED}Error: docker-compose not found${NC}"
        exit 1
    fi
fi

# Helper to check container health
check_health() {
    local service=$1
    if [ "$USE_PODMAN" = true ]; then
        # Podman doesn't expose Health in ps, use inspect instead
        podman inspect mirix_test_$service --format "{{.State.Health.Status}}" 2>/dev/null | grep -q "healthy"
    else
        $COMPOSE_CMD -f "$COMPOSE_FILE" ps $service 2>/dev/null | grep -q "healthy"
    fi
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    cd "$PROJECT_ROOT"
    $COMPOSE_CMD -f "$COMPOSE_FILE" down 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start infrastructure
echo -e "${GREEN}Starting test infrastructure...${NC}"
cd "$PROJECT_ROOT"

# Clean up any existing containers
$COMPOSE_CMD -f "$COMPOSE_FILE" down 2>/dev/null || true

# Start database and Redis
echo -e "${YELLOW}Starting database and Redis...${NC}"
$COMPOSE_CMD -f "$COMPOSE_FILE" up -d test_db test_redis

# Wait for database to be healthy
echo -e "${YELLOW}Waiting for database...${NC}"
timeout=60
elapsed=0

while [ $elapsed -lt $timeout ]; do
    if check_health "db"; then
        echo -e "${GREEN}✓ Database ready${NC}"
        break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done

if [ $elapsed -eq $timeout ]; then
    echo -e "${RED}Error: Database failed to start${NC}"
    $COMPOSE_CMD -f "$COMPOSE_FILE" logs test_db | tail -20
    exit 1
fi

# Set test environment variables
export MIRIX_PG_URI="postgresql+pg8000://test:test@localhost:5433/mirix_test"
export MIRIX_REDIS_ENABLED="true"
export MIRIX_REDIS_HOST="localhost"
export MIRIX_REDIS_PORT="6380"
export MIRIX_LANGFUSE_ENABLED="false"

# Start server if requested
if [ "$START_SERVER" = true ]; then
    echo -e "${GREEN}Starting test server on port $SERVER_PORT...${NC}"
    export SERVER_PORT
    
    # Check if port is in use
    if command -v lsof &> /dev/null && lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}Error: Port $SERVER_PORT is already in use${NC}"
        exit 1
    fi
    
    # Start server container
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d test_server
    
    # Wait for server health check
    echo -e "${YELLOW}Waiting for server...${NC}"
    timeout=90
    elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        # Check container health status
        if check_health "server"; then
            echo -e "${GREEN}✓ Server ready${NC}"
            break
        fi
        
        # Check if container exited
        if [ "$USE_PODMAN" = true ]; then
            if podman ps -a --filter "name=mirix_test_server" --format "{{.Status}}" | grep -q "Exited"; then
                echo -e "${RED}Error: Server container exited${NC}"
                podman logs mirix_test_server | tail -30
                exit 1
            fi
        else
            if $COMPOSE_CMD -f "$COMPOSE_FILE" ps test_server | grep -q "Exited"; then
                echo -e "${RED}Error: Server container exited${NC}"
                $COMPOSE_CMD -f "$COMPOSE_FILE" logs test_server | tail -30
                exit 1
            fi
        fi
        
        # Try HTTP health check
        if command -v curl &> /dev/null && curl -f -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server ready${NC}"
            break
        fi
        
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    if [ $elapsed -ge $timeout ]; then
        echo -e "${RED}Error: Server failed to start${NC}"
        if [ "$USE_PODMAN" = true ]; then
            podman logs mirix_test_server | tail -30
        else
            $COMPOSE_CMD -f "$COMPOSE_FILE" logs test_server | tail -30
        fi
        exit 1
    fi
    
    export MIRIX_API_URL="http://localhost:$SERVER_PORT"
fi

# Run tests
echo -e "${GREEN}Running tests...${NC}"
cd "$PROJECT_ROOT"

if command -v poetry &> /dev/null && [ -f "pyproject.toml" ]; then
    PYTEST_CMD="poetry run pytest"
else
    PYTEST_CMD="pytest"
fi

if [ ${#PYTEST_ARGS[@]} -eq 0 ]; then
    $PYTEST_CMD tests/ -v
else
    $PYTEST_CMD "${PYTEST_ARGS[@]}"
fi
