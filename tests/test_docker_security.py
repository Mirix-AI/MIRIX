import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
REQUIRED_SECRETS = ("MIRIX_JWT_SECRET_KEY", "MIRIX_PG_PASSWORD", "MIRIX_REDIS_PASSWORD")


def _compose_env():
    env = {key: value for key, value in os.environ.items() if not key.startswith("MIRIX_")}
    env.update({key: f"test-only-{key}-" + "x" * 32 for key in REQUIRED_SECRETS})
    env["MIRIX_NEO4J_PASSWORD"] = "test-only-neo4j-password"
    return env


def _compose(env, filename="docker-compose.yml"):
    if not shutil.which("docker"):
        pytest.skip("Docker Compose CLI is unavailable")
    result = subprocess.run(
        ["docker", "compose", "-f", str(REPO / filename),
         "--env-file", os.devnull, "--profile", "graph", "config", "--format", "json"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=30,
    )
    if "is not a docker command" in result.stderr:
        pytest.skip("Docker Compose plugin is unavailable")
    return result


@pytest.mark.parametrize("missing", REQUIRED_SECRETS)
def test_compose_fails_without_each_required_secret(missing):
    env = _compose_env()
    env.pop(missing)
    result = _compose(env)
    assert result.returncode != 0
    assert missing in result.stderr


def test_compose_publishes_only_loopback_and_enables_redis_auth():
    result = _compose(_compose_env())
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    for service in services.values():
        for port in service.get("ports", []):
            assert port["host_ip"] == "127.0.0.1"
    assert services["redis"]["environment"]["MIRIX_REDIS_PASSWORD"]
    assert (
        services["redis"]["environment"]["MIRIX_REDIS_PASSWORD"]
        == services["mirix_api"]["environment"]["MIRIX_REDIS_PASSWORD"]
    )


@pytest.mark.parametrize("secret", ["too-short", "change-this-to-a-secure-secret-key"])
def test_server_rejects_insecure_jwt_secrets(monkeypatch, secret):
    from mirix.services import admin_user_manager
    monkeypatch.setattr(admin_user_manager.settings, "jwt_secret_key", secret)
    with pytest.raises(ValueError, match="unique secret"):
        admin_user_manager._get_or_create_jwt_secret()


def test_test_infrastructure_also_publishes_only_loopback():
    env = _compose_env()
    env.update({
        "MIRIX_TEST_PG_PASSWORD": "test-only-postgres-password",
        "MIRIX_TEST_REDIS_PASSWORD": "test-only-redis-password",
        "MIRIX_TEST_JWT_SECRET_KEY": "test-only-jwt-secret-" + "x" * 32,
    })
    result = _compose(env, "docker-compose.test.yml")
    assert result.returncode == 0, result.stderr
    for service in json.loads(result.stdout)["services"].values():
        for port in service.get("ports", []):
            assert port["host_ip"] == "127.0.0.1"
