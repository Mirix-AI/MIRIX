# Mirix Client Build Notes

This documents the clean steps used to produce the lean `mirix_client-0.1.0` wheel.

## Prereqs
- Version is fixed at `0.1.0` in `mirix/__init__.py` and `scripts/packaging/setup_client.py`.
- Dependencies come from `scripts/packaging/requirements_client.txt` (client-only).

## Build Steps
1) (Optional but recommended) Temporarily move the project-level `pyproject.toml` so PEP 621 metadata does not override the client setup script:
   - `Move-Item pyproject.toml pyproject.toml.backup_client`
2) Clean previous artifacts:
   - `Remove-Item -Recurse -Force build, dist, mirix_client.egg-info`
3) Build the client package from the repo root:
   - `python scripts/packaging/setup_client.py sdist bdist_wheel`
4) Restore `pyproject.toml` if you moved it:
   - `Move-Item pyproject.toml.backup_client pyproject.toml`

Output artifacts land in `dist/`:
- `dist/mirix_client-0.1.0-py3-none-any.whl`
- `dist/mirix_client-0.1.0.tar.gz`

## Quick Install Test
```powershell
python -m venv client_env
.\client_env\Scripts\activate
pip install --upgrade pip
pip install dist\mirix_client-0.1.0-py3-none-any.whl
python - <<'PY'
from mirix import MirixClient
print("MirixClient import ok", MirixClient)
PY
```
