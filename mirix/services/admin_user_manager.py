"""
Client Authentication Manager for dashboard authentication.

Handles client dashboard login, authentication, and JWT token management.
Clients can optionally have email/password for dashboard access.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import bcrypt

try:
    # PyJWT provides the encode/decode helpers we rely on
    import jwt
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("PyJWT is required for dashboard authentication. Install with `pip install PyJWT`.") from exc

# Some environments may have an incompatible `jwt` package installed; validate it early
if not hasattr(jwt, "encode") or not hasattr(jwt, "decode"):  # pragma: no cover - import guard
    raise ImportError(
        "Incompatible `jwt` package detected. Install PyJWT (e.g. `pip install PyJWT>=2.10.1`) "
        "and remove conflicting jwt packages."
    )

from mirix.log import get_logger
from mirix.orm.client import Client as ClientModel
from mirix.orm.errors import NoResultFound
from mirix.schemas.client import Client as PydanticClient
from mirix.settings import settings
from mirix.utils import enforce_types

logger = get_logger(__name__)


def _get_or_create_jwt_secret() -> str:
    """
    Get or create the JWT secret key.

    Priority:
    1. Environment variable MIRIX_JWT_SECRET_KEY (if set)
    2. Persisted secret in ~/.mirix/jwt_secret (if exists)
    3. Generate new secret and persist to ~/.mirix/jwt_secret

    This ensures tokens remain valid across server restarts without
    requiring manual configuration.
    """
    from pathlib import Path

    # 1. Check environment variable first
    if settings.jwt_secret_key:
        logger.debug("Using JWT secret from MIRIX_JWT_SECRET_KEY environment variable")
        return settings.jwt_secret_key

    # 2. Check for persisted secret file
    mirix_dir = settings.mirix_dir or Path.home() / ".mirix"
    secret_file = mirix_dir / "jwt_secret"

    if secret_file.exists():
        try:
            secret = secret_file.read_text().strip()
            if secret and len(secret) >= 32:
                logger.debug("Using persisted JWT secret from %s", secret_file)
                return secret
        except Exception as e:
            logger.warning("Failed to read JWT secret file: %s", e)

    # 3. Generate new secret and persist it
    new_secret = secrets.token_hex(32)

    try:
        # Ensure directory exists
        mirix_dir.mkdir(parents=True, exist_ok=True)

        # Write secret with restricted permissions
        secret_file.write_text(new_secret)

        # On Unix, set file permissions to owner-only (600)
        try:
            import os

            os.chmod(secret_file, 0o600)
        except (OSError, AttributeError):
            pass  # Windows doesn't support chmod the same way

        logger.info("Generated and persisted new JWT secret to %s", secret_file)
    except Exception as e:
        logger.warning(
            "Could not persist JWT secret to %s: %s. " "Sessions may be invalidated on server restart.", secret_file, e
        )

    return new_secret


# JWT Configuration
JWT_SECRET_KEY = _get_or_create_jwt_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = settings.jwt_expiration_hours


class ClientAuthManager:
    """Manager class to handle client dashboard authentication."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    # =========================================================================
    # Password Hashing
    # =========================================================================

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except Exception as e:
            logger.warning("Password verification failed: %s", e)
            return False

    # =========================================================================
    # JWT Token Management
    # =========================================================================

    @staticmethod
    def get_admin_user_id_for_client(client_id: str) -> str:
        """Get the admin user ID associated with a client."""
        return f"user-{client_id.replace('client-', '')}"

    @staticmethod
    def create_access_token(client: PydanticClient, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token for a client."""
        if expires_delta is None:
            expires_delta = timedelta(hours=JWT_EXPIRATION_HOURS)

        expire = datetime.now(timezone.utc) + expires_delta

        # Include admin user_id in token for memory operations
        admin_user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)

        payload = {
            "sub": client.id,
            "name": client.name,
            "email": client.email,
            "scope": client.scope,
            "admin_user_id": admin_user_id,  # For memory operations
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }

        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    @staticmethod
    def decode_access_token(token: str) -> Optional[dict]:
        """Decode and validate a JWT access token."""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token: %s", e)
            return None

    # =========================================================================
    # Client Dashboard Authentication
    # =========================================================================

    @enforce_types
    def register_client_for_dashboard(
        self,
        name: str,
        email: str,
        password: str,
        organization_id: Optional[str] = None,
        scope: str = "admin",
    ) -> PydanticClient:
        """
        Register a new client with dashboard login credentials.

        Also creates an admin user for this client (for memory operations).

        Args:
            name: The client name
            email: Email for dashboard login
            password: Password for dashboard login
            organization_id: Optional organization ID
            scope: Client scope (default: admin for dashboard users)

        Returns:
            The created client

        Raises:
            ValueError: If email already exists
        """
        import uuid

        from mirix.orm.user import User as UserModel
        from mirix.services.organization_manager import OrganizationManager

        org_id = organization_id or OrganizationManager.DEFAULT_ORG_ID

        with self.session_maker() as session:
            # Check if email already exists
            existing_email = (
                session.query(ClientModel)
                .filter(ClientModel.email == email.lower(), ClientModel.is_deleted == False)
                .first()
            )
            if existing_email:
                raise ValueError(f"Email '{email}' already exists")

            # Hash the password
            password_hash = self.hash_password(password)

            # Generate client ID
            client_id = f"client-{uuid.uuid4().hex[:8]}"

            # Create client with dashboard credentials
            client = ClientModel(
                id=client_id,
                name=name,
                email=email.lower(),
                password_hash=password_hash,
                status="active",
                scope=scope,
                organization_id=org_id,
            )

            client.create(session)
            logger.info("Registered client for dashboard: %s (%s)", client.name, client.email)

            # Create an admin user for this client
            # User ID is derived from client ID for easy association
            admin_user_id = f"user-{client.id.replace('client-', '')}"

            try:
                # Check if user already exists
                existing_user = session.query(UserModel).filter(UserModel.id == admin_user_id).first()

                if not existing_user:
                    admin_user = UserModel(
                        id=admin_user_id,
                        name=f"Admin",
                        status="active",
                        timezone="UTC",
                        organization_id=org_id,
                        client_id=client.id,  # Link user to client
                        is_admin=True,  # Mark as admin user
                    )
                    admin_user.create(session)
                    logger.info("Created admin user for client: %s -> %s", client.id, admin_user_id)
            except Exception as e:
                logger.warning("Failed to create admin user for client %s: %s", client.id, e)
                # Don't fail client registration if user creation fails

            return client.to_pydantic()

    @enforce_types
    def authenticate(self, email: str, password: str) -> Tuple[Optional[PydanticClient], Optional[str], str]:
        """
        Authenticate a client for dashboard access and return client + JWT token.

        Args:
            email: The email address
            password: The password

        Returns:
            Tuple of (client, access_token, status)
            status can be:
                - "ok": authentication successful
                - "not_found": email not found
                - "inactive": client is not active
                - "no_password": client has no password set
                - "wrong_password": password mismatch
        """
        with self.session_maker() as session:
            client = (
                session.query(ClientModel)
                .filter(ClientModel.email == email.lower(), ClientModel.is_deleted == False)
                .first()
            )

            if not client:
                logger.warning("Login attempt for non-existent email: %s", email)
                return None, None, "not_found"

            if client.status != "active":
                logger.warning("Login attempt for inactive client: %s", email)
                return None, None, "inactive"

            if not client.password_hash:
                logger.warning("Login attempt for client without password: %s", email)
                return None, None, "no_password"

            if not self.verify_password(password, client.password_hash):
                logger.warning("Failed login attempt for client: %s", email)
                return None, None, "wrong_password"

            # Update last login time
            client.last_login = datetime.now(timezone.utc)
            client.update(session, actor=None)

            pydantic_client = client.to_pydantic()
            access_token = self.create_access_token(pydantic_client)

            logger.info("Successful dashboard login for client: %s", email)
            return pydantic_client, access_token, "ok"

    @enforce_types
    def get_client_by_id(self, client_id: str) -> Optional[PydanticClient]:
        """Get a client by ID."""
        with self.session_maker() as session:
            try:
                client = ClientModel.read(db_session=session, identifier=client_id)
                if client.is_deleted:
                    return None
                return client.to_pydantic()
            except NoResultFound:
                return None

    @enforce_types
    def get_client_by_email(self, email: str) -> Optional[PydanticClient]:
        """Get a client by email."""
        with self.session_maker() as session:
            client = (
                session.query(ClientModel)
                .filter(ClientModel.email == email.lower(), ClientModel.is_deleted == False)
                .first()
            )
            if client:
                return client.to_pydantic()
            return None

    @enforce_types
    def list_dashboard_clients(self, cursor: Optional[str] = None, limit: int = 50) -> List[PydanticClient]:
        """List all clients that have dashboard access (email set)."""
        with self.session_maker() as session:
            query = (
                session.query(ClientModel)
                .filter(ClientModel.is_deleted == False, ClientModel.email.isnot(None))
                .order_by(ClientModel.created_at.desc())
            )

            if cursor:
                query = query.filter(ClientModel.id < cursor)

            if limit:
                query = query.limit(limit)

            clients = query.all()
            return [client.to_pydantic() for client in clients]

    @enforce_types
    def set_client_password(self, client_id: str, email: str, password: str) -> PydanticClient:
        """
        Set dashboard credentials for an existing client.

        Args:
            client_id: The client ID
            email: Email for dashboard login
            password: Password for dashboard login

        Returns:
            Updated client
        """
        with self.session_maker() as session:
            client = ClientModel.read(db_session=session, identifier=client_id)

            if client.is_deleted:
                raise ValueError("Cannot update deleted client")

            # Check if email already exists on another client
            existing_email = (
                session.query(ClientModel)
                .filter(
                    ClientModel.email == email.lower(), ClientModel.id != client_id, ClientModel.is_deleted == False
                )
                .first()
            )
            if existing_email:
                raise ValueError(f"Email '{email}' already exists on another client")

            client.email = email.lower()
            client.password_hash = self.hash_password(password)
            client.update(session, actor=None)

            logger.info("Set dashboard credentials for client: %s", client.name)
            return client.to_pydantic()

    @enforce_types
    def change_password(self, client_id: str, current_password: str, new_password: str) -> bool:
        """
        Change a client's dashboard password.

        Args:
            client_id: The client ID
            current_password: The current password
            new_password: The new password

        Returns:
            True if successful, False otherwise
        """
        with self.session_maker() as session:
            client = ClientModel.read(db_session=session, identifier=client_id)

            if not client.password_hash:
                logger.warning("Password change failed: client has no password set: %s", client_id)
                return False

            if not self.verify_password(current_password, client.password_hash):
                logger.warning("Password change failed: incorrect current password for %s", client_id)
                return False

            client.password_hash = self.hash_password(new_password)
            client.update(session, actor=None)

            logger.info("Password changed for client: %s", client.name)
            return True

    @enforce_types
    def count_dashboard_clients(self) -> int:
        """Count total clients with dashboard access."""
        with self.session_maker() as session:
            return (
                session.query(ClientModel)
                .filter(ClientModel.is_deleted == False, ClientModel.email.isnot(None))
                .count()
            )

    @enforce_types
    def is_first_dashboard_user(self) -> bool:
        """Check if there are no dashboard users yet (for bootstrap)."""
        return self.count_dashboard_clients() == 0

    # =========================================================================
    # Scope-based Authorization
    # =========================================================================

    @staticmethod
    def can_manage_clients(scope: str) -> bool:
        """Check if a scope can manage clients and API keys."""
        return scope == "admin"

    @staticmethod
    def can_manage_memories(scope: str) -> bool:
        """Check if a scope can modify memories."""
        return scope in ["admin", "read_write"]

    @staticmethod
    def can_view_dashboard(scope: str) -> bool:
        """Check if a scope can view the dashboard."""
        return scope in ["admin", "read_write", "read"]


# Backward compatibility alias
AdminUserManager = ClientAuthManager
