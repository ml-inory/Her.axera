import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.config import get_settings


class TestUserService:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Point the data dir to a temp path
        import app.services.user_service as mod
        self._orig_data_dir = mod._DATA_DIR
        self._orig_users_file = mod._USERS_FILE
        mod._DATA_DIR = tmp_path
        mod._USERS_FILE = tmp_path / "users.json"
        # Force reload of the user_service module's UserService instance
        from app.services.user_service import UserService
        self.svc = UserService()
        yield
        mod._DATA_DIR = self._orig_data_dir
        mod._USERS_FILE = self._orig_users_file

    def test_create_and_list(self) -> None:
        u = self.svc.create_user("Alice")
        assert u.name == "Alice"
        assert u.role == "user"
        assert u.user_id.startswith("usr_")
        assert len(u.api_key) > 10

        users = self.svc.list_users()
        assert len(users) == 1
        assert users[0].user_id == u.user_id

    def test_create_admin(self) -> None:
        u = self.svc.create_user("Admin", role="admin")
        assert u.role == "admin"

    def test_get_user(self) -> None:
        u = self.svc.create_user("Bob")
        found = self.svc.get_user(u.user_id)
        assert found is not None
        assert found.name == "Bob"

        assert self.svc.get_user("nonexistent") is None

    def test_delete_user(self) -> None:
        u = self.svc.create_user("Charlie")
        assert self.svc.delete_user(u.user_id) is True
        assert self.svc.get_user(u.user_id) is None
        # Double delete returns False
        assert self.svc.delete_user(u.user_id) is False

    def test_validate_api_key(self) -> None:
        u = self.svc.create_user("Dave")
        result = self.svc.validate_api_key(u.api_key)
        assert result is not None
        assert result.user_id == u.user_id

        assert self.svc.validate_api_key("bad_key") is None

    def test_persistence(self, tmp_path: Path) -> None:
        """Verifies users are saved to and loaded from disk."""
        svc1 = self.svc
        u = svc1.create_user("Eve")

        # Create a new service instance that loads from same file
        import importlib
        import app.services.user_service as mod
        importlib.reload(mod)
        from app.services.user_service import UserService
        # Patch the file path before creating instance
        mod._DATA_DIR = tmp_path
        mod._USERS_FILE = tmp_path / "users.json"
        svc2 = UserService()
        found = svc2.get_user(u.user_id)
        assert found is not None
        assert found.name == "Eve"
