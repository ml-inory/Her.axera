import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.models.user import User

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_USERS_FILE = _DATA_DIR / "users.json"


class UserService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._users: dict[str, User] = {}
        self._key_index: dict[str, str] = {}  # api_key -> user_id
        self._load()

    def _load(self) -> None:
        if not _USERS_FILE.exists():
            return
        try:
            data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
            for uid, udata in data.items():
                user = User(**udata)
                self._users[uid] = user
                self._key_index[user.api_key] = uid
            logger.info("Loaded %d users", len(self._users))
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load users", exc_info=True)

    def _save(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {uid: u.model_dump() for uid, u in self._users.items()}
        _USERS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def validate_api_key(self, api_key: str) -> User | None:
        uid = self._key_index.get(api_key)
        if uid is None:
            return None
        return self._users.get(uid)

    def create_user(self, name: str, role: str = "user") -> User:
        user = User(
            user_id=f"usr_{uuid4().hex[:12]}",
            name=name,
            api_key=f"her_{uuid4().hex}",
            role=role,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._users[user.user_id] = user
        self._key_index[user.api_key] = user.user_id
        self._save()
        return user

    def list_users(self) -> list[User]:
        return list(self._users.values())

    def delete_user(self, user_id: str) -> bool:
        user = self._users.pop(user_id, None)
        if user is None:
            return False
        self._key_index.pop(user.api_key, None)
        self._save()
        return True

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)


user_service = UserService()
