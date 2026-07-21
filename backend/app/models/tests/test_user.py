from app.models.user import User, UserCreateRequest, UserResponse, UserDeleteResponse


class TestUser:
    def test_construction(self) -> None:
        u = User(user_id="u1", name="Alice", api_key="sk-xxx", role="user", created_at="2024-01-01")
        assert u.user_id == "u1"
        assert u.name == "Alice"
        assert u.role == "user"

    def test_default_role(self) -> None:
        u = User(user_id="u2", name="Bob", api_key="k")
        assert u.role == "user"

    def test_admin_role(self) -> None:
        u = User(user_id="u3", name="Admin", api_key="k", role="admin")
        assert u.role == "admin"

    def test_api_key_excluded_from_repr(self) -> None:
        u = User(user_id="u1", name="A", api_key="secret")
        r = repr(u)
        assert "secret" not in r


class TestUserCreateRequest:
    def test_default_role(self) -> None:
        r = UserCreateRequest(name="Alice")
        assert r.name == "Alice"
        assert r.role == "user"

    def test_admin(self) -> None:
        r = UserCreateRequest(name="Root", role="admin")
        assert r.role == "admin"


class TestUserResponse:
    def test_fields(self) -> None:
        u = User(user_id="u1", name="A", api_key="k")
        r = UserResponse(trace_id="t1", user=u)
        assert r.trace_id == "t1"
        assert r.user.user_id == "u1"


class TestUserDeleteResponse:
    def test_deleted(self) -> None:
        r = UserDeleteResponse(trace_id="t1", user_id="u1", deleted=True)
        assert r.deleted is True
