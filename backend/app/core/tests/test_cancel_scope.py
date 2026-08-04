from app.core.cancel_scope import CancelScope


class TestCancelScope:
    def test_initial_generation(self) -> None:
        scope = CancelScope()
        assert scope.generation == 0
        assert scope.is_stale(0) is False

    def test_cancel_invalidates_old_generations(self) -> None:
        scope = CancelScope()
        old = scope.generation
        new = scope.cancel()
        assert new == old + 1
        assert scope.is_stale(old) is True
        assert scope.is_stale(new) is False

    def test_multiple_cancels(self) -> None:
        scope = CancelScope(initial=5)
        gen = scope.cancel()
        assert gen == 6
        gen = scope.cancel()
        assert gen == 7
        assert scope.is_stale(6) is True
        assert scope.is_stale(7) is False
