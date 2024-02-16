# Inspired by: https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
from __future__ import annotations

pytest_plugins = [
    "pachyderm.test_fixtures.logging",
    "pachyderm.test_fixtures.objects",
]
