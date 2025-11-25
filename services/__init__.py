"""Service package.

Imports are intentionally deferred to avoid circular dependencies. Import
components directly from their modules, e.g.:

```
from services.background import BackgroundIndexer
from services.rerank import LinearReranker
```
"""

__all__: list[str] = []
