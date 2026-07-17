"""SQL validation for read-only DuckDB access.

These checks catch bad statements early with clear messages; the
``read_only=True`` connection in db.py is the actual enforcement boundary.
"""

import re


class SQLValidationError(ValueError):
    pass


_DENY = (
    "ATTACH",
    "DETACH",
    "COPY",
    "INSTALL",
    "LOAD",
    "PRAGMA",
    "SET",
    "RESET",
    "EXPORT",
    "IMPORT",
    "CREATE",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CALL",
    "CHECKPOINT",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "VACUUM",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
)

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def validate_sql(sql: str) -> str:
    """Return the cleaned statement or raise SQLValidationError."""
    cleaned = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql)).strip()
    cleaned = cleaned.rstrip(";").strip()
    if not cleaned:
        raise SQLValidationError("Empty SQL statement.")
    if ";" in cleaned:
        raise SQLValidationError("Only a single SQL statement is allowed.")
    first = re.split(r"\s+", cleaned, maxsplit=1)[0].upper()
    if first not in ("SELECT", "WITH"):
        raise SQLValidationError("Only SELECT or WITH...SELECT statements are allowed.")
    for word in _DENY:
        if re.search(rf"\b{word}\b", cleaned, re.IGNORECASE):
            raise SQLValidationError(f"Statement contains blocked keyword: {word}.")
    return cleaned
