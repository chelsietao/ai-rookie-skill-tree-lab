from __future__ import annotations

import os
from pathlib import Path

import pymysql


def main() -> None:
    connection = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        user="root",
        password="",
        database="rookie_lab",
    )
    with connection, connection.cursor() as cursor:
        for statement in Path("sql/mysql/schema.sql").read_text().split(";"):
            if statement.strip():
                cursor.execute(statement)
        cursor.execute("SHOW TABLES LIKE 'customers'")
        assert cursor.fetchone() is not None


if __name__ == "__main__":
    main()
