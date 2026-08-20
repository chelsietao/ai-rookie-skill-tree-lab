"""Compile one schema for three RDBMS dialects without requiring remote servers."""

from __future__ import annotations

from sqlalchemy import Boolean, Column, Integer, MetaData, Numeric, String, Table, create_mock_engine
from sqlalchemy.schema import CreateTable


def customer_table() -> Table:
    metadata = MetaData()
    return Table(
        "customers",
        metadata,
        Column("customer_id", String(20), primary_key=True),
        Column("age", Integer, nullable=False),
        Column("tenure_days", Integer, nullable=False),
        Column("region", String(20), nullable=False),
        Column("annual_spend", Numeric(12, 2), nullable=False),
        Column("churned", Boolean, nullable=False),
    )


def compiled_schemas() -> dict[str, str]:
    table = customer_table()
    schemas: dict[str, str] = {}
    for name, url in {
        "sqlite": "sqlite://",
        "mysql": "mysql+pymysql://",
        "mssql": "mssql+pyodbc://",
    }.items():
        engine = create_mock_engine(url, lambda *_args, **_kwargs: None)
        schemas[name] = str(CreateTable(table).compile(dialect=engine.dialect))
    return schemas
