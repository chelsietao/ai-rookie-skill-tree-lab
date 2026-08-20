CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL,
    tenure_days INTEGER NOT NULL,
    region TEXT NOT NULL,
    annual_spend REAL NOT NULL,
    churned INTEGER NOT NULL CHECK (churned IN (0, 1))
);

