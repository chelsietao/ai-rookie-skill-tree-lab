CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL,
    tenure_days INTEGER NOT NULL,
    region TEXT NOT NULL,
    annual_spend REAL NOT NULL,
    churned INTEGER NOT NULL CHECK (churned IN (0, 1))
);

CREATE TABLE IF NOT EXISTS product_views (
    view_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product_id TEXT NOT NULL,
    day INTEGER NOT NULL,
    dwell_seconds REAL NOT NULL,
    page_depth INTEGER NOT NULL,
    converted INTEGER NOT NULL CHECK (converted IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_product_views_customer_day
ON product_views(customer_id, day);
