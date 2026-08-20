IF OBJECT_ID('dbo.customers', 'U') IS NULL
CREATE TABLE dbo.customers (
    customer_id NVARCHAR(20) NOT NULL PRIMARY KEY,
    age INT NOT NULL,
    tenure_days INT NOT NULL,
    region NVARCHAR(20) NOT NULL,
    annual_spend DECIMAL(12, 2) NOT NULL,
    churned BIT NOT NULL
);

IF OBJECT_ID('dbo.product_views', 'U') IS NULL
CREATE TABLE dbo.product_views (
    view_id NVARCHAR(24) NOT NULL PRIMARY KEY,
    customer_id NVARCHAR(20) NOT NULL,
    product_id NVARCHAR(20) NOT NULL,
    day INT NOT NULL,
    dwell_seconds DECIMAL(10, 2) NOT NULL,
    page_depth INT NOT NULL,
    converted BIT NOT NULL,
    CONSTRAINT fk_views_customer FOREIGN KEY (customer_id) REFERENCES dbo.customers(customer_id)
);
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'idx_product_views_customer_day')
CREATE INDEX idx_product_views_customer_day ON dbo.product_views(customer_id, day);
