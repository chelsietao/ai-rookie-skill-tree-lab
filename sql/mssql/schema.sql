IF OBJECT_ID('dbo.customers', 'U') IS NULL
CREATE TABLE dbo.customers (
    customer_id NVARCHAR(20) NOT NULL PRIMARY KEY,
    age INT NOT NULL,
    tenure_days INT NOT NULL,
    region NVARCHAR(20) NOT NULL,
    annual_spend DECIMAL(12, 2) NOT NULL,
    churned BIT NOT NULL
);

