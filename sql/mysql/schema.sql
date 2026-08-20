CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    age INT NOT NULL,
    tenure_days INT NOT NULL,
    region VARCHAR(20) NOT NULL,
    annual_spend DECIMAL(12, 2) NOT NULL,
    churned BOOLEAN NOT NULL,
    INDEX idx_customers_region (region)
) ENGINE=InnoDB;

