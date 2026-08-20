CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    age INT NOT NULL,
    tenure_days INT NOT NULL,
    region VARCHAR(20) NOT NULL,
    annual_spend DECIMAL(12, 2) NOT NULL,
    churned BOOLEAN NOT NULL,
    INDEX idx_customers_region (region)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS product_views (
    view_id VARCHAR(24) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    product_id VARCHAR(20) NOT NULL,
    day INT NOT NULL,
    dwell_seconds DECIMAL(10, 2) NOT NULL,
    page_depth INT NOT NULL,
    converted BOOLEAN NOT NULL,
    CONSTRAINT fk_views_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    INDEX idx_product_views_customer_day (customer_id, day)
) ENGINE=InnoDB;
