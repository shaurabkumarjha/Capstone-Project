--New Database
CREATE DATABASE flipkart_sales;

-- All Tables 
-- CUSTOMERS TABLE
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_unique_id VARCHAR(50),
    customer_zip_code_prefix INTEGER,
    customer_city VARCHAR(100),
    customer_state VARCHAR(10)
);

-- SELLERS TABLE
CREATE TABLE sellers (
    seller_id VARCHAR(50) PRIMARY KEY,
    seller_zip_code_prefix INTEGER,
    seller_city VARCHAR(100),
    seller_state VARCHAR(10)
);

-- PRODUCTS TABLE
CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_category_name VARCHAR(100),
    product_name_lenght NUMERIC,
    product_description_lenght NUMERIC,
    product_photos_qty NUMERIC,
    product_weight_g NUMERIC,
    product_length_cm NUMERIC,
    product_height_cm NUMERIC,
    product_width_cm NUMERIC
);

-- ORDERS TABLE
CREATE TABLE orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    order_status VARCHAR(30),
    order_purchase_timestamp TIMESTAMP,
    order_approved_at VARCHAR(50),
    order_delivered_carrier_date VARCHAR(50),
    order_delivered_customer_date VARCHAR(50),
    order_estimated_delivery_date TIMESTAMP
);

-- ORDER ITEMS TABLE
CREATE TABLE order_items (
    order_id VARCHAR(50),
    order_item_id INTEGER,
    product_id VARCHAR(50),
    seller_id VARCHAR(50),
    shipping_limit_date TIMESTAMP,
    price NUMERIC,
    freight_value NUMERIC
);

-- PAYMENTS TABLE
CREATE TABLE payments (
    order_id VARCHAR(50),
    payment_sequential INTEGER,
    payment_type VARCHAR(30),
    payment_installments INTEGER,
    payment_value NUMERIC
);

-- REVIEWS TABLE
CREATE TABLE reviews (
    review_id VARCHAR(50),
    order_id VARCHAR(50),
    review_score INTEGER,
    review_comment_title VARCHAR(200),
    review_comment_message TEXT,
    review_creation_date TIMESTAMP,
    review_answer_timestamp TIMESTAMP
);

--Data Loading in 
--Load 1 — Customers
COPY customers
FROM 'C:/Users/Shaurab kumar jha/Google Drive/Flipkart_Sales_Project/1_Data/raw/olist_customers_dataset.csv'
DELIMITER ','
CSV HEADER;

--Load 2 — Sellers
COPY sellers
FROM 'C:/Users/Shaurab kumar jha/Google Drive/Flipkart_Sales_Project/1_Data/raw/olist_sellers_dataset.csv'
DELIMITER ','
CSV HEADER;

--Load 3 — Products
COPY products
FROM 'C:/Users/Shaurab kumar jha/Google Drive/Flipkart_Sales_Project/1_Data/raw/olist_products_dataset.csv'
DELIMITER ','
CSV HEADER;

--Load 4 — Orders
COPY orders
FROM 'C:/Users/Shaurab kumar jha/Google Drive/Flipkart_Sales_Project/1_Data/raw/olist_orders_dataset.csv'
DELIMITER ','
CSV HEADER;

--Load 5 — Order Items
COPY order_items
FROM 'C:/Users/Shaurab kumar jha/Google Drive/Flipkart_Sales_Project/1_Data/raw/olist_order_items_dataset.csv'
DELIMITER ','
CSV HEADER;

--Load 6 — Payments
COPY payments
FROM 'C:/Users/Shaurab kumar jha/Google Drive/Flipkart_Sales_Project/1_Data/raw/olist_order_payments_dataset.csv'
DELIMITER ','
CSV HEADER;

--Load 7 — Reviews
COPY reviews
FROM 'C:/Users/Shaurab kumar jha/Google Drive/Flipkart_Sales_Project/1_Data/raw/olist_order_reviews_dataset.csv'
DELIMITER ','
CSV HEADER;

--Verify all data is loaded or not
SELECT 'customers'  AS table_name, COUNT(*) FROM customers  UNION ALL
SELECT 'sellers'    AS table_name, COUNT(*) FROM sellers     UNION ALL
SELECT 'products'   AS table_name, COUNT(*) FROM products    UNION ALL
SELECT 'orders'     AS table_name, COUNT(*) FROM orders      UNION ALL
SELECT 'order_items'AS table_name, COUNT(*) FROM order_items UNION ALL
SELECT 'payments'   AS table_name, COUNT(*) FROM payments    UNION ALL
SELECT 'reviews'    AS table_name, COUNT(*) FROM reviews;

--Query 1 — Total Revenue
SELECT 
    ROUND(SUM(payment_value)::NUMERIC, 2) AS total_revenue,
    COUNT(DISTINCT order_id)              AS total_orders,
    ROUND(AVG(payment_value)::NUMERIC, 2) AS avg_order_value
FROM payments;

--Query 2 — Top 10 Product Categories
SELECT 
    p.product_category_name,
    COUNT(oi.order_id)                    AS total_orders,
    ROUND(SUM(oi.price)::NUMERIC, 2)      AS total_revenue
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.product_category_name
ORDER BY total_revenue DESC
LIMIT 10;

--Query 3 — Region wise Revenue
SELECT 
    c.customer_state                      AS region,
    COUNT(DISTINCT o.order_id)            AS total_orders,
    ROUND(SUM(p.payment_value)::NUMERIC, 2) AS total_revenue
FROM orders o
JOIN customers c   ON o.customer_id  = c.customer_id
JOIN payments p    ON o.order_id     = p.order_id
GROUP BY c.customer_state
ORDER BY total_revenue DESC
LIMIT 10;

--Query 4 — Monthly Sales Trend
SELECT 
    TO_CHAR(o.order_purchase_timestamp, 'YYYY-MM') AS month,
    COUNT(DISTINCT o.order_id)                      AS total_orders,
    ROUND(SUM(p.payment_value)::NUMERIC, 2)         AS monthly_revenue
FROM orders o
JOIN payments p ON o.order_id = p.order_id
GROUP BY month
ORDER BY month;

--Query 5 — Order Status Breakdown
SELECT 
    order_status,
    COUNT(*) AS total_orders,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM orders
GROUP BY order_status
ORDER BY total_orders DESC;

--Query 6 — Average Order Value by Category
SELECT 
    p.product_category_name,
    ROUND(AVG(oi.price)::NUMERIC, 2)  AS avg_price,
    ROUND(MAX(oi.price)::NUMERIC, 2)  AS max_price,
    ROUND(MIN(oi.price)::NUMERIC, 2)  AS min_price
FROM order_items oi
JOIN products p ON oi.product_id = p.product_id
GROUP BY p.product_category_name
ORDER BY avg_price DESC
LIMIT 10;
