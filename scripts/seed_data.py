"""
Data Seeding & Sample Generation Script for Multi-Agent Knowledge Base Q&A Platform.
Generates realistic sample files (CSV, Parquet, Excel, Markdown, Text) in data/samples/
and provides automated ingestion for local testing and demonstration.
"""

import csv
import json
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.database.connection import get_db_manager
from src.ingestion.structured import StructuredIngestionEngine
from src.ingestion.unstructured import UnstructuredIngestionEngine

SAMPLES_DIR = BASE_DIR / "data" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_sample_files():
    """Create sample datasets on disk if they don't already exist."""
    # 1. orders.csv (Ensure present)
    orders_path = SAMPLES_DIR / "orders.csv"
    if not orders_path.exists():
        with open(orders_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["order_id", "customer_id", "order_date", "status", "total_amount", "shipping_city", "payment_method", "item_count"])
            sample_orders = [
                [1001, 501, "2024-01-15 10:30:00", "completed", 249.99, "New York", "credit_card", 3],
                [1002, 502, "2024-01-16 14:15:00", "completed", 89.50, "San Francisco", "paypal", 1],
                [1003, 503, "2024-01-17 09:00:00", "shipped", 450.00, "Seattle", "credit_card", 5],
                [1004, 501, "2024-01-18 16:45:00", "completed", 120.00, "New York", "apple_pay", 2],
                [1005, 504, "2024-01-19 11:20:00", "cancelled", 35.00, "Austin", "credit_card", 1],
                [1006, 505, "2024-01-20 13:10:00", "completed", 980.25, "Chicago", "bank_transfer", 8],
                [1007, 506, "2024-01-21 15:30:00", "shipped", 75.00, "Boston", "credit_card", 1],
                [1008, 502, "2024-01-22 17:00:00", "completed", 310.40, "San Francisco", "apple_pay", 4],
                [1009, 507, "2024-01-23 08:45:00", "pending", 150.00, "Denver", "paypal", 2],
                [1010, 508, "2024-01-24 12:00:00", "completed", 520.00, "Miami", "credit_card", 6],
            ]
            writer.writerows(sample_orders)
        print(f"Generated {orders_path}")

    # 2. customers (try parquet via pyarrow/pandas, fallback to csv/json)
    customers_data = [
        {"customer_id": 501, "first_name": "Alice", "last_name": "Smith", "email": "alice@example.com", "country": "USA", "signup_date": "2023-01-10", "total_spent": 684.99, "is_active": True},
        {"customer_id": 502, "first_name": "Bob", "last_name": "Jones", "email": "bob@example.com", "country": "USA", "signup_date": "2023-02-14", "total_spent": 399.90, "is_active": True},
        {"customer_id": 503, "first_name": "Charlie", "last_name": "Brown", "email": "charlie@example.com", "country": "Canada", "signup_date": "2023-03-22", "total_spent": 1340.00, "is_active": True},
        {"customer_id": 504, "first_name": "Diana", "last_name": "Prince", "email": "diana@example.com", "country": "UK", "signup_date": "2023-04-05", "total_spent": 180.80, "is_active": False},
        {"customer_id": 505, "first_name": "Evan", "last_name": "Wright", "email": "evan@example.com", "country": "Germany", "signup_date": "2023-05-18", "total_spent": 1022.75, "is_active": True},
        {"customer_id": 506, "first_name": "Fiona", "last_name": "Gallagher", "email": "fiona@example.com", "country": "USA", "signup_date": "2023-06-30", "total_spent": 130.00, "is_active": True},
        {"customer_id": 507, "first_name": "George", "last_name": "Clark", "email": "george@example.com", "country": "Australia", "signup_date": "2023-07-12", "total_spent": 890.50, "is_active": True},
        {"customer_id": 508, "first_name": "Hannah", "last_name": "Abbott", "email": "hannah@example.com", "country": "USA", "signup_date": "2023-08-01", "total_spent": 520.00, "is_active": True},
    ]

    parquet_path = SAMPLES_DIR / "customers.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(customers_data)
        pq.write_table(table, str(parquet_path))
        print(f"Generated {parquet_path} via pyarrow")
    except ImportError:
        try:
            import pandas as pd
            df = pd.DataFrame(customers_data)
            df.to_parquet(parquet_path)
            print(f"Generated {parquet_path} via pandas")
        except Exception:
            # Write CSV representation if parquet library not installed
            customers_csv = SAMPLES_DIR / "customers.csv"
            with open(customers_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(customers_data[0].keys()))
                writer.writeheader()
                writer.writerows(customers_data)
            print(f"Generated {customers_csv} (fallback)")

    # 3. sales_q3.xlsx (try openpyxl / pandas, fallback to csv)
    sales_data = [
        {"transaction_id": "TX101", "customer_id": 501, "product_name": "Wireless Noise-Canceling Headphones", "category": "Electronics", "quantity": 2, "unit_price": 124.99, "discount": 0.0, "sales_amount": 249.98, "region": "North America", "sales_rep": "Sarah Jenkins"},
        {"transaction_id": "TX102", "customer_id": 502, "product_name": "Ergonomic Mechanical Keyboard", "category": "Accessories", "quantity": 1, "unit_price": 89.50, "discount": 0.0, "sales_amount": 89.50, "region": "North America", "sales_rep": "Michael Chang"},
        {"transaction_id": "TX103", "customer_id": 503, "product_name": "Ultra-Wide 34-Inch Curved Monitor", "category": "Electronics", "quantity": 1, "unit_price": 450.00, "discount": 0.0, "sales_amount": 450.00, "region": "Europe", "sales_rep": "Emma Watson"},
        {"transaction_id": "TX104", "customer_id": 505, "product_name": "Standing Desk Converter Pro", "category": "Furniture", "quantity": 2, "unit_price": 490.12, "discount": 0.1, "sales_amount": 882.22, "region": "Europe", "sales_rep": "Emma Watson"},
        {"transaction_id": "TX105", "customer_id": 507, "product_name": "USB-C Dual 4K Display Dock", "category": "Accessories", "quantity": 3, "unit_price": 246.83, "discount": 0.05, "sales_amount": 703.47, "region": "Asia-Pacific", "sales_rep": "Kenji Sato"},
    ]

    excel_path = SAMPLES_DIR / "sales_q3.xlsx"
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Q3_Sales"
        headers = list(sales_data[0].keys())
        ws.append(headers)
        for r in sales_data:
            ws.append([r[k] for k in headers])
        wb.save(excel_path)
        print(f"Generated {excel_path} via openpyxl")
    except ImportError:
        try:
            import pandas as pd
            df = pd.DataFrame(sales_data)
            df.to_excel(excel_path, index=False)
            print(f"Generated {excel_path} via pandas")
        except Exception:
            sales_csv = SAMPLES_DIR / "sales_q3.csv"
            with open(sales_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(sales_data[0].keys()))
                writer.writeheader()
                writer.writerows(sales_data)
            print(f"Generated {sales_csv} (fallback)")


def seed_all_datasets():
    """Ingest sample datasets into storage and database catalog."""
    generate_sample_files()
    db_mgr = get_db_manager()
    structured_engine = StructuredIngestionEngine(db_manager=db_mgr)
    unstructured_engine = UnstructuredIngestionEngine(db_manager=db_mgr)

    print("\nIngesting structured sample datasets...")
    for filename in ["orders.csv", "customers.parquet", "customers.csv", "sales_q3.xlsx", "sales_q3.csv"]:
        file_path = SAMPLES_DIR / filename
        if file_path.exists():
            try:
                ds = structured_engine.ingest_file(file_path, filename=filename)
                print(f"✓ Ingested structured dataset: {ds.name} (ID: {ds.id}, Rows: {ds.row_count})")
            except Exception as e:
                print(f"⚠ Skipping {filename}: {e}")

    print("\nIngesting unstructured sample documents...")
    for filename in ["company_handbook.md", "policy_document.txt"]:
        file_path = SAMPLES_DIR / filename
        if file_path.exists():
            try:
                ds = unstructured_engine.ingest_file(file_path, filename=filename)
                print(f"✓ Ingested unstructured dataset: {ds.name} (ID: {ds.id}, Chunks: {ds.row_count})")
            except Exception as e:
                print(f"⚠ Skipping {filename}: {e}")

    print("\n✓ Seeding complete!")


if __name__ == "__main__":
    seed_all_datasets()
