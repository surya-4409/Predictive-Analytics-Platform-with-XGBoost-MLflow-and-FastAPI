import pandas as pd
from sqlalchemy import MetaData, Table, Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Index, text

def create_schema_and_tables(engine) -> None:
    # 1. Create the schema if it doesn't exist
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS shopsense"))
        conn.commit()

    # 2. Define the metadata bound to our schema
    metadata = MetaData(schema="shopsense")

    # 3. Define Tables
    customers = Table('customers', metadata,
        Column('customer_id', String, primary_key=True),
        Column('signup_date', DateTime),
        Column('age', Integer),
        Column('gender', String),
        Column('city', String),
        Column('acquisition_channel', String),
        Column('is_premium', Boolean),
        Column('churn_label', Integer)
    )

    products = Table('products', metadata,
        Column('product_id', String, primary_key=True),
        Column('product_name', String),
        Column('category', String),
        Column('base_price', Float),
        Column('brand_tier', String),
        Column('avg_rating', Float)
    )

    transactions = Table('transactions', metadata,
        Column('transaction_id', String, primary_key=True),
        Column('customer_id', String, ForeignKey('shopsense.customers.customer_id')),
        Column('transaction_date', DateTime),
        Column('product_id', String, ForeignKey('shopsense.products.product_id')),
        Column('category', String),
        Column('quantity', Integer),
        Column('unit_price', Float),
        Column('discount_pct', Float),
        Column('payment_method', String),
        Column('return_flag', Integer)
    )

    events = Table('events', metadata,
        Column('event_id', String, primary_key=True),
        Column('customer_id', String, ForeignKey('shopsense.customers.customer_id')),
        Column('event_date', DateTime),
        Column('event_type', String),
        Column('session_duration_sec', Integer),
        Column('device_type', String),
        Column('page_category', String)
    )

    # 4. Create minimum required indexes
    Index('ix_transactions_customer_id', transactions.c.customer_id)
    Index('ix_transactions_transaction_date', transactions.c.transaction_date)
    Index('ix_events_customer_id', events.c.customer_id)
    Index('ix_events_event_date', events.c.event_date)

    # 5. Execute creation
    metadata.create_all(engine)

def load_dataframe_to_db(df: pd.DataFrame, table_name: str, engine, schema: str = "shopsense", if_exists: str = "replace") -> int:
    """
    Loads a pandas DataFrame into the database. 
    Note: if_exists='replace' in pandas drops the table. If you want to keep the SQLAlchemy
    indexes and foreign keys defined above, pass if_exists='append' when calling this in practice.
    """
    inserted_rows = df.to_sql(name=table_name, con=engine, schema=schema, if_exists=if_exists, index=False)
    # Return count depending on SQL driver behavior, fallback to length of df
    return inserted_rows if inserted_rows is not None else len(df)

def execute_query(query: str, engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), con=conn)