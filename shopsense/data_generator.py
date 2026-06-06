import pandas as pd
import numpy as np
from datetime import timedelta

def generate_customers(n_customers: int = 10000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    
    # IDs
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, n_customers + 1)]
    
    # Signup Dates: Jan 2021 to Dec 2022
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2022-12-31')
    days_range = (end_date - start_date).days
    random_days = rng.integers(0, days_range, n_customers)
    signup_dates = start_date + pd.to_timedelta(random_days, unit='D')
    
    # Age: Normal dist, mean=35, std=10, clipped [18, 70]
    ages = np.clip(rng.normal(loc=35, scale=10, size=n_customers), 18, 70).astype(int)
    
    # Gender
    genders = rng.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04], size=n_customers)
    
    # City (20 cities, weighted)
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 
              'Ahmedabad', 'Jaipur', 'Surat', 'Lucknow', 'Kanpur', 'Nagpur', 'Patna', 
              'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Vadodara', 'Ghaziabad']
    weights = [0.15, 0.15, 0.10] + [0.60 / 17] * 17
    assigned_cities = rng.choice(cities, p=weights, size=n_customers)
    
    # Acquisition
    channels = ['organic', 'paid_search', 'social_media', 'referral', 'email']
    assigned_channels = rng.choice(channels, size=n_customers)
    
    # Premium status
    is_premium = rng.choice([True, False], p=[0.20, 0.80], size=n_customers)
    
    # Churn Label (~25%)
    churn_label = rng.choice([0, 1], p=[0.75, 0.25], size=n_customers)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'signup_date': signup_dates,
        'age': ages,
        'gender': genders,
        'city': assigned_cities,
        'acquisition_channel': assigned_channels,
        'is_premium': is_premium,
        'churn_label': churn_label
    })
    
    return df

def generate_products(random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_products = 200
    
    product_ids = [f"PROD_{str(i).zfill(3)}" for i in range(1, n_products + 1)]
    categories = ['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports', 'Books']
    assigned_categories = rng.choice(categories, size=n_products)
    
    # Category-wise base pricing (log-normal)
    base_prices = []
    for cat in assigned_categories:
        if cat == 'Electronics':
            price = rng.lognormal(mean=8.5, sigma=1.0) # Higher prices
        elif cat == 'Fashion':
            price = rng.lognormal(mean=6.5, sigma=0.8)
        else:
            price = rng.lognormal(mean=6.0, sigma=0.8)
        base_prices.append(np.clip(price, 50, 50000))
        
    brands = rng.choice(['budget', 'mid', 'premium'], p=[0.4, 0.4, 0.2], size=n_products)
    
    # Beta distribution for ratings (skewed towards 4-5)
    ratings = rng.beta(a=8, b=2, size=n_products) * 5
    ratings = np.clip(ratings, 1.0, 5.0).round(1)
    
    df = pd.DataFrame({
        'product_id': product_ids,
        'product_name': [f"Product_{i}" for i in product_ids],
        'category': assigned_categories,
        'base_price': np.round(base_prices, 2),
        'brand_tier': brands,
        'avg_rating': ratings
    })
    
    return df

def generate_transactions(customers_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    
    # Product distribution setup
    categories = ['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports', 'Books']
    product_ids = [f"PROD_{str(i).zfill(3)}" for i in range(1, 201)]
    prod_category_map = {pid: rng.choice(categories) for pid in product_ids}
    
    records = []
    txn_counter = 1
    
    # Transaction dates: Jan 2021 to Dec 2023
    global_end = pd.to_datetime('2023-12-31')
    
    for _, cust in customers_df.iterrows():
        # Churned customers have fewer transactions and stop buying earlier
        if cust['churn_label'] == 1:
            n_txns = rng.poisson(lam=3)
            # Stop buying 3 to 12 months before global end
            cust_end = global_end - pd.Timedelta(days=rng.integers(90, 365))
        else:
            n_txns = rng.poisson(lam=15)
            cust_end = global_end
            
        n_txns = max(1, n_txns) # Ensure at least 1 for simplicity (or allow 0 by skipping)
        if n_txns == 0:
            continue
            
        # Generate random dates between signup and cust_end
        days_active = (cust_end - cust['signup_date']).days
        if days_active < 1:
            days_active = 1
            
        random_days = rng.integers(0, days_active, n_txns)
        txn_dates = cust['signup_date'] + pd.to_timedelta(random_days, unit='D')
        
        # Apply Seasonality (Shift dates towards Q4 probabilistically)
        for i in range(len(txn_dates)):
            if rng.random() < 0.2: # 20% chance to force into Oct, Nov, Dec
                month = rng.choice([10, 11, 12])
                try:
                    txn_dates.values[i] = txn_dates[i].replace(month=month)
                except ValueError:
                    pass # Handle edge cases like Nov 31st simply by skipping
                    
        txn_dates = sorted(txn_dates)
        
        for t_date in txn_dates:
            pid = rng.choice(product_ids)
            cat = prod_category_map[pid]
            
            # Log-normal prices based on category
            if cat == 'Electronics':
                u_price = rng.lognormal(mean=8.5, sigma=1.0)
                ret_prob = 0.12
                disc_prob = 0.1
            elif cat in ['Fashion', 'Beauty']:
                u_price = rng.lognormal(mean=6.5, sigma=0.8)
                ret_prob = 0.10
                disc_prob = rng.uniform(0.1, 0.5)
            else:
                u_price = rng.lognormal(mean=6.0, sigma=0.8)
                ret_prob = 0.05
                disc_prob = rng.uniform(0.0, 0.3)
                
            u_price = np.clip(u_price, 50, 50000)
            qty = int(np.clip(rng.lognormal(mean=0.0, sigma=0.5), 1, 10))
            
            records.append({
                'transaction_id': f"TXN_{str(txn_counter).zfill(7)}",
                'customer_id': cust['customer_id'],
                'transaction_date': t_date,
                'product_id': pid,
                'category': cat,
                'quantity': qty,
                'unit_price': np.round(u_price, 2),
                'discount_pct': np.round(disc_prob if rng.random() < 0.6 else 0.0, 2),
                'payment_method': rng.choice(['UPI', 'credit_card', 'debit_card', 'COD', 'wallet']),
                'return_flag': 1 if rng.random() < ret_prob else 0
            })
            txn_counter += 1

    return pd.DataFrame(records)

def generate_events(customers_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    records = []
    evt_counter = 1
    global_end = pd.to_datetime('2023-12-31')
    categories = ['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports', 'Books']
    event_types = ['page_view', 'add_to_cart', 'wishlist_add', 'checkout_start', 'purchase', 'search']
    
    for _, cust in customers_df.iterrows():
        # Churned customers have fewer events
        n_events = rng.poisson(lam=10) if cust['churn_label'] == 1 else rng.poisson(lam=40)
        n_events = max(1, n_events)
        
        cust_end = global_end - pd.Timedelta(days=rng.integers(90, 365)) if cust['churn_label'] == 1 else global_end
        days_active = (cust_end - cust['signup_date']).days
        if days_active < 1: days_active = 1
        
        random_days = rng.integers(0, days_active, n_events)
        evt_dates = cust['signup_date'] + pd.to_timedelta(random_days, unit='D')
        
        # Simulating declining trend for churned customers: push dates towards earlier months
        if cust['churn_label'] == 1:
            evt_dates = evt_dates - pd.to_timedelta(rng.integers(0, 60, size=n_events), unit='D')
            
        for e_date in evt_dates:
            records.append({
                'event_id': f"EVT_{str(evt_counter).zfill(7)}",
                'customer_id': cust['customer_id'],
                'event_date': e_date,
                'event_type': rng.choice(event_types, p=[0.5, 0.15, 0.1, 0.1, 0.05, 0.1]),
                'session_duration_sec': max(5, int(rng.normal(loc=240, scale=120))),
                'device_type': rng.choice(['mobile', 'desktop', 'tablet'], p=[0.65, 0.30, 0.05]),
                'page_category': rng.choice(categories)
            })
            evt_counter += 1
            
    return pd.DataFrame(records)