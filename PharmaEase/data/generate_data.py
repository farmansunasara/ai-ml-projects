"""
PharmaEase — Phase 1: Dataset Generator
Generates 5 CSV files with ~5000 records each for ML model training.
Run this script once to create all datasets inside the /data folder.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("PharmaEase — Generating datasets (5000 records each)...")
print("=" * 52)

# ─────────────────────────────────────────────
# MASTER REFERENCE DATA
# ─────────────────────────────────────────────

medicine_names = [
    'Amoxicillin 500mg', 'Azithromycin 250mg', 'Ciprofloxacin 500mg',
    'Paracetamol 500mg', 'Ibuprofen 400mg', 'Diclofenac 50mg',
    'Omeprazole 20mg', 'Pantoprazole 40mg', 'Ranitidine 150mg',
    'Vitamin C 500mg', 'Vitamin D3 1000IU', 'Vitamin B12 500mcg',
    'Metformin 500mg', 'Glimepiride 2mg', 'Insulin Glargine 100IU',
    'Amlodipine 5mg', 'Atenolol 50mg', 'Losartan 50mg',
    'Cetirizine 10mg', 'Loratadine 10mg', 'Montelukast 10mg',
    'Fluconazole 150mg', 'Clotrimazole Cream', 'Terbinafine 250mg',
    'Dextromethorphan Syrup', 'Ambroxol 30mg', 'Salbutamol Inhaler',
    'Betamethasone Cream', 'Calamine Lotion', 'Hydrocortisone 1%',
    'Aspirin 75mg', 'Clopidogrel 75mg', 'Atorvastatin 10mg',
    'Levothyroxine 50mcg', 'Prednisolone 5mg', 'Dexamethasone 4mg',
    'Ondansetron 4mg', 'Domperidone 10mg', 'Metoclopramide 10mg',
    'Albendazole 400mg', 'Ivermectin 12mg', 'Mebendazole 100mg',
    'Chlorpheniramine 4mg', 'Folic Acid 5mg', 'Iron Sucrose 100mg',
    'Calcium Carbonate 500mg', 'Magnesium Oxide 250mg', 'Zinc Sulphate 20mg',
    'Clonazepam 0.5mg', 'Alprazolam 0.25mg'
]

categories = [
    'Antibiotic', 'Antibiotic', 'Antibiotic',
    'Painkiller', 'Painkiller', 'Painkiller',
    'Antacid', 'Antacid', 'Antacid',
    'Vitamin', 'Vitamin', 'Vitamin',
    'Antidiabetic', 'Antidiabetic', 'Antidiabetic',
    'Antihypertensive', 'Antihypertensive', 'Antihypertensive',
    'Antihistamine', 'Antihistamine', 'Antihistamine',
    'Antifungal', 'Antifungal', 'Antifungal',
    'Cough & Cold', 'Cough & Cold', 'Cough & Cold',
    'Skin Care', 'Skin Care', 'Skin Care',
    'Cardiac', 'Cardiac', 'Cardiac',
    'Thyroid', 'Steroid', 'Steroid',
    'Antiemetic', 'Antiemetic', 'Antiemetic',
    'Antiparasitic', 'Antiparasitic', 'Antiparasitic',
    'Antihistamine', 'Supplement', 'Supplement',
    'Supplement', 'Supplement', 'Supplement',
    'Neurological', 'Neurological'
]

suppliers = [
    'MedLine Pharma', 'SunPharma Dist.', 'Cipla Supplies',
    'Zydus Wholesale', 'Alkem Distributors', 'Abbott India Ltd'
]

first_names = [
    'Aarav', 'Aditi', 'Amit', 'Ananya', 'Arjun', 'Bhavna', 'Chirag',
    'Deepa', 'Dhruv', 'Divya', 'Farhan', 'Geeta', 'Hardik', 'Isha',
    'Jay', 'Kavya', 'Kiran', 'Lakshmi', 'Manish', 'Meera', 'Mihir',
    'Nisha', 'Om', 'Pooja', 'Pratik', 'Priya', 'Rahul', 'Riya',
    'Rohit', 'Sachin', 'Sanjay', 'Shreya', 'Smita', 'Suresh', 'Tanvi',
    'Usha', 'Varun', 'Vidya', 'Vijay', 'Yogesh', 'Zara', 'Neel',
    'Hetal', 'Jinal', 'Krunal', 'Mital', 'Payal', 'Ravi', 'Seema', 'Tushar'
]

last_names = [
    'Shah', 'Patel', 'Mehta', 'Joshi', 'Desai', 'Modi', 'Trivedi',
    'Chauhan', 'Agrawal', 'Gupta', 'Sharma', 'Verma', 'Singh', 'Kumar',
    'Pandya', 'Bhatt', 'Dave', 'Parikh', 'Thakkar', 'Raval', 'Soni',
    'Kapoor', 'Malhotra', 'Iyer', 'Nair', 'Reddy', 'Rao', 'Pillai',
    'Naik', 'Patil'
]

doctors = [
    'Dr. R. Mehta', 'Dr. P. Shah', 'Dr. A. Patel', 'Dr. K. Joshi',
    'Dr. S. Desai', 'Dr. N. Modi', 'Dr. V. Trivedi', 'Dr. H. Chauhan',
    'Dr. M. Agrawal', 'Dr. D. Gupta', 'Dr. T. Sharma', 'Dr. B. Verma'
]

cities    = ['Surat', 'Ahmedabad', 'Vadodara', 'Rajkot', 'Gandhinagar', 'Anand', 'Bharuch', 'Navsari']
dosages   = ['Once daily', 'Twice daily', 'Thrice daily', 'As needed', 'Every 8 hours', 'Every 12 hours']
durations = ['3 days', '5 days', '7 days', '10 days', '14 days', '30 days', '60 days', '90 days']
roles     = ['Pharmacist', 'Pharmacist', 'Cashier', 'Cashier', 'Store Manager', 'Data Entry', 'Delivery Staff']
shifts    = ['Morning (8AM-2PM)', 'Afternoon (2PM-8PM)', 'Night (8PM-2AM)']

def random_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def random_phone():
    return f"+91 {random.randint(6,9)}{random.randint(100000000, 999999999)}"

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

# ─────────────────────────────────────────────
# 1. MEDICINES.CSV — 50 rows (master reference)
# ─────────────────────────────────────────────

medicines = []
for i, name in enumerate(medicine_names):
    price   = round(random.uniform(15, 850), 2)
    stock   = random.randint(20, 500)
    reorder = random.randint(10, 50)
    expiry  = random_date(datetime(2025, 3, 1), datetime(2027, 3, 1))
    medicines.append({
        'medicine_id':   f'MED{str(i+1).zfill(3)}',
        'name':          name,
        'category':      categories[i],
        'price':         price,
        'stock_qty':     stock,
        'expiry_date':   expiry.strftime('%Y-%m-%d'),
        'supplier':      random.choice(suppliers),
        'reorder_level': reorder
    })

df_medicines = pd.DataFrame(medicines)
df_medicines.to_csv(os.path.join(OUTPUT_DIR, 'medicines.csv'), index=False)
print(f"  medicines.csv        → {len(df_medicines):>5} rows  (master reference — all medicines)")

# ─────────────────────────────────────────────
# 2. EMPLOYEES.CSV — 5000 records
# ─────────────────────────────────────────────

salary_map = {
    'Pharmacist':    (28000, 45000),
    'Cashier':       (15000, 22000),
    'Store Manager': (50000, 70000),
    'Data Entry':    (12000, 18000),
    'Delivery Staff':(10000, 15000)
}

employees = []
for i in range(5000):
    role    = random.choice(roles)
    lo, hi  = salary_map[role]
    join    = random_date(datetime(2018, 1, 1), datetime(2023, 12, 31))
    employees.append({
        'employee_id':       f'EMP{str(i+1).zfill(4)}',
        'name':              random_name(),
        'role':              role,
        'shift':             random.choice(shifts),
        'salary':            random.randint(lo, hi),
        'join_date':         join.strftime('%Y-%m-%d'),
        'performance_score': round(random.uniform(55, 99), 1),
        'city':              random.choice(cities)
    })

df_employees = pd.DataFrame(employees)
df_employees.to_csv(os.path.join(OUTPUT_DIR, 'employees.csv'), index=False)
print(f"  employees.csv        → {len(df_employees):>5} rows")

# ─────────────────────────────────────────────
# 3. CUSTOMERS.CSV — 5000 records
# ─────────────────────────────────────────────

customers = []
for i in range(5000):
    customers.append({
        'customer_id':    f'CUST{str(i+1).zfill(4)}',
        'name':           random_name(),
        'age':            random.randint(18, 80),
        'gender':         random.choice(['Male', 'Female']),
        'contact':        random_phone(),
        'city':           random.choice(cities),
        'total_purchases':random.randint(1, 60)
    })

df_customers = pd.DataFrame(customers)
df_customers.to_csv(os.path.join(OUTPUT_DIR, 'customers.csv'), index=False)
print(f"  customers.csv        → {len(df_customers):>5} rows")

# ─────────────────────────────────────────────
# 4. SALES.CSV — 5000 records with seasonal patterns
# ─────────────────────────────────────────────

med_ids    = df_medicines['medicine_id'].tolist()
med_prices = dict(zip(df_medicines['medicine_id'], df_medicines['price']))
emp_ids    = [f'EMP{str(i+1).zfill(4)}' for i in range(50)]
cust_ids   = [f'CUST{str(i+1).zfill(4)}' for i in range(5000)]

# Monthly demand multipliers per category (index 0 = Jan, 11 = Dec)
cat_season = {
    'Cough & Cold':    [1.4,1.3,1.0,0.7,0.6,0.7,0.9,0.9,0.7,0.8,1.2,1.4],
    'Antihistamine':   [0.7,0.7,1.2,1.3,1.2,0.8,0.7,0.7,1.1,1.2,0.8,0.7],
    'Antidiabetic':    [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
    'Vitamin':         [1.2,1.1,0.9,0.8,0.8,0.8,0.8,0.8,0.9,1.0,1.1,1.2],
    'Antacid':         [0.8,0.8,0.9,1.0,1.1,1.3,1.4,1.3,1.1,0.9,0.8,0.8],
    'Antibiotic':      [1.2,1.1,1.0,0.9,0.9,1.0,1.1,1.1,1.0,0.9,1.0,1.2],
    'Skin Care':       [0.8,0.8,1.0,1.2,1.3,1.2,1.0,1.0,1.1,1.1,0.9,0.8],
}

def season_weight(category, month):
    weights = cat_season.get(category, [1.0]*12)
    return weights[month - 1]

sales = []
start = datetime(2023, 1, 1)
end   = datetime(2024, 12, 31)

for i in range(5000):
    sale_date = random_date(start, end)
    med_id    = random.choice(med_ids)
    med_row   = df_medicines[df_medicines['medicine_id'] == med_id].iloc[0]
    category  = med_row['category']
    weight    = season_weight(category, sale_date.month)

    base_qty  = random.randint(1, 5)
    qty       = max(1, int(base_qty * weight))
    unit_price= med_prices[med_id]
    discount  = random.choice([0, 0, 0, 0, 5, 10, 15])
    total     = round(unit_price * qty * (1 - discount / 100), 2)

    sales.append({
        'sale_id':       f'SALE{str(i+1).zfill(5)}',
        'date':          sale_date.strftime('%Y-%m-%d'),
        'month':         sale_date.month,
        'quarter':       (sale_date.month - 1) // 3 + 1,
        'medicine_id':   med_id,
        'medicine_name': med_row['name'],
        'category':      category,
        'quantity':      qty,
        'unit_price':    unit_price,
        'discount_pct':  discount,
        'total_price':   total,
        'customer_id':   random.choice(cust_ids),
        'employee_id':   random.choice(emp_ids)
    })

df_sales = pd.DataFrame(sales)
df_sales = df_sales.sort_values('date').reset_index(drop=True)
df_sales.to_csv(os.path.join(OUTPUT_DIR, 'sales.csv'), index=False)
print(f"  sales.csv            → {len(df_sales):>5} rows  (2 years · seasonal patterns · discounts)")

# ─────────────────────────────────────────────
# 5. PRESCRIPTIONS.CSV — 5000 records
# ─────────────────────────────────────────────

prescriptions = []
for i in range(5000):
    rx_date  = random_date(datetime(2023, 1, 1), datetime(2024, 12, 31))
    med_id   = random.choice(med_ids)
    med_name = df_medicines[df_medicines['medicine_id'] == med_id]['name'].values[0]
    prescriptions.append({
        'prescription_id': f'RX{str(i+1).zfill(5)}',
        'patient_name':    random_name(),
        'age':             random.randint(5, 85),
        'gender':          random.choice(['Male', 'Female']),
        'doctor':          random.choice(doctors),
        'date':            rx_date.strftime('%Y-%m-%d'),
        'medicine_id':     med_id,
        'medicine_name':   med_name,
        'dosage':          random.choice(dosages),
        'duration':        random.choice(durations),
        'city':            random.choice(cities)
    })

df_prescriptions = pd.DataFrame(prescriptions)
df_prescriptions.to_csv(os.path.join(OUTPUT_DIR, 'prescriptions.csv'), index=False)
print(f"  prescriptions.csv    → {len(df_prescriptions):>5} rows")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

total = (len(df_medicines) + len(df_employees) + len(df_customers)
         + len(df_sales) + len(df_prescriptions))
print("=" * 52)
print(f"  Total records generated : {total:,}")
print(f"  Output folder           : {OUTPUT_DIR}")
print("\n  Phase 1 complete. Open Jupyter to start Phase 2.")