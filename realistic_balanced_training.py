import pandas as pd
import numpy as np

known_tribes = pd.read_csv('tribal_names.csv')['tribe'].astype(str).str.lower().str.strip().tolist()
rows = []
np.random.seed(888)
N_high = 200_000  # Extreme/high risk (fraud=1)
N_medium = 200_000  # Medium risk
N_low = 100_000  # Low risk

# حالات High
for _ in range(N_high):
    amount = np.round(np.random.uniform(10000, 20_000_000), 2)
    hour = np.random.choice([0,1,2,3,4,21,22,23])
    device = 'new'
    tribe = 'notknown' if np.random.rand() < 0.9 else np.random.choice(known_tribes)
    fraud = 1
    rows.append([amount, hour, device, tribe, fraud])

# حالات Medium
for _ in range(N_medium):
    amount = np.round(np.random.uniform(4000, 25000), 2)
    hour = np.random.choice([8,9,10,11,12,13,14,15,16,17,18,19,20])
    device = np.random.choice(['new', 'known'], p=[0.6, 0.4])
    tribe = 'notknown' if np.random.rand() < 0.5 else np.random.choice(known_tribes)
    # fraud=1 في 40% من الحالات المتوسطة فقط
    fraud = 1 if np.random.rand() < 0.4 else 0
    rows.append([amount, hour, device, tribe, fraud])

# حالات Low (سليمة جدًا)
for _ in range(N_low):
    amount = np.round(np.random.uniform(100, 5000), 2)
    hour = np.random.choice([10,12,15,16,13,11])
    device = 'known'
    tribe = np.random.choice(known_tribes)
    fraud = 0
    rows.append([amount, hour, device, tribe, fraud])

df = pd.DataFrame(rows, columns=['amount', 'hour', 'device', 'tribe', 'fraud'])
df.to_csv('realistic_balanced_training.csv', index=False)
print("تم إنشاء الملف realistic_balanced_training.csv بعدد صفوف:", len(df))
