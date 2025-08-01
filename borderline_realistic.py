import pandas as pd
import numpy as np

known_tribes = pd.read_csv('tribal_names.csv')['tribe'].astype(str).str.lower().str.strip().tolist()
np.random.seed(2025)

rows = []
N = 300_000

for _ in range(N):
    amount = np.round(np.random.uniform(7000, 20000), 2)
    hour = np.random.choice(range(8, 21))
    device = np.random.choice(['new', 'known'], p=[0.6, 0.4])
    tribe = np.random.choice(known_tribes + ['notknown'], p=[0.7/len(known_tribes)]*len(known_tribes) + [0.3])
    
    # Fraud logic: غالبًا طبيعي إلا إذا توفرت شرطين أو أكثر
    if device == 'new' and tribe == 'notknown':
        fraud = 1 if np.random.rand() < 0.7 else 0  # مشبوه غالبًا
    elif amount > 12000 and device == 'new':
        fraud = 1 if np.random.rand() < 0.5 else 0
    elif amount > 15000 and tribe == 'notknown':
        fraud = 1 if np.random.rand() < 0.4 else 0
    else:
        fraud = 1 if np.random.rand() < 0.2 else 0  # أغلبها طبيعي

    rows.append([amount, hour, device, tribe, fraud])

df = pd.DataFrame(rows, columns=['amount', 'hour', 'device', 'tribe', 'fraud'])
df.to_csv('borderline_realistic.csv', index=False)
print("تم إنشاء الملف borderline_realistic.csv بعدد صفوف:", len(df))

