import pandas as pd

# حمّل الملفين
df1 = pd.read_csv('realistic_balanced_training.csv')
df2 = pd.read_csv('borderline_cases.csv')

# دمج الملفين (بدون إعادة فهرسة الأعمدة)
df_merged = pd.concat([df1, df2], ignore_index=True)

# حفظ الملف النهائي
df_merged.to_csv('fraud_merged_data2.csv', index=False)
print("تم الدمج! عدد الصفوف:", len(df_merged))
