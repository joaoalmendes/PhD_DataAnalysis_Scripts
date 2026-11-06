from scipy import stats

# Insert your values here
f_stat = 22.50
df1 = 1
df2 = 52

# Calculate p-value
p_value = stats.f.sf(f_stat, df1, df2)
print(f"P-value: {p_value}")
