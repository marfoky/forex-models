
new_data_set_scaled = sc.transform(backcandles)

# Step 2: Create sequences (like rolling window) of last 30 records
backcandles = 30  # same as in your original code
X_new = []

for j in range(8):  # 8 feature columns as in your original code
    X_new.append([])
    for i in range(backcandles, new_data_set_scaled.shape[0]):
        X_new[j].append(new_data_set_scaled[i-backcandles:i, j])

# Step 3: Reshape data
X_new = np.moveaxis(X_new, [0], [2])