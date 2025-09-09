import sys; sys.path += [".."]
from osfem.data import get_creep, split_data_list
from osfem.modeller import Model

# Define model information
# MODEL_INFO, LIMITS = "stf_ew", (0, 0.6)
MODEL_INFO, LIMITS = "ttf_mg", (0,10000)

# Define model
model = Model(MODEL_INFO)

# Read data
data_list = get_creep("data")
data_list = [data.update({"stress": data["stress"]/80}) or data for data in data_list]
data_list = [data.update({"temperature": data["temperature"]/1000}) or data for data in data_list]
cal_data_list, val_data_list = split_data_list(data_list)

# Optimise
opt_params = model.optimise(cal_data_list)
model.plot_1to1(cal_data_list, val_data_list, limits=LIMITS)
cal_are = model.get_are(cal_data_list)
val_are = model.get_are(val_data_list)

# Display results
print(f"Params:   {opt_params}")
print(f"Cal. ARE: {cal_are}")
print(f"Val. ARE: {val_are}")
