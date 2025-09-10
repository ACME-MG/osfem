import sys; sys.path += [".."]
from osfem.data import get_creep, split_data_list
from osfem.modeller import Model

# Define model information
# NAME, INIT, LIMITS, EXP = "stf_tpm", None, (0, 0.6),  None
# NAME, INIT, LIMITS, EXP = "ttf_mg", [3,0.5], (0,10000), 3
# NAME, INIT, LIMITS, EXP = "ttf_mc", None, (0,10000), 3
NAME, INIT, LIMITS, EXP = "ttf_llm", None, (0,10000), 3
# NAME, INIT, LIMITS, EXP = "mcr_arr", [0.067836, 5.1867, 245100.0], None, None
# NAME, INIT, LIMITS, EXP = "mcr_bar", [0.067836, 5.1867, 245100.0], None, None

# Define model
model = Model(NAME)

# Read data
data_list = get_creep("data")
data_list = [data.update({"stress": data["stress"]/80}) or data for data in data_list]
data_list = [data.update({"temperature": data["temperature"]/1000}) or data for data in data_list]
cal_data_list, val_data_list = split_data_list(data_list)

# Optimise
opt_params = model.optimise(cal_data_list, INIT)

# Display results
cal_are = model.get_are(cal_data_list)
val_are = model.get_are(val_data_list)
print(f"Params:   {opt_params}")
print(f"Cal. ARE: {cal_are}")
print(f"Val. ARE: {val_are}")

# Plot results
model.plot_1to1(cal_data_list, val_data_list, limits=LIMITS, exp=EXP)
