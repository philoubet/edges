from bw2data import Database, projects, databases, __version__

# Determine Brightway version
if __version__ < (4, 0, 0):
    is_bw2 = True
else:
    is_bw2 = False
    try:
        projects.migrate_project_25()
    except:
        pass

print(f"Using bw2: {is_bw2}")

# Set project
project = "EdgeLCC-Test" if is_bw2 else "EdgeLCC-Test-bw25"
projects.set_current(project)

# Clean up if exists
if "lcc-test-db" in databases:
    del databases["lcc-test-db"]

# Define process/product names
processes = [
    "Production of electricity",
    "Production of wood",
    "Production of timber",
    "Production of chair",
    "Use of chair",
    "Disposal of broken chair",
]

products = [
    "Electricity",
    "Wood",
    "Timber",
    "Chair",
    "Sitting",
    "Broken chair",
]

# CPC codes and matched descriptions (based on earlier mapping)
cpc_classifications = {
    "Production of electricity": ("311", "Electricity, gas, steam and air conditioning supply"),
    "Production of wood": ("313", "Wood in the rough, treated with paint, stains, creosote or other preservatives"),
    "Production of timber": ("3120", "Logs of non-coniferous wood"),
    "Production of chair": ("31450", "Plywood, veneer panels and similar laminated wood"),
    "Use of chair": None,
    "Disposal of broken chair": ("94339", "Other services related to waste treatment and disposal n.e.c."),
}

# Input-output matrix (rows = products, cols = processes)
data = [
    [1, -0.5, 0, -2, 0, 0],    # Electricity
    [0,  5.0, 0, -5, 0, 0],    # Wood
    [0, -7.0, 1,  0, 0, 0],    # Timber
    [0,  0.0, 0,  5, -5, 0],   # Chair
    [0,  0.0, 0,  0, 10, 0],   # Sitting
    [0,  0.0, 0,  0,  5, -1],  # Broken chair
]

# Build Brightway-compatible dataset
db = Database("lcc-test-db")
db_data = {}

for col, process_name in enumerate(processes):
    exchanges = []

    # The process produces its own reference product
    ref_product = products[col]
    exchanges.append({
        "amount": data[col][col],
        "type": "production",
        "product": ref_product,
        "unit": "unit",
        "name": process_name,
        "input": ("lcc-test-db", process_name),
    })

    for row, product_name in enumerate(products):
        amount = data[row][col]
        if row != col and amount != 0:
            exchanges.append({
                "amount": abs(amount),
                "type": "technosphere",
                "input": ("lcc-test-db", processes[row]),
                "unit": "unit",
            })

    # Apply classification if available
    cpc_entry = cpc_classifications[process_name]
    classification = [("CPC", f"{cpc_entry[0]}: {cpc_entry[1]}")] if cpc_entry else []

    db_data[("lcc-test-db", process_name)] = {
        "name": process_name,
        "reference product": ref_product,
        "unit": "unit",
        "location": "CH",
        "classifications": classification,
        "exchanges": exchanges,
    }

# Write to Brightway
db.write(db_data)
print(f"The following databases are now available: {databases}")
