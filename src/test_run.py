from src.descriptor_engine import compute_properties_table

sample_smiles = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "invalid_smiles",
]

df = compute_properties_table(sample_smiles)
print(df.to_string(index=False))