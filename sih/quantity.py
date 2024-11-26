import pandas as pd

# Load the Excel file
file_path = r'C:\\Users\\vamoh\Downloads\\fertilizer123.xlsx'
df = pd.read_excel(file_path)

def calculate_fertilizer_amount(fertilizer_type, nutrient_type, nutrient_kg_per_ha):
    # Find the row corresponding to the given fertilizer type
    fertilizer_row = df[df['Fertilizer'].str.lower() == fertilizer_type.lower()]
    
    if fertilizer_row.empty:
        return f"Fertilizer type '{fertilizer_type}' not found."
    
    # Get the percentage of the specific nutrient in the fertilizer
    nutrient_percentage = fertilizer_row[nutrient_type].values[0]
    
    if nutrient_percentage == 0:
        return f"The fertilizer '{fertilizer_type}' does not contain the nutrient '{nutrient_type}'."
    
    # Calculate the amount of fertilizer needed
    fertilizer_amount = (nutrient_kg_per_ha / nutrient_percentage) * 100
    return fertilizer_amount

# Example usage
fertilizer_type = "Urea"  # Replace with desired fertilizer
nutrient_type = "N"       # Replace with "N", "P", or "K"
nutrient_kg_per_ha = 50   # Replace with the required kg/ha nutrient

amount = calculate_fertilizer_amount(fertilizer_type, nutrient_type, nutrient_kg_per_ha)
print(f"The amount of {fertilizer_type} needed is: {amount} kg/ha")
