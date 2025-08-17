
import pandas as pd
import os

def main():
	input_path = input("Enter the path to the CSV file: ").strip()
	if not os.path.isfile(input_path):
		print(f"File not found: {input_path}")
		return
	try:
		df = pd.read_csv(input_path)
	except Exception as e:
		print(f"Error reading CSV: {e}")
		return

	print("Choose cleaning method:")
	print("1. Drop rows with missing values")
	print("2. Fill missing values with a default")
	choice = input("Enter 1 or 2: ").strip()

	if choice == '1':
		cleaned_df = df.dropna()
	elif choice == '2':
		fill_value = input("Enter value to fill missing cells: ")
		cleaned_df = df.fillna(fill_value)
	else:
		print("Invalid choice.")
		return

	output_path = input("Enter path to save cleaned CSV: ").strip()
	try:
		cleaned_df.to_csv(output_path, index=False)
		print(f"Cleaned CSV saved to {output_path}")
	except Exception as e:
		print(f"Error saving CSV: {e}")

if __name__ == "__main__":
	main()
