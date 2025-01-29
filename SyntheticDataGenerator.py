################################################################################################################################################
#                                                                                                                                              #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt    #
#                                                                                                                                              #
################################################################################################################################################

### Generic Python Code developed in assistance with ChatGPT 3.5

import pandas as pd
import numpy as np
from faker import Faker

# Function to generate a larger dataset from a smaller dataset
def generate_large_dataset(input_file, output_file, num_rows):
    # Load the basic dataset
    basic_data = pd.read_csv(input_file)
    num_original_rows = basic_data.shape[0]
    
    # Initialize Faker for generating synthetic data
    faker = Faker()
    
    # Create a copy of the dataset to expand
    large_dataset = basic_data.copy()

    # Generate additional rows
    while large_dataset.shape[0] < num_rows:
        # Select a random row to duplicate and slightly modify
        random_row = basic_data.sample(1).iloc[0]
        
        # Modify the data randomly
        synthetic_row = {}
        for col in basic_data.columns:
            if basic_data[col].dtype == 'object':  # For categorical/text data
                if "name" in col.lower():
                    synthetic_row[col] = faker.name()
                elif "address" in col.lower():
                    synthetic_row[col] = faker.address()
                elif "email" in col.lower():
                    synthetic_row[col] = faker.email()
                else:
                    synthetic_row[col] = random_row[col]
            elif np.issubdtype(basic_data[col].dtype, np.number):  # For numeric data
                noise = np.random.normal(0, 0.1 * (basic_data[col].max() - basic_data[col].min()))
                synthetic_row[col] = max(min(random_row[col] + noise, basic_data[col].max()), basic_data[col].min())
            elif np.issubdtype(basic_data[col].dtype, np.datetime64):  # For datetime data
                synthetic_row[col] = pd.to_datetime(random_row[col]) + pd.to_timedelta(np.random.randint(-30, 30), unit='d')
            else:
                synthetic_row[col] = random_row[col]
        
        # Append the synthetic row to the dataset
        large_dataset = pd.concat([large_dataset, pd.DataFrame([synthetic_row])], ignore_index=True)

    # Shuffle the dataset
    large_dataset = large_dataset.sample(frac=1).reset_index(drop=True)

    # Save the larger dataset to a CSV file
    large_dataset.to_csv(output_file, index=False)
    print(f"Larger dataset with {num_rows} rows saved to {output_file}!")

# Example usage
if __name__ == "__main__":
    # Path to your input basic dataset
    input_csv = "model_dataset.csv"
    # Path to save the generated large dataset
    output_csv = "BigDataGenerated.csv"
    # Number of rows to generate
    target_rows = 2500 # Change this value as needed
    
    generate_large_dataset(input_csv, output_csv, target_rows)

# Input dataset has to be set at line 62. Generated output dataset is specified at line 64.
