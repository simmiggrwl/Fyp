import pickle

# Replace 'your_file.pkl' with the actual path to your .pkl file
file_path = '/home/shruti/Documents/FYP/mc_uav/data/op/1depots/3agents/coop/km/20/train_seed1111_L2.pkl'

try:
    with open(file_path, 'rb') as file:
        # Load the content of the pickle file
        data = pickle.load(file)

        # Print or process the loaded data
        print(data)
except FileNotFoundError:
    print(f"The file {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")