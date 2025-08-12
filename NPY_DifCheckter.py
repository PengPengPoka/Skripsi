import numpy as np

# Load the .npy files
arr1 = np.load('Testing Results\\HSV Read Test.npy')
arr2 = np.load('Testing Results\\HSV Tif Read Test.npy')

# Check if shape is the same
if arr1.shape != arr2.shape:
    print("❌ The arrays have different shapes:", arr1.shape, "vs", arr2.shape)
else:
    # Compare content
    if np.array_equal(arr1, arr2):
        print("✅ The arrays are exactly the same.")
    else:
        print("❌ The arrays have the same shape but different values.")
