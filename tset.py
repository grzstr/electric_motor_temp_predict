# Checking the availability and location of keras.wrappers in the current environment
import importlib.util

# Define the module name
module_name = 'tensorflow.keras.wrappers.scikit_learn'

# Check if the module is available
module_available = importlib.util.find_spec(module_name) is not None

module_available