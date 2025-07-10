import json
import os

try:
    file_path = 'test_results/certificate_accuracy_results.json'
    print(f"Loading {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        exit(1)
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("File loaded successfully!")
    
    # Check structure
    if 'detailed_results' not in data:
        print("Error: 'detailed_results' not found in data")
        print("Available keys:", list(data.keys()))
        exit(1)
    
    if 'validation' not in data['detailed_results']:
        print("Error: 'validation' not found in detailed_results")
        print("Available keys:", list(data['detailed_results'].keys()))
        exit(1)
        
    print("\nFailed validations:")
    print("-" * 60)
    
    failure_count = 0
    for system in data['detailed_results']['validation']['system_results']:
        for result in system['results']:
            if not result['validation_correct']:
                failure_count += 1
                print(f"\n{system['system_name']}: {result['certificate']}")
                print(f"  Expected: {result['expected_valid']}, Actual: {result['actual_valid']}")
                if 'violations' in result and result['violations']:
                    print(f"  First violation: {result['violations'][0]}")
    
    if failure_count == 0:
        print("No validation failures found! All tests passed.")
    else:
        print(f"\nTotal failures: {failure_count}")
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc() 