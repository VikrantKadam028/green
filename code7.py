import os

def check_file_exists(filename_path):
    # This function checks if a file exists at the given path.
    # Simplified: os.path.exists directly returns a boolean.
    return os.path.exists(filename_path)

def read_file_content(filepath):
    # Reads content of a file
    # Removed redundant check_file_exists call.
    # Improved error handling to distinguish FileNotFoundError.
    try:
        with open(filepath, 'r') as f:
            file_content = f.read()
        return file_content
    except FileNotFoundError:
        # Specific message for file not found, as per original intent
        print(f"File not found: {filepath}")
        return None
    except IOError as e:
        # Catch other potential I/O errors (e.g., permissions, corrupted file)
        print(f"Error reading file {filepath}: {e}")
        return None

def process_data_lines(data_string):
    # Processes each line of the data string
    # Using a list comprehension for conciseness and efficiency.
    if not data_string:
        return []
    
    # Split by newline, strip whitespace from each line, convert to upper case,
    # and filter out any lines that become empty after stripping.
    processed_lines = [line.strip().upper() for line in data_string.split('\n') if line.strip()]
    return processed_lines

def main():
    config_file_name = "my_config.txt"
    current_dir = os.getcwd()
    config_filepath = os.path.join(current_dir, config_file_name)

    print(f"Checking for config file at: {config_filepath}")

    config_content = read_file_content(config_filepath)

    if config_content:
        print("\nConfig file content read successfully. Processing lines...")
        processed_data = process_data_lines(config_content)
        if processed_data:
            print("\nProcessed data:")
            for item in processed_data:
                print(f"- {item}")
        else:
            print("\nNo valid data found in config file after processing.")
    else:
        # This message covers cases where read_file_content returned None (error) or "" (empty file).
        print("\nFailed to read config file or file is empty.")

if __name__ == "__main__":
    main()
