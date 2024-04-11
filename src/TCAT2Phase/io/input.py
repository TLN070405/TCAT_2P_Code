"""input.py"""




def read_txt_file(filename):
    """
    Read external text file
    """
    try:
        with open(filename, 'r') as file:
            # Read the lines of the file
            lines = file.readlines()
            numbers = []
            for line in lines:
                # Split each line into numbers
                line_numbers = line.strip().split()
                line_numbers=line_numbers[1:]
                # Convert the numbers from strings to floats and append to the list
                numbers.extend([float(num) for num in line_numbers])
            return numbers
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
