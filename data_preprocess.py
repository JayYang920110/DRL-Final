import re
import pickle # For saving the processed data in the expected format

def convert_single_coord_to_int_tuple(coord_part_string, board_size=19):
    """
    Converts a single coordinate part (e.g., 'j10', 'k9')
    to an (row, column) integer tuple.

    Assumes format: [column_letter][row_number].
    Column 'a' maps to 0, 'b' to 1, etc.
    Row numbers are 1-based, converting to 0-based index.
    """
    match = re.match(r'([a-z])(\d+)', coord_part_string)
    if not match:
        # print(f"Warning: Could not parse single coordinate part: {coord_part_string}")
        return None

    col_letter = match.group(1)
    row_num_str = match.group(2)

    column = ord(col_letter) - ord('a')
    try:
        row = int(row_num_str) - 1
    except ValueError:
        # print(f"Warning: Could not convert row number to int: {row_num_str}")
        return None

    if not (0 <= row < board_size and 0 <= column < board_size):
        # print(f"Warning: Coordinate ({row}, {column}) from '{coord_part_string}' is out of bounds (0-{board_size-1}) for board size {board_size}x{board_size}.")
        return None

    return (row, column)


def parse_connect6_record(record_string, board_size=19):
    """
    Parses a single Connect6 game record string and yields moves.
    Each yield will be (player, [coord1, coord2]) where coord1 and coord2 are (row, col) tuples.
    Assumes Connect6 format where a turn involves placing exactly two stones.
    """
    move_pattern = re.compile(r'(?:;)?([WB])\[([^\]]+)\]')

    for match in move_pattern.finditer(record_string):
        player = match.group(1) # 'W' or 'B'
        raw_coords_string = match.group(2) # e.g., 'j10k9'

        # This regex now expects two coordinate parts immediately following each other
        # This will need careful adjustment if your SGF uses spaces like "j10 k9"
        coord_parts_pattern = re.compile(r'([a-z]\d+)([a-z]\d+)')
        coord_match = coord_parts_pattern.match(raw_coords_string)

        if not coord_match:
            # Handle cases with single stone (like resignation 'W[]' or 'B[]') or malformed
            if raw_coords_string.strip() == "": # Pass/Resign move represented by empty brackets
                yield (player, []) # Represent a pass/resign
                continue
            # print(f"Warning: Unexpected coordinate format for Connect6 move: '{raw_coords_string}'. Skipping this move.")
            continue

        coord_str1 = coord_match.group(1)
        coord_str2 = coord_match.group(2)

        int_coord1 = convert_single_coord_to_int_tuple(coord_str1, board_size)
        int_coord2 = convert_single_coord_to_int_tuple(coord_str2, board_size)

        if int_coord1 and int_coord2: # Both coordinates must be valid
            yield (player, [int_coord1, int_coord2])


def get_match_trajectories_connect6(file_path, board_size=19):
    """
    Reads Connect6 game records from a file and returns a list of trajectories.
    Each trajectory is a list of Connect6 moves, where each move is a list of two (row, col) tuples.
    Example output: [[(9,9), (8,10)], [(11,9), (12,12)], ...]
    """
    all_trajectories = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                cleaned_line = line.strip()

                if cleaned_line:
                    current_game_moves = []
                    # parse_connect6_record now yields (player, [coord1, coord2]) or (player, []) for pass
                    for player, stone_placements in parse_connect6_record(cleaned_line, board_size):
                        if stone_placements: # Ensure it's not an empty move (pass/resign)
                            current_game_moves.append(stone_placements)
                        # For Connect6, we might want to represent a "pass" or "resign" turn.
                        # If a turn has no stone placements, it could signify a pass/resign.
                        # You'll need to define how your Connect6Game handles passes.
                        # For now, we only append actual stone placements.

                    if current_game_moves:
                        all_trajectories.append(current_game_moves)

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found. Please check the path and filename.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return all_trajectories

def save_connect6_trajectories_to_pickle(trajectories, output_file_path):
    """
    Saves the list of Connect6 trajectories to a pickle file.
    Each trajectory is a list of lists, where each inner list contains two (row, col) tuples.
    """
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(trajectories, f)
        print(f"Connect6 trajectories saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred while saving trajectories to pickle: {e}")

def main():
    input_file_path = 'raw_record.sgf'  # Replace with your input file path
    output_file_path = 'processed_connect6_records.txt'  # Desired output pickle file path

    # Assuming Connect6 board size is 19x19 for this example
    board_size = 19
    trajectories = get_match_trajectories_connect6(input_file_path, board_size)

    if trajectories is not None:
        save_connect6_trajectories_to_pickle(trajectories, output_file_path)

if __name__ == "__main__":
    main()