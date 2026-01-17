import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from bot import ScreenBot
import time
import cv2
import os
import numpy as np

def print_help():
    print("""
Available commands:
  help                    - Show this help message
  find text <text>        - Find text on screen using OCR
  click text <text> [n]   - Click nth occurrence of text (default: first)
  find object <class>     - Find object class using YOLO
  click object <class> [n]- Click nth occurrence of object (default: first)
  list objects            - List all detected objects on screen
  click <x> <y>           - Click at specific coordinates
  type <text>             - Type text
  press <key>             - Press a keyboard key (e.g., 'enter', 'space', 'esc')
  wait <seconds>          - Wait for specified number of seconds (e.g., 'wait 2', 'wait 1.5')
  screenshot              - Save current screenshot
  visualize text          - Show all detected text with bounding boxes
  visualize objects       - Show all detected objects with bounding boxes
  visualize all           - Show both text and objects with bounding boxes
  region set <x> <y> <w> <h> - Set screen region (only detect in this area)
  region clear            - Clear region (detect on full screen)
  region show             - Show current region settings
  horizontal [repeat]     - Run built-in horizontal sequence
  vertical [repeat]       - Run built-in vertical sequence
  runfile <file> [repeat] - Execute commands from a custom file (e.g., 'runfile myfile.txt 5')
  exit/quit               - Exit the bot

Examples:
  > find text "Login"
  > click text "Submit" 0
  > find object "button"
  > click object "person" 1
  > list objects
  > click 500 300
  > type "Hello World"
  > press enter
  > horizontal
  > horizontal 5
  > vertical
  > runfile myfile.txt
  > visualize text
  > visualize all
""")

def visualize_text_detections(bot):
    """Visualize all OCR text detections with bounding boxes"""
    print("Scanning screen for text...")
    # Use cropped region for detection, full screen for visualization
    screen_img = bot.take_screenshot()
    all_text = bot.get_all_ocr_text(screen_img)
    
    if not all_text:
        print("No text detected on screen")
        return
    
    # Get full screen for visualization
    full_img = bot.take_screenshot(full_screen=True)
    vis_img = full_img.copy()
    
    # Draw region border if region is set
    if bot.screen_region is not None:
        x, y, width, height = bot.screen_region
        cv2.rectangle(vis_img, (x, y), (x + width, y + height), (255, 255, 0), 3)
    
    # Draw bounding boxes and labels
    for center_x, center_y, bbox, text, confidence in all_text:
        # Draw bounding box polygon
        bbox_points = np.array(bbox, dtype=np.int32)
        cv2.polylines(vis_img, [bbox_points], True, (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw text label
        label = f"{text} ({confidence:.2f})"
        label_pos = (int(bbox[0][0]), int(bbox[0][1]) - 10)
        cv2.putText(vis_img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save visualization
    filename = f"visualization_text_{int(time.time())}.png"
    cv2.imwrite(filename, vis_img)
    print(f"✓ Detected {len(all_text)} text elements")
    print(f"✓ Visualization saved as: {filename}")
    
    return filename

def visualize_object_detections(bot):
    """Visualize all YOLO object detections with bounding boxes"""
    print("Scanning screen for objects...")
    # Use cropped region for detection, full screen for visualization
    screen_img = bot.take_screenshot()
    detections = bot.find_objects_yolo(screen_img=screen_img, return_bbox=True)
    
    if not detections:
        print("No objects detected on screen")
        return
    
    # Get full screen for visualization
    full_img = bot.take_screenshot(full_screen=True)
    vis_img = full_img.copy()
    
    # Draw region border if region is set
    if bot.screen_region is not None:
        x, y, width, height = bot.screen_region
        cv2.rectangle(vis_img, (x, y), (x + width, y + height), (255, 255, 0), 3)
    
    # Draw bounding boxes and labels
    for center_x, center_y, class_name, confidence, (x1, y1, x2, y2) in detections:
        # Draw bounding box rectangle
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw center point
        cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw label
        label = f"{class_name} ({confidence:.2f})"
        label_pos = (x1, y1 - 10)
        cv2.putText(vis_img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Save visualization
    filename = f"visualization_objects_{int(time.time())}.png"
    cv2.imwrite(filename, vis_img)
    print(f"✓ Detected {len(detections)} objects")
    print(f"✓ Visualization saved as: {filename}")
    
    return filename

def visualize_all_detections(bot):
    """Visualize both OCR text and YOLO objects with bounding boxes"""
    print("Scanning screen for text and objects...")
    # Use cropped region for detection, full screen for visualization
    screen_img = bot.take_screenshot()
    all_text = bot.get_all_ocr_text(screen_img)
    detections = bot.find_objects_yolo(screen_img=screen_img, return_bbox=True)
    
    if not all_text and not detections:
        print("No text or objects detected on screen")
        return
    
    # Get full screen for visualization
    full_img = bot.take_screenshot(full_screen=True)
    vis_img = full_img.copy()
    
    # Draw region border if region is set (yellow border)
    if bot.screen_region is not None:
        x, y, width, height = bot.screen_region
        cv2.rectangle(vis_img, (x, y), (x + width, y + height), (255, 255, 0), 3)
    
    # Draw text bounding boxes (green)
    for center_x, center_y, bbox, text, confidence in all_text:
        bbox_points = np.array(bbox, dtype=np.int32)
        cv2.polylines(vis_img, [bbox_points], True, (0, 255, 0), 2)
        cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)
        label = f"{text} ({confidence:.2f})"
        label_pos = (int(bbox[0][0]), int(bbox[0][1]) - 10)
        cv2.putText(vis_img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw object bounding boxes (blue)
    for center_x, center_y, class_name, confidence, (x1, y1, x2, y2) in detections:
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)
        label = f"{class_name} ({confidence:.2f})"
        label_pos = (x1, y1 - 10)
        cv2.putText(vis_img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save visualization
    filename = f"visualization_all_{int(time.time())}.png"
    cv2.imwrite(filename, vis_img)
    print(f"✓ Detected {len(all_text)} text elements and {len(detections)} objects")
    print(f"✓ Visualization saved as: {filename}")
    print(f"  - Green boxes: Text (OCR)")
    print(f"  - Blue boxes: Objects (YOLO)")
    print(f"  - Red dots: Click centers")
    if bot.screen_region is not None:
        print(f"  - Yellow border: Detection region")
    
    return filename

def execute_single_command(bot, command: str, retry_count: int = 3, retry_delay: float = 1.5) -> bool:
    """
    Execute a single command with retry logic
    
    Args:
        bot: ScreenBot instance
        command: Command string to execute
        retry_count: Number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.5)
    
    Returns:
        True if command succeeded, False otherwise
    """
    parts = command.split()
    if not parts:
        return False
    
    cmd = parts[0].lower()
    
    for attempt in range(retry_count):
        try:
            if cmd == 'find' and len(parts) >= 3:
                mode = parts[1].lower()
                target = ' '.join(parts[2:])
                
                if mode == 'text':
                    matches = bot.find_text_ocr(target)
                    if matches:
                        if attempt > 0:
                            print(f"    ✓ Found on retry attempt {attempt + 1}")
                        return True
                
                elif mode == 'object':
                    detections = bot.find_objects_yolo(target)
                    if detections:
                        if attempt > 0:
                            print(f"    ✓ Found on retry attempt {attempt + 1}")
                        return True
            
            elif cmd == 'click':
                if len(parts) == 3:
                    # Direct coordinates - no retry needed, always succeeds
                    try:
                        x, y = int(parts[1]), int(parts[2])
                        bot.click(x, y)
                        return True
                    except ValueError:
                        print(f"  ✗ Invalid coordinates: {parts[1]} {parts[2]}")
                        return False
                
                elif len(parts) >= 3:
                    mode = parts[1].lower()
                    target = ' '.join(parts[2:-1]) if len(parts) > 3 else parts[2]
                    index = int(parts[-1]) if len(parts) > 3 and parts[-1].isdigit() else 0
                    
                    if mode == 'text':
                        success = bot.find_and_click_text(target, index)
                        if success:
                            if attempt > 0:
                                print(f"    ✓ Clicked on retry attempt {attempt + 1}")
                            return True
                    
                    elif mode == 'object':
                        success = bot.find_and_click_object(target, index)
                        if success:
                            if attempt > 0:
                                print(f"    ✓ Clicked on retry attempt {attempt + 1}")
                            return True
            
            elif cmd == 'type' and len(parts) >= 2:
                text = ' '.join(parts[1:])
                bot.type_text(text)
                return True
            
            elif cmd == 'press' and len(parts) >= 2:
                key = parts[1].lower()
                bot.press_key(key)
                return True
            
            elif cmd == 'list' and len(parts) >= 2 and parts[1].lower() == 'objects':
                bot.list_available_objects()
                return True
            
            elif cmd == 'wait' and len(parts) >= 2:
                # Wait command - doesn't need retries, just execute once
                try:
                    wait_time = float(parts[1])
                    if attempt == 0:  # Only print once
                        print(f"  ⏳ Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    if attempt == 0:
                        print(f"  ✓ Wait complete")
                    return True
                except ValueError:
                    if attempt == 0:
                        print(f"  ✗ Invalid wait time: {parts[1]}")
                    return False
            
            else:
                if attempt == 0:
                    print(f"  ✗ Unknown command: {command}")
                return False
        
        except Exception as e:
            if attempt == retry_count - 1:
                print(f"  ✗ Error: {e}")
            # Continue to retry on exceptions too
    
        # Retry with delay (except on last attempt)
        if attempt < retry_count - 1:
            print(f"    ⏳ Retrying in {retry_delay}s... (attempt {attempt + 2}/{retry_count})")
            time.sleep(retry_delay)
    
    return False

def execute_command_strings(bot, command_string: str, retry_count: int = 3, retry_delay: float = 1.5):
    """Execute commands from a string (same format as file) with retry logic and advanced flow control"""
    raw_lines = command_string.strip().split('\n')
    
    commands_with_directives = []
    current_command_info = {'command': None, 'if_fail_then': None, 'loop_if_success': None, 'stop_on_fail': False}
    
    for line_num, line in enumerate(raw_lines, 1):
        stripped_line = line.strip()
        
        if not stripped_line:
            continue
        
        if stripped_line.startswith('#'):
            # This is a directive
            directive_content = stripped_line[1:].strip()  # Remove '#' and any leading whitespace
            if not directive_content:  # Skip if just '#'
                continue
            
            directive_parts = directive_content.split(' ', 1)
            directive_name = directive_parts[0].upper()
            directive_value = directive_parts[1] if len(directive_parts) > 1 else None

            if current_command_info['command'] is None:
                print(f"Warning: Directive '{stripped_line}' on line {line_num} has no preceding command. Skipping.")
                continue
            
            if directive_name == 'IF_FAIL_THEN':
                current_command_info['if_fail_then'] = directive_value
            elif directive_name == 'LOOP_IF_SUCCESS':
                try:
                    current_command_info['loop_if_success'] = int(directive_value)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid line number for LOOP_IF_SUCCESS on line {line_num}. Skipping.")
            elif directive_name == 'STOP_ON_FAIL':
                current_command_info['stop_on_fail'] = True
            else:
                print(f"Warning: Unknown directive '{directive_name}' on line {line_num}. Skipping.")
        else:
            # This is a command
            if current_command_info['command'] is not None:
                # Save the previous command and its directives
                commands_with_directives.append(current_command_info)
            # Start a new command info block
            current_command_info = {'command': stripped_line, 'if_fail_then': None, 'loop_if_success': None, 'stop_on_fail': False, 'original_line': line_num}
    
    # Add the last command if it exists
    if current_command_info['command'] is not None:
        commands_with_directives.append(current_command_info)

    if not commands_with_directives:
        print("No commands found in the command string.")
        return False

    current_command_idx = 0
    success_count = 0
    total_commands_attempted = 0
    
    while current_command_idx < len(commands_with_directives):
        cmd_info = commands_with_directives[current_command_idx]
        command = cmd_info['command']
        original_line = cmd_info['original_line']
        
        total_commands_attempted += 1
        print(f"\n[{original_line}] Executing: {command}")
        
        command_succeeded = execute_single_command(bot, command, retry_count, retry_delay)
        
        if command_succeeded:
            success_count += 1
            if cmd_info['loop_if_success'] is not None:
                target_line = cmd_info['loop_if_success']
                # Find the index of the command at the target line number
                found_idx = -1
                for i, info in enumerate(commands_with_directives):
                    if info['original_line'] == target_line:
                        found_idx = i
                        break
                
                if found_idx != -1:
                    print(f"  ✓ Command succeeded. Looping to line {target_line}. (New index: {found_idx})")
                    current_command_idx = found_idx
                    continue # Restart the while loop from the new index
                else:
                    print(f"  ✗ LOOP_IF_SUCCESS target line {target_line} not found. Continuing to next command.")
                    current_command_idx += 1
            else:
                current_command_idx += 1 # Move to the next command normally
        else:
            print(f"  ✗ Command failed after {retry_count} attempts: {command}")
            
            if cmd_info['if_fail_then'] is not None:
                alternative_command = cmd_info['if_fail_then']
                print(f"  Trying alternative command: {alternative_command}")
                alt_succeeded = execute_single_command(bot, alternative_command, retry_count, retry_delay)
                if alt_succeeded:
                    print(f"  ✓ Alternative command succeeded.")
                    success_count += 1 # Count alternative command as success
                else:
                    print(f"  ✗ Alternative command also failed.")
            
            if cmd_info['stop_on_fail']:
                print("  STOP_ON_FAIL directive encountered. Stopping preset execution.")
                break
            
            current_command_idx += 1 # Move to the next command even if failed (unless stopped)
    
    print("\n" + "-" * 50)
    print(f"Preset execution complete: {success_count}/{total_commands_attempted} commands succeeded (including alternatives).")
    return True

def execute_command_file(bot, filename: str, retry_count: int = 3, retry_delay: float = 1.5):
    """Execute commands from a preset file with retry logic and advanced flow control"""
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        return False
    
    print(f"Executing preset: {filename}")
    print(f"Retry settings: {retry_count} attempts, {retry_delay}s delay")
    print("-" * 50)
    
    with open(filename, 'r') as f:
        command_string = f.read()
    
    return execute_command_strings(bot, command_string, retry_count, retry_delay)

def main():
    print("Screen Automation Bot - Interactive Mode")
    print("Type 'help' for commands, 'exit' to quit\n")
    
    bot = ScreenBot()
    
    while True:
        try:
            command = input("> ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            elif cmd == 'help':
                print_help()
            
            elif cmd == 'find' and len(parts) >= 3:
                mode = parts[1].lower()
                target = ' '.join(parts[2:])
                
                if mode == 'text':
                    matches = bot.find_text_ocr(target)
                    if matches:
                        print(f"Found at: {matches}")
                    else:
                        print(f"Text '{target}' not found")
                
                elif mode == 'object':
                    detections = bot.find_objects_yolo(target)
                    if detections:
                        print(f"Found {len(detections)} object(s):")
                        for i, (x, y, cls, conf) in enumerate(detections):
                            print(f"  {i}: {cls} at ({x}, {y}) - conf: {conf:.2f}")
                    else:
                        print(f"Object class '{target}' not found")
                
                else:
                    print(f"Unknown find mode: {mode}. Use 'text' or 'object'")
            
            elif cmd == 'click':
                if len(parts) == 3:
                    # Direct coordinates
                    try:
                        x, y = int(parts[1]), int(parts[2])
                        bot.click(x, y)
                    except ValueError:
                        print("Invalid coordinates. Use: click <x> <y>")
                
                elif len(parts) >= 3:
                    mode = parts[1].lower()
                    target = ' '.join(parts[2:-1]) if len(parts) > 3 else parts[2]
                    index = int(parts[-1]) if len(parts) > 3 and parts[-1].isdigit() else 0
                    
                    if mode == 'text':
                        success = bot.find_and_click_text(target, index)
                        if not success:
                            print("Failed to find and click text")
                    
                    elif mode == 'object':
                        success = bot.find_and_click_object(target, index)
                        if not success:
                            print("Failed to find and click object")
                    
                    else:
                        print(f"Unknown click mode: {mode}. Use 'text' or 'object'")
                else:
                    print("Usage: click <mode> <target> [index] or click <x> <y>")
            
            elif cmd == 'list' and len(parts) >= 2 and parts[1].lower() == 'objects':
                bot.list_available_objects()
            
            elif cmd == 'type' and len(parts) >= 2:
                text = ' '.join(parts[1:])
                bot.type_text(text)
            
            elif cmd == 'press' and len(parts) >= 2:
                key = parts[1].lower()
                bot.press_key(key)
            
            elif cmd == 'wait' and len(parts) >= 2:
                try:
                    wait_time = float(parts[1])
                    print(f"Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    print("Wait complete")
                except ValueError:
                    print(f"Invalid wait time: {parts[1]}")
            
            elif cmd == 'screenshot':
                img = bot.take_screenshot()
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, img)
                print(f"Screenshot saved as {filename}")
            
            elif cmd == 'visualize':
                if len(parts) < 2:
                    print("Usage: visualize <text|objects|all>")
                else:
                    mode = parts[1].lower()
                    if mode == 'text':
                        visualize_text_detections(bot)
                    elif mode == 'objects':
                        visualize_object_detections(bot)
                    elif mode == 'all':
                        visualize_all_detections(bot)
                    else:
                        print(f"Unknown visualize mode: {mode}. Use 'text', 'objects', or 'all'")
            
            elif cmd == 'region':
                if len(parts) < 2:
                    print("Usage: region <set|clear|show>")
                    print("  region set <x> <y> <width> <height> - Set detection region")
                    print("  region clear - Use full screen")
                    print("  region show - Show current region")
                else:
                    subcmd = parts[1].lower()
                    if subcmd == 'set' and len(parts) >= 6:
                        try:
                            x, y, width, height = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                            bot.set_screen_region(x, y, width, height)
                        except ValueError:
                            print("Error: All region values must be integers")
                    elif subcmd == 'clear':
                        bot.clear_screen_region()
                    elif subcmd == 'show':
                        if bot.screen_region is None:
                            print("No region set - using full screen")
                        else:
                            x, y, width, height = bot.screen_region
                            print(f"Current region: x={x}, y={y}, width={width}, height={height}")
                    else:
                        print(f"Unknown region command: {subcmd}")
            
            elif cmd == 'horizontal':
                repeat_count = 1
                if len(parts) >= 2:
                    try:
                        repeat_count = int(parts[1])
                        if repeat_count < 1:
                            print("Repeat count must be at least 1")
                            continue
                    except ValueError:
                        print(f"Invalid repeat count: {parts[1]}. Using default (1)")
                        repeat_count = 1
                
                print("Executing built-in horizontal sequence")
                print("Retry settings: 3 attempts, 1.5s delay")
                print("-" * 50)
                
                # Built-in horizontal sequence
                horizontal_commands = """click text x20 2
# IF_FAIL_THEN click text x5o 2
click text ok 1 
wait 3
click 1800 1000 
wait 1
click 1800 300
wait 3
click text close 1 
wait 4
click text cancel 1
# STOP_ON_FAIL
# LOOP_IF_SUCCESS 1"""
                
                for iteration in range(repeat_count):
                    if repeat_count > 1:
                        print(f"\n{'='*50}")
                        print(f"ITERATION {iteration + 1} of {repeat_count}")
                        print(f"{'='*50}\n")
                    execute_command_strings(bot, horizontal_commands)
                    if iteration < repeat_count - 1:
                        print(f"\n⏳ Waiting 1 second before next iteration...\n")
                        time.sleep(1)
            
            elif cmd == 'vertical':
                repeat_count = 1
                if len(parts) >= 2:
                    try:
                        repeat_count = int(parts[1])
                        if repeat_count < 1:
                            print("Repeat count must be at least 1")
                            continue
                    except ValueError:
                        print(f"Invalid repeat count: {parts[1]}. Using default (1)")
                        repeat_count = 1
                
                print("Executing built-in vertical sequence")
                print("Retry settings: 3 attempts, 1.5s delay")
                print("-" * 50)
                
                # Built-in vertical sequence
                vertical_commands = """click text x20 2
# IF_FAIL_THEN click text x5o 2
click text ok 1 
wait 3
click 900 1030 
wait 1
click 900 700
wait 3
click text close 1 
wait 4
click text cancel 1
# STOP_ON_FAIL
# LOOP_IF_SUCCESS 1"""
                
                for iteration in range(repeat_count):
                    if repeat_count > 1:
                        print(f"\n{'='*50}")
                        print(f"ITERATION {iteration + 1} of {repeat_count}")
                        print(f"{'='*50}\n")
                    execute_command_strings(bot, vertical_commands)
                    if iteration < repeat_count - 1:
                        print(f"\n⏳ Waiting 1 second before next iteration...\n")
                        time.sleep(1)
            
            elif cmd == 'runfile' and len(parts) >= 2:
                preset_file = parts[1]
                repeat_count = 1
                
                # Check if repeat count is specified
                if len(parts) >= 3:
                    try:
                        repeat_count = int(parts[2])
                        if repeat_count < 1:
                            print("Repeat count must be at least 1")
                            continue
                    except ValueError:
                        print(f"Invalid repeat count: {parts[2]}. Using default (1)")
                        repeat_count = 1
                
                # Execute the preset file multiple times
                for iteration in range(repeat_count):
                    if repeat_count > 1:
                        print(f"\n{'='*50}")
                        print(f"ITERATION {iteration + 1} of {repeat_count}")
                        print(f"{'='*50}\n")
                    execute_command_file(bot, preset_file)
                    
                    # Add a small delay between iterations (except after the last one)
                    if iteration < repeat_count - 1:
                        print(f"\n⏳ Waiting 1 second before next iteration...\n")
                        time.sleep(1)
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit or continue with commands.")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
