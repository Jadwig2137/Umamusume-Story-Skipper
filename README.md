# Umamusume Story Skipper 

A Python bot that can find and click elements on your screen using OCR (Optical Character Recognition) and YOLO object detection. Perfect for getting free Carats without paying attention 😎  (most of the time lol)


## Features

- **OCR Text Detection**: Find and click text on screen using EasyOCR
- **YOLO Object Detection**: Detect and interact with objects using YOLOv8
- **Flexible Interface**: Command-line arguments or interactive mode
- **Multiple Detection**: Handle multiple occurrences of the same element

## Requirements

- Python 3.8+
- Windows/Linux/macOS
- GPU optional but recommended for faster OCR/YOLO performance

## Installation

1. **Install PyTorch with CUDA support (Optional but recommended, If you don't want it skip to step 3):**

   For CUDA 12.9:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
   ```

   For CUDA 12.1/12.4 (backward compatible):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   For CUDA 11.8:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Verify GPU is detected:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
   Should print `CUDA available: True` if GPU is working correctly.

3. **Install other Python dependencies:**
```bash
pip install -r requirements.txt
```   

4. **Note on Models:**
   - EasyOCR will automatically download its models on first use
   - YOLO will automatically download the `yolov8n.pt` model on first use (~6.5MB)
   - First run may take a few minutes to download models
   - Downloaded models are ignored by git to keep repository size small

## Project Structure

```
Umamusume-Story-Skipper/
├── bot.py                 # Core bot functionality (OCR, YOLO, automation)
├── bot_gui.py            # Graphical user interface with live preview
├── interactive_bot.py    # Command-line interactive mode
├── launcher.py           # Simple launcher menu (GUI/CLI/Exit)
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules (excludes models, cache, generated files)
└── README.md            # This file
```

## Usage

### Launcher 
Run the launcher for an easy menu to choose between GUI and CLI modes:
```bash
python launcher.py
```

### GUI Mode (Recommended for ease of use) 
Launch the graphical interface:
```bash
python bot_gui.py
```

### Interactive Mode (Recommended for advanced users)

The easiest way to use the bot:

```bash
python interactive_bot.py
```

Then use commands like:
```
> find text "Login"
> click text "Submit"
> find object "button"
> click object "person" 1
> list objects
> click 500 300
> press enter
```
## Ability to create your own automation

Create a TXT file, input all of the commands that you'd want the program to make (all commands under Interactive Mode), save the file, and either choose it in custom files (GUI), or use the ```run filename.txt``` command in Interactive Mode.

There's 3 other functions besides all of those commands like: 

- ```# IF_FAIL_THEN "command"``` : A simple IF statement, if **true** continue execution, if **false** then do the command specified.
- ```# STOP_ON_FAIL``` : Stops execution after *3* attempts (Number of attempts changable for advanced users on lines **186**, **301** and **412**)
- ```# LOOP_IF_SUCCESS "line-number"``` : Simple loop function that goes back to the line defined by user after completing the whole sequence.


## Safety Features

- **Fail-safe**: Move mouse to top-left corner to emergency stop
- **Pause between actions**: Built-in delays to prevent accidental rapid clicks
- **Confidence thresholds**: Only act on high-confidence detections

## Tips

1. **OCR works best with**: Clear text, good contrast, standard fonts
2. **YOLO works best with**: Common objects from its training set
3. **Confidence**: Lower confidence (e.g., 0.3) finds more objects but may have false positives
4. **Multiple occurrences**: Use `--index` or specify index in interactive mode to click the 2nd, 3rd, etc. occurrence

## Troubleshooting

**Models not downloading:**
- Check internet connection (models download on first use)
- For YOLO, manually download from: https://github.com/ultralytics/assets/releases

**OCR not finding text:**
- Ensure text is visible and not too small
- Try adjusting screen resolution
- Text matching is case-insensitive but must be exact substring match

**YOLO not detecting objects:**
- Try lowering `--confidence` threshold
- Object must be a common class from COCO dataset
- Ensure good lighting and clear view of object


