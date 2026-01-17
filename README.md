# Umamusume Story Skipper 

A Python bot that can find and click elements on your screen using OCR (Optical Character Recognition) and YOLO object detection. Perfect for automating tasks where you need to interact with buttons, text, or objects on screen.


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
umastoryskipper/
├── bot.py                 # Core bot functionality (OCR, YOLO, automation)
├── bot_gui.py            # Graphical user interface with live preview
├── interactive_bot.py    # Command-line interactive mode
├── launcher.py           # Simple launcher menu (GUI/CLI/Exit)
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules (excludes models, cache, generated files)
└── README.md            # This file
```

## Usage

### Launcher (Recommended)
Run the launcher for an easy menu to choose between GUI and CLI modes:
```bash
python launcher.py
```

### GUI Mode
Launch the graphical interface:
```bash
python bot_gui.py
```

### CLI Mode
Use the command-line interface:
```bash
python interactive_bot.py
```

### Interactive Mode (Recommended)

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
> type "Hello World"
> press enter
```

### Command Line Mode

**Find text using OCR:**
```bash
python bot.py --mode text --target "Login Button"
python bot.py --mode text --target "Login Button" --click
```

**Find objects using YOLO:**
```bash
python bot.py --mode object --target "button" --click
python bot.py --mode object --target "person" --index 1 --click
```

**List all detected objects:**
```bash
python bot.py --mode list
```

**Options:**
- `--mode`: `text`, `object`, or `list`
- `--target`: Text or object class to find
- `--index`: Which occurrence to use (0 = first, default)
- `--confidence`: YOLO confidence threshold (0.0-1.0, default: 0.5)
- `--click`: Actually click the found element (otherwise just locates)

## Examples

**Click a login button:**
```bash
python interactive_bot.py
> click text "Login"
```

**Find and click the second person detected:**
```bash
python interactive_bot.py
> click object "person" 1
```

**See what objects are on screen:**
```bash
python interactive_bot.py
> list objects
```

**Click at specific coordinates:**
```bash
python interactive_bot.py
> click 640 480
```

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


