import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import sys
from io import StringIO
from bot import ScreenBot
import time
import cv2
import numpy as np
from PIL import Image, ImageTk

class BotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Screen Automation Bot")
        self.root.geometry("1000x750")
        
        # Initialize bot in a separate thread-safe way
        self.bot = None
        self.bot_initializing = True
        
        # Queue for thread-safe GUI updates
        self.log_queue = queue.Queue()
        
        # Setup GUI
        self.setup_ui()
        
        # Start bot initialization in background
        self.init_bot()
        
        # Check for log messages
        self.root.after(100, self.process_log_queue)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Screen Automation Bot", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Initializing bot...", 
                                     foreground="orange")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # Tabbed interface
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Controls tab
        controls_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(controls_frame, text="Controls")
        
        self.setup_controls_tab(controls_frame)
        
        # Log tab
        log_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(log_frame, text="Log")
        
        self.setup_log_tab(log_frame)
        
        # Preview tab
        preview_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(preview_frame, text="Live Preview")
        
        self.setup_preview_tab(preview_frame)
        
        # Execution thread
        self.execution_thread = None
        self.stop_flag = threading.Event()
        
        # Selected file
        self.selected_file = None
        
        # Preview variables
        self.preview_active = False
        self.preview_thread = None
        self.preview_image = None
    
    def setup_controls_tab(self, parent):
        """Setup the controls tab"""
        # Built-in sequence buttons
        ttk.Label(parent, text="Built-in Sequences:", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.vertical_btn = ttk.Button(parent, text="▶ Vertical", 
                                       command=self.run_vertical, state="disabled")
        self.vertical_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Repeat count
        ttk.Label(parent, text="Repeat:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        self.repeat_var = tk.StringVar(value="1")
        repeat_spin = ttk.Spinbox(parent, from_=1, to=999, textvariable=self.repeat_var, width=10)
        repeat_spin.grid(row=2, column=1, sticky=tk.W, pady=(10, 5))
        
        # Separator
        ttk.Separator(parent, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Custom file button
        ttk.Label(parent, text="Custom Files:", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.runfile_btn = ttk.Button(parent, text="📁 Run Custom File", 
                                      command=self.run_custom_file, state="disabled")
        self.runfile_btn.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        self.current_file_label = ttk.Label(parent, text="No file selected", 
                                           foreground="gray", font=("Arial", 8))
        self.current_file_label.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Separator
        ttk.Separator(parent, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Visualization buttons
        ttk.Label(parent, text="Visualization:", font=("Arial", 10, "bold")).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.viz_text_btn = ttk.Button(parent, text="🔍 Text Detection", 
                                       command=self.visualize_text, state="disabled")
        self.viz_text_btn.grid(row=9, column=0, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
        
        self.viz_all_btn = ttk.Button(parent, text="🔍 All Detections", 
                                      command=self.visualize_all, state="disabled")
        self.viz_all_btn.grid(row=9, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Separator
        ttk.Separator(parent, orient="horizontal").grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Region settings
        ttk.Label(parent, text="Screen Region:", font=("Arial", 10, "bold")).grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        region_frame = ttk.Frame(parent)
        region_frame.grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(region_frame, text="X:").grid(row=0, column=0, padx=(0, 2))
        self.region_x = ttk.Entry(region_frame, width=6)
        self.region_x.grid(row=0, column=1, padx=2)
        self.region_x.insert(0, "0")
        
        ttk.Label(region_frame, text="Y:").grid(row=0, column=2, padx=(10, 2))
        self.region_y = ttk.Entry(region_frame, width=6)
        self.region_y.grid(row=0, column=3, padx=2)
        self.region_y.insert(0, "0")
        
        ttk.Label(region_frame, text="W:").grid(row=1, column=0, padx=(0, 2), pady=(5, 0))
        self.region_w = ttk.Entry(region_frame, width=6)
        self.region_w.grid(row=1, column=1, padx=2, pady=(5, 0))
        self.region_w.insert(0, "1000")
        
        ttk.Label(region_frame, text="H:").grid(row=1, column=2, padx=(10, 2), pady=(5, 0))
        self.region_h = ttk.Entry(region_frame, width=6)
        self.region_h.grid(row=1, column=3, padx=2, pady=(5, 0))
        self.region_h.insert(0, "1080")
        
        ttk.Button(parent, text="Set Region", command=self.set_region).grid(row=13, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 2))
        ttk.Button(parent, text="Clear Region", command=self.clear_region).grid(row=14, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        # Stop button
        self.stop_btn = ttk.Button(parent, text="⏹ Stop Execution", 
                                   command=self.stop_execution, state="disabled")
        self.stop_btn.grid(row=15, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
    
    def setup_log_tab(self, parent):
        """Setup the log tab"""
        self.log_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, 
                                                  width=80, height=25, font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Clear log button
        ttk.Button(parent, text="Clear Log", command=self.clear_log).grid(row=1, column=0, pady=(5, 0))
    
    def setup_preview_tab(self, parent):
        """Setup the preview tab"""
        # Control buttons
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.preview_btn = ttk.Button(control_frame, text="▶ Start Live Preview", 
                                      command=self.toggle_preview, state="disabled")
        self.preview_btn.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Label(control_frame, text="Update Interval (sec):").grid(row=0, column=1, padx=(0, 5))
        self.preview_interval = ttk.Entry(control_frame, width=5)
        self.preview_interval.grid(row=0, column=2, padx=(0, 10))
        self.preview_interval.insert(0, "2")
        
        # Preview canvas
        self.preview_canvas = tk.Canvas(parent, bg="gray", width=800, height=600)
        self.preview_canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        # Status label
        self.preview_status = ttk.Label(parent, text="Preview not active")
        self.preview_status.grid(row=2, column=0, pady=(5, 0))
    
    def init_bot(self):
        """Initialize bot in background thread"""
        def init_in_thread():
            try:
                self.bot = ScreenBot()
                self.log("✓ Bot initialized successfully!")
                self.log(f"✓ GPU: {'Enabled' if self.bot.use_gpu else 'Disabled (using CPU)'}")
                self.update_status("Ready", "green")
                self.bot_initializing = False
                
                # Enable buttons
                self.root.after(0, self.enable_buttons)
            except Exception as e:
                self.log(f"✗ Error initializing bot: {e}")
                self.update_status("Initialization failed", "red")
                self.bot_initializing = False
        
        thread = threading.Thread(target=init_in_thread, daemon=True)
        thread.start()
    
    def enable_buttons(self):
        """Enable all control buttons"""
        self.vertical_btn.config(state="normal")
        self.runfile_btn.config(state="normal")
        self.viz_text_btn.config(state="normal")
        self.viz_all_btn.config(state="normal")
        self.preview_btn.config(state="normal")
    
    def log(self, message):
        """Thread-safe logging"""
        self.log_queue.put(message)
    
    def process_log_queue(self):
        """Process log messages from queue"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)
    
    def clear_log(self):
        """Clear the log text area"""
        self.log_text.delete(1.0, tk.END)
    
    def update_status(self, message, color="black"):
        """Update status label"""
        self.status_label.config(text=message, foreground=color)
    
    def set_region(self):
        """Set screen region from input fields"""
        if not self.bot:
            messagebox.showerror("Error", "Bot not initialized yet!")
            return
        
        try:
            x = int(self.region_x.get())
            y = int(self.region_y.get())
            w = int(self.region_w.get())
            h = int(self.region_h.get())
            
            self.bot.set_screen_region(x, y, w, h)
            self.log(f"✓ Screen region set: x={x}, y={y}, width={w}, height={h}")
            self.update_status(f"Region: {w}x{h} @ ({x}, {y})", "blue")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers for region values!")
    
    def clear_region(self):
        """Clear screen region"""
        if not self.bot:
            return
        self.bot.clear_screen_region()
        self.log("✓ Screen region cleared")
        self.update_status("Ready", "green")
    
    def visualize_text(self):
        """Visualize text detections"""
        if not self.bot:
            messagebox.showerror("Error", "Bot not initialized yet!")
            return
        
        def visualize():
            try:
                self.log("Scanning screen for text...")
                from interactive_bot import visualize_text_detections
                visualize_text_detections(self.bot)
                self.log("✓ Visualization complete! Check saved image file.")
            except Exception as e:
                self.log(f"✗ Error: {e}")
        
        threading.Thread(target=visualize, daemon=True).start()
    
    def visualize_all(self):
        """Visualize all detections"""
        if not self.bot:
            messagebox.showerror("Error", "Bot not initialized yet!")
            return
        
        def visualize():
            try:
                self.log("Scanning screen for text and objects...")
                from interactive_bot import visualize_all_detections
                visualize_all_detections(self.bot)
                self.log("✓ Visualization complete! Check saved image file.")
            except Exception as e:
                self.log(f"✗ Error: {e}")
        
        threading.Thread(target=visualize, daemon=True).start()
    
    def run_vertical(self):
        """Run vertical sequence"""
        if not self.bot or self.bot_initializing:
            return
        
        try:
            repeat_count = int(self.repeat_var.get())
        except ValueError:
            repeat_count = 1
        
        self.stop_flag.clear()
        self.update_status("Running vertical sequence...", "blue")
        self.stop_btn.config(state="normal")
        self.disable_buttons()
        
        def run():
            try:
                from interactive_bot import execute_command_strings
                
                # Capture print output
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
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
                
                self.log(f"Starting vertical sequence (x{repeat_count})...")
                for i in range(repeat_count):
                    if self.stop_flag.is_set():
                        self.log("✗ Execution stopped by user")
                        break
                    
                    if repeat_count > 1:
                        self.log(f"\n--- Iteration {i+1} of {repeat_count} ---")
                    
                    execute_command_strings(self.bot, vertical_commands)
                    
                    # Capture any output
                    output = sys.stdout.getvalue()
                    if output:
                        self.log(output.strip())
                        sys.stdout = StringIO()
                    
                    if i < repeat_count - 1 and not self.stop_flag.is_set():
                        time.sleep(1)
                
                # Restore stdout
                sys.stdout = old_stdout
                self.log("✓ Vertical sequence complete!")
            except Exception as e:
                self.log(f"✗ Error: {e}")
            finally:
                self.root.after(0, lambda: self.enable_buttons())
                self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
                self.root.after(0, lambda: self.update_status("Ready", "green"))
        
        self.execution_thread = threading.Thread(target=run, daemon=True)
        self.execution_thread.start()
    
    def run_custom_file(self):
        """Run custom preset file"""
        if not self.bot or self.bot_initializing:
            return
        
        # Open file dialog if no file selected
        if not self.selected_file:
            file_path = filedialog.askopenfilename(
                title="Select Command File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if not file_path:
                return
            self.selected_file = file_path
            self.current_file_label.config(text=os.path.basename(file_path), foreground="black")
        
        try:
            repeat_count = int(self.repeat_var.get())
        except ValueError:
            repeat_count = 1
        
        self.stop_flag.clear()
        self.update_status(f"Running {os.path.basename(self.selected_file)}...", "blue")
        self.stop_btn.config(state="normal")
        self.disable_buttons()
        
        def run():
            try:
                from interactive_bot import execute_command_file
                
                # Capture print output
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                self.log(f"Starting custom file execution (x{repeat_count})...")
                for i in range(repeat_count):
                    if self.stop_flag.is_set():
                        self.log("✗ Execution stopped by user")
                        break
                    
                    if repeat_count > 1:
                        self.log(f"\n--- Iteration {i+1} of {repeat_count} ---")
                    
                    execute_command_file(self.bot, self.selected_file)
                    
                    # Capture any output
                    output = sys.stdout.getvalue()
                    if output:
                        self.log(output.strip())
                        sys.stdout = StringIO()
                    
                    if i < repeat_count - 1 and not self.stop_flag.is_set():
                        time.sleep(1)
                
                # Restore stdout
                sys.stdout = old_stdout
                self.log("✓ Custom file execution complete!")
            except Exception as e:
                self.log(f"✗ Error: {e}")
            finally:
                self.root.after(0, lambda: self.enable_buttons())
                self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
                self.root.after(0, lambda: self.update_status("Ready", "green"))
        
        self.execution_thread = threading.Thread(target=run, daemon=True)
        self.execution_thread.start()
    
    def disable_buttons(self):
        """Disable control buttons during execution"""
        self.vertical_btn.config(state="disabled")
        self.runfile_btn.config(state="disabled")
        self.preview_btn.config(state="disabled")
    
    def stop_execution(self):
        """Stop current execution"""
        self.stop_flag.set()
        self.log("⚠ Stop requested...")
        self.update_status("Stopping...", "orange")
    
    def toggle_preview(self):
        """Toggle live preview on/off"""
        if self.preview_active:
            self.stop_preview()
        else:
            self.start_preview()
    
    def start_preview(self):
        """Start live preview"""
        if not self.bot:
            messagebox.showerror("Error", "Bot not initialized yet!")
            return
        
        try:
            interval = float(self.preview_interval.get())
            if interval < 0.5:
                interval = 0.5
        except ValueError:
            interval = 2.0
        
        self.preview_active = True
        self.preview_btn.config(text="⏹ Stop Live Preview")
        self.preview_status.config(text="Preview active")
        
        def preview_loop():
            while self.preview_active and self.bot:
                try:
                    # Take screenshot
                    screen_img = self.bot.take_screenshot()
                    
                    # Get OCR results
                    all_text = self.bot.get_all_ocr_text(screen_img)
                    
                    # Create visualization
                    vis_img = screen_img.copy()
                    
                    # Draw bounding boxes and labels
                    for center_x, center_y, bbox, text, confidence in all_text:
                        # Adjust bbox to region coordinates
                        if self.bot.screen_region:
                            region_x, region_y, _, _ = self.bot.screen_region
                            adjusted_bbox = [(p[0] - region_x, p[1] - region_y) for p in bbox]
                        else:
                            adjusted_bbox = bbox
                        
                        # Draw bounding box polygon
                        bbox_points = np.array(adjusted_bbox, dtype=np.int32)
                        cv2.polylines(vis_img, [bbox_points], True, (0, 255, 0), 2)
                        
                        # Draw center point
                        cv2.circle(vis_img, (center_x if not self.bot.screen_region else center_x - self.bot.screen_region[0], 
                                            center_y if not self.bot.screen_region else center_y - self.bot.screen_region[1]), 
                                 3, (0, 0, 255), -1)
                        
                        # Draw text label
                        label = f"{text} ({confidence:.2f})"
                        if adjusted_bbox:
                            label_pos = (int(adjusted_bbox[0][0]), int(adjusted_bbox[0][1]) - 5)
                            cv2.putText(vis_img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Convert to PIL Image
                    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(vis_img_rgb)
                    
                    # Resize to fit canvas
                    canvas_width = self.preview_canvas.winfo_width()
                    canvas_height = self.preview_canvas.winfo_height()
                    if canvas_width > 1 and canvas_height > 1:
                        pil_img.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
                    
                    # Convert to PhotoImage
                    self.preview_image = ImageTk.PhotoImage(pil_img)
                    
                    # Update canvas
                    self.preview_canvas.delete("all")
                    self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_image)
                    
                    # Update status
                    self.root.after(0, lambda: self.preview_status.config(text=f"Preview active - {len(all_text)} texts detected"))
                    
                except Exception as e:
                    self.log(f"Preview error: {e}")
                
                # Wait for next update
                time.sleep(interval)
            
            # Reset when stopped
            self.root.after(0, lambda: self.preview_btn.config(text="▶ Start Live Preview"))
            self.root.after(0, lambda: self.preview_status.config(text="Preview stopped"))
        
        self.preview_thread = threading.Thread(target=preview_loop, daemon=True)
        self.preview_thread.start()
    
    def stop_preview(self):
        """Stop live preview"""
        self.preview_active = False
        if self.preview_thread:
            self.preview_thread.join(timeout=1)
    
    def on_closing(self):
        """Handle window close event"""
        self.stop_preview()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
