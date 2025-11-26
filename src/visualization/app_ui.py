import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from src.preprocessing.bg_subtractor import create_subtractor
from src.preprocessing.thresholding import apply_threshold
from src.preprocessing.morphology import clean_mask
from src.detection.contour_detector import detect_people
from src.tracking.centroid_tracker import CentroidTracker
from src.counting.people_counter import count_people
from src.utils.drawer import draw_line, draw_box
from src.utils.config import VIDEO_PATH

class PeopleCounterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("People Counter")
        self.root.geometry("900x650")
        
        self.video_path = VIDEO_PATH
        self.cap = cv2.VideoCapture(self.video_path)
        self.subtractor = create_subtractor()
        self.tracker = CentroidTracker()
        self.counted_ids = set()
        self.total = 0
        self.old_objects = {}
        self.is_running = False
        self.skip_frames = 0  # Skip frames để tăng tốc độ
        self.frame_count = 0
        
        # Video label
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls
        controls_frame = ttk.Frame(root)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.browse_btn = ttk.Button(controls_frame, text="Browse Video", command=self.browse_video)
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        self.video_label_text = ttk.Label(controls_frame, text="No video selected", font=("Arial", 10))
        self.video_label_text.pack(side=tk.LEFT, padx=5)
        
        self.count_label = ttk.Label(controls_frame, text="Count: 0", font=("Arial", 14, "bold"))
        self.count_label.pack(side=tk.LEFT, padx=5)
        
        # Speed Control
        ttk.Label(controls_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.IntVar(value=0)
        self.speed_scale = ttk.Scale(controls_frame, from_=0, to=5, orient=tk.HORIZONTAL, 
                                     variable=self.speed_var, length=100, command=self.update_speed)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.speed_label = ttk.Label(controls_frame, text="Normal")
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(controls_frame, text="Start", command=self.start_video)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(controls_frame, text="Stop", command=self.stop_video)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(controls_frame, text="Reset", command=self.reset_counter)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

    def update_speed(self, value):
        speed_levels = {0: "Normal", 1: "1.5x", 2: "2x", 3: "2.5x", 4: "3x", 5: "4x"}
        skip_levels = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6}
        level = int(float(value))
        self.skip_frames = skip_levels[level]
        self.speed_label.config(text=speed_levels[level])

    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            self.subtractor = create_subtractor()
            self.tracker = CentroidTracker()
            self.counted_ids = set()
            self.total = 0
            self.old_objects = {}
            self.frame_count = 0
            self.video_label_text.config(text=file_path.split('/')[-1])
            self.count_label.config(text="Count: 0")

    def update_frame(self):
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            self.root.after(1, self.update_frame)
            return
        
        # Skip frames để tăng tốc độ
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) != 0:
            self.root.after(1, self.update_frame)
            return
        
        frame = cv2.resize(frame, (640, 480))
        LINE_Y = 240
        
        fg_mask = self.subtractor.apply(frame)
        th = apply_threshold(fg_mask)
        clean = clean_mask(th)
        boxes = detect_people(clean)
        objects = self.tracker.update(boxes)
        
        self.total += count_people(objects, self.old_objects, LINE_Y, self.counted_ids)
        self.old_objects = objects.copy()
        
        draw_line(frame, LINE_Y)
        for obj_id, (cx, cy) in objects.items():
            for box in boxes:
                bx, by, bw, bh = box
                if bx < cx < bx+bw and by < cy < by+bh:
                    draw_box(frame, box, obj_id)
        
        cv2.putText(frame, f"Count: {self.total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        self.count_label.config(text=f"Count: {self.total}")
        
        self.root.after(1, self.update_frame)

    def start_video(self):
        self.is_running = True
        self.update_frame()

    def stop_video(self):
        self.is_running = False

    def reset_counter(self):
        self.total = 0
        self.counted_ids = set()
        self.frame_count = 0
        self.count_label.config(text="Count: 0")

def run_ui():
    root = tk.Tk()
    ui = PeopleCounterUI(root)
    root.mainloop()