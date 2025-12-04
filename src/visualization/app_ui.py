import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# Import các module chức năng (Giữ nguyên)
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
        self.root.title("AI People Counter System")
        self.root.geometry("1000x750")
        
        # --- Cấu hình Style (Giao diện đẹp) ---
        self.setup_styles()
        
        # --- Khởi tạo biến (Giữ nguyên logic) ---
        self.video_path = VIDEO_PATH
        self.cap = cv2.VideoCapture(self.video_path)
        self.subtractor = create_subtractor()
        self.tracker = CentroidTracker()
        self.counted_ids = set()
        self.total = 0
        self.old_objects = {}
        self.is_running = False
        self.skip_frames = 0
        self.frame_count = 0
        
        # --- Xây dựng giao diện ---
        self.create_widgets()

    def setup_styles(self):
        """Thiết lập màu sắc, font chữ và theme"""
        style = ttk.Style()
        style.theme_use('clam')  # Sử dụng theme 'clam' để dễ tùy chỉnh màu

        # Định nghĩa màu sắc
        self.colors = {
            'bg_main': '#2b2b2b',       # Màu nền chính (Xám đậm)
            'bg_panel': '#363636',      # Màu nền panel điều khiển
            'accent': '#00ADB5',        # Màu điểm nhấn (Cyan)
            'text': '#EEEEEE',          # Màu chữ (Trắng)
            'success': '#28a745',       # Màu nút Start
            'danger': '#dc3545',        # Màu nút Stop
            'warning': '#ffc107',       # Màu nút Reset
            'secondary': '#6c757d'      # Màu nút Browse
        }

        self.root.configure(bg=self.colors['bg_main'])

        # Style cho Frame
        style.configure('Main.TFrame', background=self.colors['bg_main'])
        style.configure('Panel.TFrame', background=self.colors['bg_panel'])

        # Style cho Label
        style.configure('TLabel', background=self.colors['bg_panel'], foreground=self.colors['text'], font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=self.colors['bg_panel'], foreground=self.colors['accent'], font=('Segoe UI', 12, 'bold'))
        style.configure('Count.TLabel', background=self.colors['bg_panel'], foreground='#00FF00', font=('Segoe UI', 18, 'bold'))

        # Style cho Button
        common_btn_style = {'font': ('Segoe UI', 10, 'bold'), 'borderwidth': 0, 'focuscolor': 'none'}
        
        # Style cho Button Action/Browse/Camera
        style.configure('Action.TButton', background=self.colors['accent'], foreground='white', **common_btn_style)
        style.map('Action.TButton', background=[('active', '#007f85')])

        style.configure('Action.TButton', background=self.colors['accent'], foreground='white', **common_btn_style)
        style.map('Action.TButton', background=[('active', '#007f85')])

        style.configure('Start.TButton', background=self.colors['success'], foreground='white', **common_btn_style)
        style.map('Start.TButton', background=[('active', '#1e7e34')])

        style.configure('Stop.TButton', background=self.colors['danger'], foreground='white', **common_btn_style)
        style.map('Stop.TButton', background=[('active', '#bd2130')])
        
        style.configure('Reset.TButton', background=self.colors['warning'], foreground='black', **common_btn_style)
        style.map('Reset.TButton', background=[('active', '#d39e00')])

    def create_widgets(self):
        """Tạo các thành phần giao diện"""
        
        # 1. Main Container
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 2. Video Area (Phần hiển thị video)
        video_frame = tk.Frame(main_frame, bg="black", bd=2, relief="sunken")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(video_frame, bg="black", text="No Video Loaded", fg="gray", font=("Arial", 14))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # 3. Control Panel (Phần điều khiển bên dưới)
        control_panel = ttk.Frame(main_frame, style='Panel.TFrame', padding=10)
        control_panel.pack(fill=tk.X, ipady=5)

        # Chia Control Panel thành 3 cột: Left (File), Center (Controls), Right (Stats)
        control_panel.columnconfigure(0, weight=1)
        control_panel.columnconfigure(1, weight=2)
        control_panel.columnconfigure(2, weight=1)

        # --- Cột Trái: Chọn File ---
        left_box = ttk.Frame(control_panel, style='Panel.TFrame')
        left_box.grid(row=0, column=0, sticky="w")
        
        ttk.Label(left_box, text="SOURCE VIDEO:", style='Header.TLabel').pack(anchor="w")
        
        file_frame = ttk.Frame(left_box, style='Panel.TFrame')
        file_frame.pack(fill=tk.X, pady=5)
        
        self.browse_btn = ttk.Button(file_frame, text="📂 Browse", command=self.browse_video, style='Action.TButton', width=10)
        self.browse_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.video_label_text = ttk.Label(file_frame, text="Default Video", font=("Segoe UI", 9, "italic"))
        self.video_label_text.pack(side=tk.LEFT)

        # --- Cột Giữa: Điều khiển Playback & Tốc độ ---
        center_box = ttk.Frame(control_panel, style='Panel.TFrame')
        center_box.grid(row=0, column=1)

        # Hàng nút bấm
        btn_frame = ttk.Frame(center_box, style='Panel.TFrame')
        btn_frame.pack(pady=5)

        self.cam_btn = ttk.Button(btn_frame, text="📷 CAM", command=self.start_webcam, style='Action.TButton', width=10)
        self.cam_btn.pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(btn_frame, text="▶ START", command=self.start_video, style='Start.TButton', width=10)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="⏹ STOP", command=self.stop_video, style='Stop.TButton', width=10)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(btn_frame, text="⟳ RESET", command=self.reset_counter, style='Reset.TButton', width=10)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # Hàng chỉnh tốc độ
        speed_frame = ttk.Frame(center_box, style='Panel.TFrame')
        speed_frame.pack(pady=(5,0))
        
        ttk.Label(speed_frame, text="Speed: ").pack(side=tk.LEFT)
        
        self.speed_var = tk.IntVar(value=0)
        self.speed_scale = ttk.Scale(speed_frame, from_=0, to=5, orient=tk.HORIZONTAL, 
                                     variable=self.speed_var, length=150, command=self.update_speed)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.speed_label = ttk.Label(speed_frame, text="Normal (1.0x)", width=12)
        self.speed_label.pack(side=tk.LEFT)

        # --- Cột Phải: Thống kê ---
        right_box = ttk.Frame(control_panel, style='Panel.TFrame')
        right_box.grid(row=0, column=2, sticky="e")
        
        ttk.Label(right_box, text="TOTAL COUNT", style='Header.TLabel').pack(anchor="e")
        self.count_label = ttk.Label(right_box, text="0", style='Count.TLabel')
        self.count_label.pack(anchor="e", pady=5)

    # --- Các hàm logic giữ nguyên như cũ ---

    def start_webcam(self):
        """Khởi động luồng video từ webcam (chỉ số 0)."""
        
        # 1. Dừng luồng cũ nếu đang chạy
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        # 2. Khởi tạo webcam (chỉ số 0) và thiết lập độ phân giải mặc định 640x480
        self.video_path = "Webcam Live"
        self.cap = cv2.VideoCapture(0)
        
        # Thiết lập độ phân giải để đồng bộ với logic xử lý (640x480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.video_label_text.config(text="Camera Failed! Check connection.", foreground=self.colors['danger'])
            print("Error: Could not open webcam.")
            return

        # 3. Reset trạng thái đếm và CV
        self.subtractor = create_subtractor()
        self.tracker = CentroidTracker()
        self.counted_ids = set()
        self.total = 0
        self.old_objects = {}
        self.frame_count = 0

        # 4. Cập nhật UI và bắt đầu luồng
        self.video_label_text.config(text="Webcam Live", foreground=self.colors['accent'])
        self.count_label.config(text="0")
        
        # Hiển thị frame đầu tiên (placeholder)
        ret, frame = self.cap.read()
        if ret:
            self.show_image(frame)
        
        self.start_video() # Tự động bắt đầu đếm

    def update_speed(self, value):
        speed_levels = {0: "Normal", 1: "1.5x", 2: "2.0x", 3: "2.5x", 4: "3.0x", 5: "4.0x"}
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
            
            # Cập nhật UI hiển thị tên file ngắn gọn
            filename = file_path.split('/')[-1]
            if len(filename) > 25: filename = filename[:22] + "..."
            self.video_label_text.config(text=filename)
            self.count_label.config(text="0")
            
            # Hiển thị frame đầu tiên (thumbnail)
            ret, frame = self.cap.read()
            if ret:
                self.show_image(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset về đầu

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
        
        # UI Overlay trên video (tùy chọn, vì đã có label bên ngoài)
        # cv2.putText(frame, f"Count: {self.total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        
        self.show_image(frame)
        self.count_label.config(text=f"{self.total}")
        
        self.root.after(1, self.update_frame)

    def show_image(self, frame):
        """Hàm phụ trợ để hiển thị ảnh lên Label"""
        # Resize frame để vừa khít với khung hình hiện tại của UI nếu cần
        # Ở đây giữ cố định 640x480 để logic vẽ không bị lệch, 
        # nhưng khi hiển thị lên GUI có thể dùng thuật toán fill
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        
        # Resize ảnh cho vừa với khung hiển thị (Responsive nhẹ)
        display_w = self.video_label.winfo_width()
        display_h = self.video_label.winfo_height()
        
        if display_w > 10 and display_h > 10:
             # Giữ tỷ lệ khung hình
             img.thumbnail((display_w, display_h), Image.Resampling.LANCZOS)
             
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk, text="") # Xóa text placeholder

    def start_video(self):
        if not self.is_running:
            self.is_running = True
            self.update_frame()

    def stop_video(self):
        self.is_running = False

    def reset_counter(self):
        self.total = 0
        self.counted_ids = set()
        self.frame_count = 0
        self.old_objects = {}
        self.count_label.config(text="0")
        
        # Reset tracker & subtractor để tránh lỗi logic khi đếm lại
        self.tracker = CentroidTracker()
        self.subtractor = create_subtractor()

def run_ui():
    root = tk.Tk()
    ui = PeopleCounterUI(root)
    root.mainloop()