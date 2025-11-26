import cv2
from src.preprocessing.bg_subtractor import create_subtractor
from src.preprocessing.thresholding import apply_threshold
from src.preprocessing.morphology import clean_mask
from src.detection.contour_detector import detect_people
from src.tracking.centroid_tracker import CentroidTracker
from src.counting.people_counter import count_people
from src.utils.drawer import draw_line, draw_box
from src.utils.config import VIDEO_PATH
from src.visualization.app_ui import run_ui

# Kích thước cố định video (nếu muốn)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    subtractor = create_subtractor()
    tracker = CentroidTracker()

    counted_ids = set()
    total = 0
    old_objects = {}

    # Lấy LINE_Y sau khi đọc frame đầu tiên
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc video")
        return

    # Resize frame nếu muốn
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    LINE_Y = FRAME_HEIGHT // 2

    # reset video về đầu để vòng lặp chính đọc từ đầu
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        fg_mask = subtractor.apply(frame)
        th = apply_threshold(fg_mask)
        clean = clean_mask(th)

        boxes = detect_people(clean)
        objects = tracker.update(boxes)

        # đếm
        total += count_people(objects, old_objects, LINE_Y, counted_ids)
        old_objects = objects.copy()

        # vẽ
        draw_line(frame, LINE_Y)

        for obj_id, (cx, cy) in objects.items():
            for box in boxes:
                bx, by, bw, bh = box
                if bx < cx < bx+bw and by < cy < by+bh:
                    draw_box(frame, box, obj_id)

        cv2.putText(frame, f"Count: {total}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        cv2.imshow("People Counter", frame)
        # cv2.imshow("Mask", clean)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # if len(sys.argv) > 1 and sys.argv[1] == "--ui":
        run_ui()
    