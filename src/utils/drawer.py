import cv2

def draw_line(frame, y):
    cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 255), 2)

def draw_box(frame, box, obj_id):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f"ID {obj_id}", (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
