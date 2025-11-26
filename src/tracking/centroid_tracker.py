from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappear=10):
        self.nextID = 1
        self.objects = {}          # {id: (cx, cy)}
        self.disappear = {}        # {id: count}
        self.max_disappear = max_disappear

    def register(self, centroid):
        self.objects[self.nextID] = centroid
        self.disappear[self.nextID] = 0
        self.nextID += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappear[obj_id]

    def update(self, boxes):
        # nếu không có ai trong frame
        if len(boxes) == 0:
            remove = []
            for obj_id in self.disappear:
                self.disappear[obj_id] += 1
                if self.disappear[obj_id] > self.max_disappear:
                    remove.append(obj_id)
            for r in remove:
                self.deregister(r)
            return self.objects

        # lấy centroid từ bounding box
        input_centroids = []
        for x, y, w, h in boxes:
            cx = int(x + w/2)
            cy = int(y + h/2)
            input_centroids.append((cx, cy))

        # nếu chưa có object → đăng ký hết
        if len(self.objects) == 0:
            for ic in input_centroids:
                self.register(ic)
            return self.objects

        # so khớp centroid cũ – mới bằng khoảng cách Euclid
        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        D = dist.cdist(obj_centroids, input_centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_cols = set()

        # cập nhật
        for r, c in zip(rows, cols):
            if c in used_cols:
                continue
            obj_id = obj_ids[r]
            self.objects[obj_id] = input_centroids[c]
            self.disappear[obj_id] = 0
            used_cols.add(c)

        # đăng ký centroid mới
        for i, centroid in enumerate(input_centroids):
            if i not in used_cols:
                self.register(centroid)

        return self.objects
 
    # def update_with_centroids(self, centroids):
    #     # Dùng lại logic cũ nhưng nhận trực tiếp list centroid thay vì boxes
    #     dummy_boxes = [(cx-1, cy-1, 2, 2) for cx, cy in centroids]  # fake box nhỏ
    #     return self.update(dummy_boxes)    