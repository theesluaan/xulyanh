counted_ids = set()

def count_people(objects, old_objects, line_y, counted_ids):
    count = 0
    for obj_id, (cx, cy) in objects.items():
        if obj_id in old_objects:
            old_y = old_objects[obj_id][1]
            if old_y < line_y <= cy and obj_id not in counted_ids:
                count += 1
                counted_ids.add(obj_id)
    return count
