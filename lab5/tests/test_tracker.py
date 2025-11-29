
import unittest
import numpy as np
import cv2
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.object_tracker import ObjectTracker, CentroidTracker

class TestObjectTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectTracker(min_area=10, max_disappeared=5, max_distance=50)
        # Warm up with empty frames
        empty_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        for _ in range(50):
            self.tracker.bg_subtractor.apply(empty_frame)

    def create_frame_with_rect(self, x, y, w, h, width=200, height=200):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), -1)
        return frame

    def test_centroid_tracker_registration(self):
        ct = CentroidTracker()
        rects = [(10, 10, 20, 20)]
        objects = ct.update(rects)
        
        self.assertEqual(len(objects), 1)
        self.assertIn(0, objects)
        self.assertEqual(objects[0], (10, 10, 20, 20))

    def test_tracking_movement(self):
        # Frame 1: Object at (10, 10)
        frame1 = self.create_frame_with_rect(10, 10, 20, 20)
        objects1, _ = self.tracker.process_frame(frame1)
        
        # Depending on learning rate, the first frame might not show the object perfectly if it was just background
        # But since we warmed up with empty frames, this new white rect should be FG.
        
        # Note: MOG2 might need a few frames to stabilize the object mask if it's complex, 
        # but for a simple rect it should be fine.
        
        # Let's check if we detected it.
        if len(objects1) == 0:
             # Try one more frame
             objects1, _ = self.tracker.process_frame(frame1)
        
        self.assertEqual(len(objects1), 1, "Failed to detect object in frame 1")
        id1 = list(objects1.keys())[0]
        
        # Frame 2: Object moved to (15, 15)
        frame2 = self.create_frame_with_rect(15, 15, 20, 20)
        objects2, _ = self.tracker.process_frame(frame2)
        
        self.assertEqual(len(objects2), 1, "Failed to detect object in frame 2")
        id2 = list(objects2.keys())[0]
        
        # ID should be preserved
        self.assertEqual(id1, id2, "Object ID changed (tracking failed)")

    def test_tracking_disappearance(self):
        # Frame 1: Object present
        frame1 = self.create_frame_with_rect(10, 10, 20, 20)
        objects1, _ = self.tracker.process_frame(frame1)
        if len(objects1) == 0:
             objects1, _ = self.tracker.process_frame(frame1)
             
        self.assertEqual(len(objects1), 1)
        
        # Frame 2: Object gone (back to empty)
        frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
        objects2, _ = self.tracker.process_frame(frame2)
        
        # The object is gone from screen, but tracker keeps it in 'disappeared' state.
        # It should still be in the returned objects list?
        # Let's check CentroidTracker.update implementation.
        # It returns self.bboxes.
        # If object is disappeared, it is NOT removed from self.bboxes immediately.
        # So yes, it should still be there.
        
        self.assertEqual(len(objects2), 1, "Object should still be tracked (in disappeared state)")
        
        # Frame 3-10: Object still gone
        for _ in range(10):
            self.tracker.process_frame(frame2)
            
        # After max_disappeared (5), it should be removed
        objects_final, _ = self.tracker.process_frame(frame2)
        self.assertEqual(len(objects_final), 0, "Object should be deregistered after max_disappeared")

    def test_multiple_objects(self):
        # Frame 1: Two objects
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (30, 30), (255, 255, 255), -1)
        cv2.rectangle(frame, (100, 100), (120, 120), (255, 255, 255), -1)
        
        objects, _ = self.tracker.process_frame(frame)
        if len(objects) < 2:
             objects, _ = self.tracker.process_frame(frame)
             
        self.assertEqual(len(objects), 2)

if __name__ == '__main__':
    unittest.main()
