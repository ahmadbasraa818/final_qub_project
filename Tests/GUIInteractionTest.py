from unittest.mock import patch
import unittest
from main import DragAndDropGUI  

class TestDragAndDropGUI(unittest.TestCase):
    @patch('easygui.fileopenbox', return_value=['/home/thatchaoskid/Documents/final_qub_project/ImagesToTest/test6.jpeg', '/home/thatchaoskid/Documents/final_qub_project/ImagesToTest/test7.png'])
    def test_drag_and_drop(self, mock_fileopenbox):
        gui = DragAndDropGUI()
        gui.drag_and_drop()
        self.assertEqual(gui.image_paths, ['/home/thatchaoskid/Documents/final_qub_project/ImagesToTest/test6.jpeg', '/home/thatchaoskid/Documents/final_qub_project/ImagesToTest/test7.png'])
