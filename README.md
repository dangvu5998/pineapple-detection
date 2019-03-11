# Hướng dẫn sử dụng
1. Tải file frozen_inference_graph.pb [tại đây](https://drive.google.com/open?id=1utVYpDlHjk0efRAGlbY0q_raaS7ViICU) và copy vào `data/checkpoints`
2. Thay đường dẫn đến file ảnh cần test vào biến `IMAGE_NAME` trong file `object_detection_image.py`
3. Chạy lệnh `python object_detection_image.py`