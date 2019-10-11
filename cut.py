import cv2

cut_seconds = [(0, 5),  # 1_1
               (40, 45),  # 1_2
               (110, 125),  # 2_1
               (210, 225),  # 3_1
               (330, 335),  # 4_1
               (345, 350),  # 4_2
               (430, 440),  # 5_1
               (450, 455),  # 5_2
               (550, 555),  # 6_1
               (585, 590),  # 6_2
               (675, 685),  # 7_1
               (695, 705),  # 7_2
               (765, 770),  # 8_1
               (800, 805)]  # 8_2

stream = cv2.VideoCapture('test.mp4')
assert stream.isOpened(), 'ERROR: Cannot open video'
fps = int(stream.get(cv2.CAP_PROP_FPS))
num_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
output_size = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
total_time = num_frames / fps
assert abs(total_time - 830) < 20, 'ERROR: Video time should be adjusted'

writer = cv2.VideoWriter('test_out.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, output_size)
assert writer.isOpened(), 'ERROR: Cannot write video'

cut_idx = 0
lower_bound = cut_seconds[cut_idx][0] * fps
upper_bound = cut_seconds[cut_idx][1] * fps
print('Cutting', cut_idx, 'Started')
for i in range(num_frames):
    success, frame = stream.read()
    if i >= lower_bound:
        writer.write(frame)
    if i == upper_bound:
        cut_idx += 1
        if cut_idx == 14:
            break
        lower_bound = cut_seconds[cut_idx][0] * fps
        upper_bound = cut_seconds[cut_idx][1] * fps
        print('Cutting', cut_idx, 'Started')

stream.release()
