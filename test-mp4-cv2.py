import cv2

# Open the video file
video_path = 'meatzoo.mp4'  # Replace 'your_video_file.mp4' with the path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
   print("Error: Unable to open video file.")
   exit()

# Loop to read and display frames from the video file
count = 0
while count < 5:
   # Read a frame from the video file
   ret, img = cap.read()
   cv2.imwrite("frame%d.jpg" % count, img)
   count = count + 1
   print(count)
   
# Check if the frame was read successfully
#if not ret:
   #print("Error: Unable to read frame.")
   #break

# Display the frame
# cv2.imshow('Video', frame)

# Check for 'q' key to exit the loop
# if cv2.waitKey(25) & 0xFF == ord('q'):
# break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

