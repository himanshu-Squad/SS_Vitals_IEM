This repository has been modified from the original version of vitallens-python to improve face detection and enhance the accuracy of RGB extraction for heart rate estimation. Key modifications include:

Key Changes: 
1. Face Detection:

Replaced the original SSD face detector with MediaPipe Face Detection. This change improved the speed and accuracy of face detection, providing a more robust input for heart rate estimation, particularly in low-light and low-resolution conditions.

2. Top-5 Face Regions for RGB Extraction:

Introduced a new method to extract RGB values from the Top-5 face regions (instead of the full face), which enhances the accuracy of heart rate estimation by focusing on stable regions of interest (ROIs) with less noise or movement artifacts.

3. Performance Comparison:

Tested and compared the results and performance of these modifications against the original repository. Our modified implementation demonstrated improved accuracy in vital sign estimation (heart rate) while maintaining or improving computational efficiency.