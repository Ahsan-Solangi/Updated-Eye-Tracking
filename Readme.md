Step - 0:
Creating a base for the model by experimenting with mediapipe

1st Problem:

MediaPipe loads its own DLL (libmediapipe.dll) using Python's ctypes
It then tries to find a C function called free() inside that DLL
On Linux/macOS, free() is available everywhere, it's part of the standard C library bundled into shared libraries
On Windows, free() is not inside libmediapipe.dll. It lives in a separate system DLL called ucrtbase.dll (the Universal C Runtime)
So MediaPipe crashes with function 'free' not found
The Fix:

ACheck if you're on Windows (os.name == 'nt')
If yes, load the Windows C runtime (ucrtbase.dll) separately
Grab the free function from there and attach it to the MediaPipe library object
Now when MediaPipe looks for free(), it finds the one we attached
In short: MediaPipe looks for free() in the wrong place on Windows. The fix just points it to the right place.

This is a bug in MediaPipe. They didn't account for how Windows handles C runtime functions differently from Linux/macOS.

2nd Problem
Created 4 3 files and modified 4th one (Calibration.py, gaze_estimation.py, visualization.py and mediapipe_new_model.py but the calibration accuracy is very poor. Grouping these in a folder named "P1" and trying out other scripts for better accuracy)
