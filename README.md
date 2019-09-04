# PyPhotoBooth
PhotoBooth using Python &amp; OpenCV

Hello and welcome to the Extremely Fun Photo Booth. For theater applications.
(scroll to the bottom for modules/installation)

Your basic modes are three.

Photo Booth Mode:
click on there main screen, and you can take four nice pictures of yourself.
When you get done taking photos, you'll get to review what you made.
Sorry, no do-overs!
You can then type in your email address, and we'll send you your photo set.

Show Mode:
Secretly, all of the photos taken have been catalogued to disk. And the computer has identified all the faces in these photos, and has them stored away too.
When you go into show mode, you'll see a bunch of faces tessellated onto the screen. 

You can make some changes to the image you see:
1. add a gaussian blur of 2 intensities
2. color-normalize the image to 2 intensities
3. transform it into an edge-detected black and white image
4. using up or down arrows, change the number of faces on the screen. This is limited by how many photos are in the memory hole.

This is a tool for performance, meant to capture audience's faces and then most likely project them onto a surface.
If your projector is warped on that surface due to an odd angle, you can correct it with...

Warp Mode:
You'll see a test image. You can use keystrokes or the mouse to drag around the corners, perspective warping the test card until it fits your goal surface. 
Tab works the same as the 'Sel' button, and arrows/arrowkeys are the same as well.
Any settings you make in warp mode will be reflected in show mode.

SHORTCUTS:
There are a few useless filters I found along the way that I like because they look interesting. They are for art only, not for use. They are implemented in the Photo Booth mode, and can be activated before you press 'record'. If you record with them on, they will end up in the system, so be careful.

pressing 'r' removes a frame that was taken on startup from every coming frame. It's like cheap background reduction.
press 'z' to take a fresh background image. With the intense noisiness of a webcam, you get a lot of color noise foregrounded with this one.
pressing 'm' gives you a real background removal algorithm. It looks kind of cool, but needs serious lighting to do anything useful.
pressing 'w' gives you a Weird setting that multiplies the reciprocal of the frame with the original frame. Up./Down arrow keys adjust threshold. I like this one.
press 'n' to return to normal.

HOW TO RUN:
Open TP_Main (with the orange flag). Run. That's it.

LIBRARIES
You will need opencv and numpy. Installations of opencv automatically include numpy. I installed opencv using homebrew: terminal command:
$Brew install opencv 
Binding opencv to the right python library can be a little tricky; things will want to revert to Python 2. I followed instructions in a video to get mine right... Here's the video:
https://www.youtube.com/watch?v=iluST-V757A&t=550s


