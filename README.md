# HarshLight
Real-time global illumination based on voxel cone tracing.

HarshLight\src folder includes all the Cpp source code as well as GLSL shaders.
HarshLight\include includes all the headers of third-party libraries.
HarshLight\lib includes all the static libs of third-party libraries.
HarshLight\bin includes all the dlls of third-party libraries.


For now in order to let the program find shaders, subdirectory src\shaders\ needs to be copied and put in the same directory as the executable (please maintain the "src\shaders\" hierarchy). The same is for subdirectory scenes\font\ for UI text rendering.

We have also included a pre-compiled release in Release\ folder. To run the application, the machine must support OpenGL 4.5 (and CUDA).

We have also captured a video: https://www.youtube.com/watch?v=J2vyKYUiGJ4 in case the program cannot run due to mystery compatibility reasons.

command line arguments:
-i <scene file name>
-g <debug mode on/off ("1"/"0")>
-m <mouse sensitivity>
-r <resolution ("720p"/"1080p"/"1440p")>
-d <voxelization dimension ("128"/"256"/"512")>

control:
W/S/A/D/Q/E moveing free camera
Mouse: rotating free camera
I/J/K/L rotating main directional light
F toggling secondary point light
Z toggling UI Text and FPS counter
