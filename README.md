# MNIST-Visualizer
A python script that allows you to see what digits your model guessed wrong by creating a video.  
The frames/plots that contain a wrong predicted digit will be 2s long the ones that do not are 1/15 * s.  
I have also provided my model and the script I used to train it.
## Prerequisites
tensorflow, numpy, opencv, matplotlib
## How to use
1. Save your trained model for mnist digit recognition
2. use the mnist-visualizer.py script

path_to_model: path to the model.h5 file  
save_temp_figs: **0** to delete the temporary jpgs made by the script; **1** to keep them

```
python mnist-visualizer.py model.h5 1
```

![](mnist_model.gif)
