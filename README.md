################ Cat Dog CNN with K-fold Cross Validation #########################################


## Installing the dependencies ##
-First you need you need to have TensorFlow installed, which can be done with bazel.
After it's needed to ser the environment variable TF_PATH to the root of where TensorFlow
was installed (suggestion: add it to bashrc).

-Then you need protobuf installed (the same version used by your TensorFlow version).
(The steps above can be found at the first article https://itnext.io/how-to-use-your-c-muscle-using-tensorflow-2-0-and-xcode-without-using-bazel-builds-9dc82d5e7f80) 

-Install absl from https://github.com/abseil/abseil-cpp/tree/master/absl and add the path where it was
 installed to the environment variable ABSL_INSTALL_PATH (e.g. if it's installed at /Dowloads/absl 
then ABSL_INSTALL_PATH = /Downloads) suggestion: add it to bashrc.

-Run protobuf inside the absl folder.

## Compiling ##
From root of this project run:

mkdir build
cd build
cmake .. && make

## Execution ##
The binaries can be found at the bin folder.

CatDog_kfold_cv : Our implementation with k fold cross validation and openmp parallelization

CatDog_simple : Original V2 implementation, classifies the images in a simpler way (link to the data 
for this implementation can be found on the original articles).



############################ BASED ON ##############################################################

# TFMacCpp
A few projects that demo how to develop on Mac with TF and XCode only

First article: https://itnext.io/how-to-use-your-c-muscle-using-tensorflow-2-0-and-xcode-without-using-bazel-builds-9dc82d5e7f80

Second article: https://itnext.io/creating-a-tensorflow-dnn-in-c-part-1-54ce69bbd586

Third article: https://towardsdatascience.com/creating-a-tensorflow-cnn-in-c-part-2-eea0de9dcada

Fourth article: https://towardsdatascience.com/creating-a-tensorflow-cnn-in-c-part-3-freezing-the-model-and-augmenting-it-59a07c7c4ec6

For DNN Part 1 use V1 files
For DNN Part 2 use V2 files
For DNN Part 3 use regular files (w/o V)
