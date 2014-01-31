NeuralNetwork
=============

A Neural Network library in C# I developed a few years ago. 

Feel free to make use of it and use it in your own solutions as long as you give credit :)

The project actually formed part of a larger solution to perform road detection in satellite images but it is completely decoupled as a standalone library and can therefore be used for any general purpose classification task.

The project contains files for:
* Vectors / Labeled Vectors with any dimension N specified by the user
* Matrix classes
* NeuralNetwork class with the ability to specify which activation function to use (or specify your own custom one). It also has support for firing events after each epoch during training in order to hook into some form of GUI.

Notes
-----
The Matrix class makes use of very naive methods but should serve the purpose of Neural Networks that dont use excessively large sizes. Feel free to send a pull request for any improvements in this area (a simple one would be to make use of the C# Parallel extensions for example). Using XNA's Matrix classes could also be an alternative solution since these are probably already heavily optimised.

The project should be capable or running in both .NET and Mono (I have not tested the latter but there should be no reason why it wouldn't work)

