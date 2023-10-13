# **Machine Learning Internship - University of Cambridge**

## **Pre-Internship**

The focus of my internship at the Computer Laboratory was largely to understand the computer science and the theory behind neural networks, rather than simply applying a model to a set of images. So, in preparation for this, I read [this book](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen and watched [this video series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3Blue1Brown. Although I didn't understand every single detail, these resources, along with pre-existing education, gave me a suitable mathematical foundation to build future knowledge on top of.

## **Structure of the Internship**

Before getting into the content, I'd like to clarify the structure of the internship. Essentially, Ian, my mentor, had a mental checklist of topics and concepts that I should be able to understand by the end. For each topic, Ian would do a mini-lecture using a whiteboard in his office. I asked plenty of questions and he would pose questions back. I'd like to emphasise that this back-and-forth was a key feature of the internship and it allowed me to have a much stronger grasp on the content than I would have otherwise. Ian would provide print-outs of resources he'd heavily adjust after finding them online. Also, he had a git repo [available on GitHub](https://github.com/ijl20/nn_intro) which contained clearly commented example code which supported my learning greatly. These provided materials gave me structure in my day-to-day.

Each morning, Ian and I would discuss how I was getting on. This was also an opportunity for me to raise anything I didn't understand, or suggest we progress on to the next topic. Depending on the day, these catch-ups would last anywhere between 45 minutes to 2 hours. This was extremely valuable and the whiteboard in his office was in-use plenty of times. On occasion, computer science topics that were somewhat weakly adjacent to machine learning would crop up but curiosity would lead to Ian and I having a 'white-board session' about something like lambda calculus, or electric circuits (electric circuits being strongly linked to his sensor research).

An otherwise understated part of the internship would be lunchtime. This gave me the opportunity to interact with other researchers at the lab in a more casual setting. I'd listen in on the higher-level conversations which oftentimes regarded their work on sensors and a problem they had to overcome. I'd engage and ask my own questions (without detracting too much) to get a picture of the issue at hand. With that said, it wasn't all technical; sometimes it was more "how are the kids getting on" or a discussion of "why is there a mandatory retirement age at the university" both of which were pleasant in their own right.

## Part 1: The Set Up (Stage 0)

For the first few days, I was working out of Ian, my mentor's, office. Here, I was given a new machine which I installed Linux onto, and then began making myself familiar with the Bash Shell. After a brief networking overview, I set up a static IP address for my machine which allowed me to SSH into the Linux box from my laptop. Next, I created a folder ('src/nn\_intro') that would hold my code. Within that, I set up a Python virtual environment (venv) and installed the libraries I would be working with into it. This included: Jupyter Notebook, Numpy, OpenCV2, and Tensorflow.keras. All the libraries used were appended to a requirements.txt file. Lastly, I turned 'src/nn\_intro' into a git repository ([link to GitHub](https://github.com/jamespilcher/nn_intro)) where I could utilise the benefits of version control. As an aside, Ian's previously mentioned repo was also called 'nn\_intro' which was mildly amusing as it was not an intentional thought-out decision - it just kind of happened. Anyway, this all came together to create a suitable development environment for my internship.

Once I had my Jupyter Notebook up and running, I began toying with numpy and matplotlib. The goal of this period was to get familiar with the functions and attributes that I would commonly use (numpy.shape comes to mind) and also to start to gain an intuition for computer vision - an image really is just a matrix of values. For example, in this period I subtracted two images of Google Maps Traffic from one another to get the difference (trafficchanges.png). For a minute I was scratching my head as to why something so simple wasn't working as I was getting a blank image – until numpy.shape came to the rescue and we realised it was subtracting the alpha channel. It was small learning opportunities like that that really added up. Also see 'deepfryer.ipynb' and it's OpenCV equivalent 'deepfryer\_cv2.py', where I mangled the colours in an image of a dog.

![](RackMultipart20231013-1-eulm2e_html_39d981731a88da4d.png) ![](RackMultipart20231013-1-eulm2e_html_c3206a69e6e28343.png)

_Figure 1: Mangled Dog Figure 2: Traffic Changes_

Throughout the rest of the internship, I would regularly apply the things I learnt in this first week, such as making regular commits, using bash commands, and using specific library functions.

## Part 2: Leaving the Nest (Stage 1)

By the end of the first week, I was placed in another office where I got to know some incredibly smart and friendly researchers who came from all over the world. I was set up at an empty desk with the help of Rohit, one of the researchers. Ian summarises the next over-arching task nicely [here](https://github.com/ijl20/nn_intro#readme):

"_The idea [was] to first develop some NN code that works in pure Python, and then continue to develop from there until we [had] a fairly comprehensive framework in pure Python supporting multiple layer types and a simple class-based declarative definition of the network"_

Coding it all from scratch would provide me with a stronger intuition about what goes on behind the scenes of a typical neural network library such as Tensorflow Keras (which I then later used). My network would classify the MNIST hand-written digits dataset and this naturally led to the exploration of topics such as one-hot encoding and linear classification. It's also worth noting that classifying this dataset is informally known as 'The Hello World of Neural Networks'.

The first sub-task was to get the network to classify each digit as 'a zero' or 'not a zero' using a single dense layer, sigmoid as the activation function, and (binary) cross-entropy as the loss function. The resource we referred to for this was originally written by Jonathon Weisberg. It can be [found here.](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/) It was much more of a loose guide than it was a copy-and-paste job. Most prominently, Weisberg did not use functions for training, loading data, predicting, etc, while my code did.

One epoch of this network looked like:

- Do a forward pass (predict)
- Calculate the loss
- Do a backward pass
- Update the weights (train)

I say 'this network' but really this is how any epoch in any network looks (in a basic sense). This task obviously built upon the my previously mentioned mathematical foundation, and it encouraged me to explore and ask questions about other parts of neural networks. For example, "Why do you need an activation function?", well, they remove linearity from the network making back-propagation possible, they can stop a neuron from 'firing' at all if below a certain value (E.G. for RELU, if the input to the activation function is negative, the activation will remain 0), and they prevent the values from getting ridiculously large over several layers.

Ian also encouraged me to explore the origins of cross entropy which lead me down a short information theory rabbit hole. This sort of exploration was routinely promoted to give me a more holistic understanding than otherwise. While we are in the realm of statistics, an interesting situation I encountered was due to the fact that about 90% of the numbers in MNIST are non-zero. As a result, the network could simply predict every single input as non-zero, and still maintain a 90% accuracy. I am being hyperbolic, but essentially the network could 'lean non-zero' just because of the training dataset. This would be something to consider if this were a proper neural network - it could be good because in real life a zero is less likely to appear than a non-zero, or alternatively, it could be bad because we are arbitrarily giving more weight to one category over another. Originally, our accuracy figure was a measure of correct classifications over the number of images but with our current knowledge, it could be argued that the number we really care about is how many zeros did it predict correctly over how many zeros appeared.

However there were some quirks/limitations in this Weisberg resource. Unreasuringly, Ian assured me that this is the case with any resource relating to neural networks. In this network, if one-hot were to be used, you would expect 2 outputs reflecting 'zero' and 'not zero'. However the network only outputted a single neuron (to make it one hot you would add another output of 1 minus the activation of the original neuron). From my understanding, this 'single-digit one-hot output' is technically binary encoding with 2 categories. Obviously the use of this encoding isn't incorrect in the sense that it achieves the goal of the network, but for a tutorial it was rather specific and a bit more niche than expected. Also, as a result of this binary encoding, this network used 'binary cross entropy' instead of typical 'cross entropy loss' as the loss function, which was a loss function I was less likely to use at this stage of my learning – but it was good to be aware of.

Another limitation was the way it implemented backward-prop. As I knew from my preparation, the goal of a neural network is to minimise the loss with respect to each weight. To clarify, the loss is a measure of how incorrect the prediction (the output of the network) was, so intuitively you want this loss to be as small as possible. You calculate this loss using the output of the network, and the desired output called 'ground-truth' which is provided with the training dataset. In order to calculate the loss with respect to each weight, the chain rule must be used. Weisberg derived this chain rule and then squashed into one mathematical formula which was implemented directly. While this did nicely illustrate the chain rule, it didn't clearly demonstrate how one would backprop through an indefinite number of layers without recalculating the entire chain rule.

To some up, the shape of my first network ever was:

- Input (784x1)
- Dense (784x1)
- Sigmoid()
- Output(1x1)

And its performance was: ![](RackMultipart20231013-1-eulm2e_html_31560e244efd86ee.png)

That last figure is unfortunately off by a factor of 100.

Once the simple network was working, I moved on to the [next resource](https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9), written by Aayush Agrawal which would abstract the layers into classes. This gave me a better understanding of back propagation. This is closer to what goes under the hood of Tensor, a.

- batches
- validation data

- Constructing networks is very similar to keras. And easier and more intuitive!
- Cookery
- overfitting

Began Cookery – what do I mean by that.

overfitting

## Part 3: Convolution (Stage 4)

-briefer

- academic

- no coding

- all theory

- ian did provide his own code worth noting. We did go over it in his office. The for loop.

- step size

- stride

- the formula

The next stage of my learning was

One giant bosh, with a single matrix.

## Part 4: The Adversarial Sneaker (Finale)

At this point the learning portion was over.

- What could I have done better.
- This model was limited
- Pretrained.
- Trained later. At home.

Functional programming

Well, you'll have to call the fire brigade

Im not sure they might be able to use extra ladders because I think they might also want to use themselves an axe to get that ladder in half

Whiteboard pitch

## Conclusion

Para 1. Outline concepts and software worked with

unfortunately didnt capture screenshots

Clarifying questions:

Why did we want a static IP address. Was it necessary for our purposes.

![](RackMultipart20231013-1-eulm2e_html_295e9eb1e9c323a.png)

![](RackMultipart20231013-1-eulm2e_html_e71c554b57758580.png)

