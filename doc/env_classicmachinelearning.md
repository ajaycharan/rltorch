These environments correspond to classical machine learning problems mapped as RL problems.

# MulticlassClassification_v0
(see t6_multiclass_classification.lua)

This environment simulates a classical iterative training procedure for multiclass classification problems where one provides a training and testing set. At each timestep, the agent receives a new data point. It then has to predict the category of this point (action). As a feedback, the agent receives a 0/1 reward, but also the true category of the datapoint

* Initialization parameters:
  * `parameters.training_examples` is a (n x N) matrix where n is the number of training examples, and N the dimension of the input space
  * `parameters.training_labels` is a (n x 1) matrix where each value is the label (int) of the corresponding example. Labels must be between 1 and C
  * `parameters.testing_examples` (optionnal) is a (n' x N) matrix where n' is the number of testing examples
  * `parameters.testing_labels` (optionnal) is a (n' x 1) matrix
* Action space: `Discrete(C)` where `C` is the number of possible categories
* Observation space: `R^N`, each observation is a datapoint
* Reset: When using `reset`, you have to specify if you want to use (`reset(true)`) or not (`reset(false)`) the testing mode: 
  * During the training mode, training examples are sampled uniformly in the training set, and the environment never stops
  * During the testing mode, testing examples are sampled from the first one to the last one in the testing set (one trajectory is an iteration over all the testing examples). The environment finihes when all the examples have been sampled
* Feedback
  * Reward: 0 or 1 if the action corresponds to the category of the preceding observation
  * `feedback.true_action=c` where `c` is the true category of the preceding observation. This is used for example by imtation policies

# SparseSequentialLearning_v0 
(see t8_predictivepolicy.lua)

This environment implements the MDP described in `Gabriel Dulac-Arnold, Ludovic Denoyer, Philippe Preux, Patrick Gallinari:Sequential approaches for learning datum-wise sparse representations. Machine Learning 89(1-2): 87-122 (2012)`:
* It is based on both a training and testing dataset with associated labels (see `MulticlassClassification_v0`)
* Each new trajectory is based on a randomly chosen training example. Each action corresponds to a features to acquire. The observation is the value of the acquired features. 
* At each timestep, the environment returns a `feedback.target` value that corresponds to the true category of the corresponding example. No reward is provided!!

Details:
* Initialization: 
  * `parameters.training_examples` is a (n x N) matrix where n is the number of training examples, and N the dimension of the input space
  * `parameters.training_labels` is a (n x 1) matrix where each value is the label (int) of the corresponding example. Labels must be between 1 and C
  * `parameters.testing_examples` (optionnal) is a (n' x N) matrix where n' is the number of testing examples
  * `parameters.testing_labels` (optionnal) is a (n' x 1) matrix
* Action space: `Discrete(N)` where `N` is the number of features
* Observation space: `R^1` the value of the acquired feature
* Reset: 
  * when using `reset(use_test)`, the `use_test` flag says if the new state is based on a testing or training example. If you are using the training set, a new training example is sampled uniformly, and the next trajectory will focus on this example. If you are using the testing set, then the next test example (or first one if switching from train to test) will be used. The reset returns a `(0)` observation since no feature has been acquired for this new example
* Feedback:
  * Reward: NO
  * `feedback.target=torch.Tensor({c})` where `c` is the category of the current example

This environment can be used with (Budgeted) Predictive Policies



