# Decision Tree with Rust

## Abstract of the Project
As a part of my lab "Efficient AI with Rust" at the university of RWTH Aachen, I had the chance to implement a decision tree using ID3 algorithm with Rust.

We gathered a dataset of air condition features alongside their target as whether or not we decide to play in the weather.

To test the project we simply need to run `carog run -- ../my_csv.csv Play`, where as the first argument is the path to the training dataset and the second argument is the name of the target column.

In the code in `decision_tree/src/main.rs` we are giving the values for a test instance. Feel free to play around with these values.

We trained our decision tree on the csv placed in `my_csv.csv`, and got the following as our trained tree:
<p float="left">
  <img src="tree.png" width="700" />
</p>

