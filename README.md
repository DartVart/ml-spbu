# ML course SPBU

## Task 1. Generation of Russian Names

Here you can generate russian (and other) names. Based on [code](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb) from [@karpathy](https://github.com/karpathy).

### Training
To launch training go to `tasks/task1` directory and run:
```commandline
python train.py train.txt
```
where `train.txt` is a file that includes russian names to train a model.

Example of output:
```
Training started...
Training finished!
Train loss: 2.04
Trained model saved in the file "model.pth".
```

### Testing
To test a model go to `tasks/task1` directory and run:
```commandline
python test.py model.pth test.txt
```
where
* `model.pth` is a file that includes model weights
* `test.txt` is a file that includes russian names to test a model

Example of output:
```
Test loss: 2.37

Examples of names generated by the model:
горьенсина.
секж.
авгагуллуаниктий.
антей.
ботсинуфа.
едиля.
тара.
гелия.
альгобросиньмера.
глина.
```