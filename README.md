# Routing Problems for Multiple Cooperative UAVs using Transformers

## Paper
Solving a variant of the Orieentering Problem (OP) called the Orienteering Problem with Multiple Prizes and Types of
Node (OP-MP-TN) with a cooperative multi-agent system based on Transformer Networks. For more details, please see our
[paper](). If this repository is useful for your work, please cite our paper:

```

``` 

## Dependencies

* Python >= 3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/) >= 1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib
* [k-means-constrained](https://joshlk.github.io/k-means-constrained/)
* fuzzy-c-means
* scikit-learn

## Usage

First, it is necessary to create training, testing, and validation sets:
```bash
python create_dataset.py --name train --seed 1111 --graph_sizes 20 --dataset_sizes 1280000 --cluster km --num_agents 2 --max_length 2
python create_dataset.py --name test --seed 1234 --graph_sizes 20 --dataset_sizes 10000 --cluster km --num_agents 2 --max_length 2
python create_dataset.py --name val --seed 4321 --graph_sizes 20 --dataset_sizes 10000 --cluster km --num_agents 2 --max_length 2
```
Note that the option `--cluster` defines the type of clustering for the initial planning: K-Means(`km`), K-Means
constrained(`kmc`), or Fuzzy C-Means(`fcm`). The option `--num_agents` defines the number of agents/clusters. The option
`max_length` indicates the normalized time limit to solve the problem.

To train a Transformer model (`attention`) use:
```bash
python run.py --model attention --graph_size 20 --max_length 2 --num_agents 2 --cluster km --data_dist coop --baseline rollout --train_dataset data/op/1depots/2agents/coop/km/20/train_seed1111_L2.pkl --val_dataset data/op/1depots/2agents/coop/km/20/val_seed4321_L2.pkl
```

Pointer Network (`pointer`) and Graph Pointer Network (`gpn`) can also be trained with the `--model` option. To resume
training, load your last saved model with the `--resume` option.

Evaluate your trained models with:
```bash
python eval.py --model outputs/op_coop20/attention_run... --num_agents 2 --test_dataset data/op/1depots/2agents/coop/km/20/test_seed1234_L2.pkl
```
If the epoch is not specified, by default the last one in the folder will be used.

Baselines like [OR-Tools](https://developers.google.com/optimization), [Gurobi](https://www.gurobi.com),
[Tsiligirides](https://www.tandfonline.com/doi/abs/10.1057/jors.1984.162),
[Compass](https://github.com/bcamath-ds/compass) or a [Genetic Algorithm](https://github.com/mc-ride/orienteering) can
be executed as follows:
```bash
python -m problems.op.op_baseline --method ortools --multiprocessing True --datasets data/op/1depots/2agents/coop/km/20/test_seed1234_L2.pkl
```
To run Compass, you need to install it by running the `install_compass.sh` script from within the `problems/op`
directory. To use Gurobi, obtain a ([free academic](http://www.gurobi.com/registration/academic-license-reg)) license
and follow the
[installation instructions](https://www.gurobi.com/documentation/8.1/quickstart_windows/installing_the_anaconda_py.html)
. OR-Tools has to be installed too (`pip install ortools`).

Finally, you can visualize an example of executions using:
```bash
python test_plot.py --graph_size 20 --num_agents 2 --data_dist coop --load_path outputs/op_coop20/attention_run... --test_coop True
```

Use the `--baseline` option to visualize the prediction of one of the baselines mentioned before:
```bash
python test_plot.py --graph_size 20 --num_agents 2 --data_dist coop --baseline ortools --test_coop True
```

### Other options and help
```bash
python run.py -h
python eval.py -h
python -m problems.op.op_baseline -h
python test_plot.py -h
```

## Acknowledgements
This repository is an adaptation of
[wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for the case of multiple
cooperative UAVs.
