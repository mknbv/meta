## Code for paper ["Trace norm regularization for multi-task learning with scarce data"](https://arxiv.org/abs/2202.06742v1)

To install the package:

```bash
pip install -e .
```

Running multiple experiments simultaneously is done through task spooler.
To install it run:
```bash
sudo apt-get install task-spooler
```

Running meta script generates experiment files from which the plots
could be produced as done in the `notebooks/plots.ipynb` Jupyter
notebook. To run figure 1 experiments:

```bash
meta --ntasks-range 100 6400 7 --task-size 10 \
    --num-runs 12 --logdir logdir/figure.01
```

figure 2 experiments:

```bash
meta --ntasks-range 100 6400 7 --task-size 25 \
    --num-runs 12 --logdir logdir/figure.02
```

figure 3 experiments:

```bash
meta --task-size-range 5 30 9 --ntasks 800 \
    --num-runs 12 --logdir logdir/figure.03
```

figure 4 experiments:

```bash
meta --ntasks-range 100 6400 7 --features-dist adversarial \
    --task-size 25 --num-runs 12 --logdir logdir/figure.04
```

figure 5 experiments:

```bash
meta --label-scale-range 1e-3 2 14 --ntasks 800 --task-size 10 \
    --num-runs 12 --logdir logdir/figure.05
```
