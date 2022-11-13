# fast-frechet
Fast discrete Fréchet distance implementation optimized with numba. It can be used to evaluate trajectories generated during the  Trajectory Challange on AI Games 2022.

## Requirements
This Fréchet distance implementation requires [numpy](https://numpy.org) and [numba](https://numba.pydata.org) packages. You can use the following command to install them with pip.

    pip install numpy numba

## Usage
The following code illustrates an example usage of the `frechet_distance` function available in the `frechet.py` module,

```python
    import numpy as np
    from frechet import frechet_distance

    trajectory_p = np.array([[80.0644976552576, 50.6552672944963],
                             [71.4585771784186, 63.2156178820878],
                             [19.9234400875866, 12.8415436018258]])

    trajectory_q = np.array([[5.88378887623549, 11.4293440245092],
                             [84.2895035166293, 67.4984930083156],
                             [90.9000392071903, 36.4088270813227],
                             [34.2789062298834, 0.568102905526757],
                             [43.9584670122713, 75.5553565453738],
                             [24.4398877490312, 30.7297872845083],
                             [35.2576361969113, 39.8860249202698],
                             [62.438058713451, 44.4697478786111],
                             [38.4228205773979, 66.4192265830934]])

    score = frechet_distance(trajectory_p, trajectory_q)
```

---
The code in this repository is based on optimized discrete Fréchet distance implementation available in https://github.com/joaofig/discrete-frechet that is described in [this article](https://towardsdatascience.com/fast-discrete-fréchet-distance-d6b422a8fb77).