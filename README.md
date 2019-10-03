# sysid-neuralnet
Deep neural networks for system identification problems.

**NOTE**: This code implement the examples of the paper: ``Deep  Convolutional  Networks  in  System  Identification``. Better documentation and more readable code structure is a working in progress...

# Interactive mode
The code is possible to run in  interactive mode mainly to enable easy access to evaluation 

To get access to the model in interactive mode write

```python
import run
(model, loaders, options) = run.run(load_model="logs/train_213214124/model.pt")

all_output = []
for i, (u, y) in enumerate(loaders["train"]):
    all_output += [model(u,y)]

all_output = np.concat(all_output,0)
```
