# sysid-neuralnet
Deep neural networks for system identification problems.

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
