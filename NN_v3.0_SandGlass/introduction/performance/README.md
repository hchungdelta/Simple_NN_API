# Performance test 


## Test model:
```
1. LSTM teacher foring mode (black line)
2. LSTM infer mode (gray line)
3. SandGlass + Batch Normalization (green line)
```
## Test model detail:
```
sentence length : 32 (without cutoff, in order to compare these models on the same footing.)
LSTM:
one layer Bi-RNN
hidden units = 320 
```
```
Optimizer: adam
learning rate: 0.002
training data: 1,000 sentences from south park's script.
amount of vocabulary: 1033 (without using pre-trained embedding vectors)
```

SangGlass: 


Loss-epoch curves 

<img src="performance_epochs.png" width="550">

Loss-time curves

<img src="performance_cost.png" width="550">
