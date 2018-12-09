# Performance test 


## Test model:
```
The diagrams are shown below.
1. SandGlass + Batch Normalization (green line)
2. SandGlass without Batch Normalization (olive line)
3. LSTM teacher foring mode (black line)
4. LSTM infer mode (gray line)

```

## Loss-epoch curves 

<img src="loss_epoch.png" width="550">

BN is short for batch normalization. The results indicate that in terms of loss-epoch tendency, the performance of this model is similar to LSTM (infer mode). If includes BN, the model can converge much faster, approaches to the performance of LSTM (teacher forcing mode). In general, SandGlass Conv2Conv model can approach to lower loss with less fluctuation.

## Loss-time curves

<img src="loss_time.png" width="550">

In terms of efficiency, the results indicate that the cost-performance of this new model can outperform LSTM. It is interesting to see that even without BN (curve with olive color), the convergence rate is still comparable to LSTM (teacher forcing mode). 



## Test model detail:
```
sentence length: 32 (without cutoff, in order to compare these models on the same footing.)
batch: 100
epoch: 20 steps (run through 2,000 data)
Optimizer: adam
learning rate: 0.002
training data: 1,000 sentences from south park's script.
amount of vocabulary: 1033 (without using pre-trained embedding vectors)
```

- **SandGlass**: 
 
<img src="test_info.png" width="500">

- **LSTM**:
```
Dense layer (1033 x 80) -> Bi-RNN -> decoder RNN -> Dense layer (320 x 1033)
hidden units = 320
```



