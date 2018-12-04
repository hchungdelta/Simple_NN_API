# Performance test 

Optimizer: adam
learning rate: 0.02
training data: 1,000 sentences from south park's script.
amount of vocabulary: 1033 (without using pre-trained embedding vectors)

Test model:
1. LSTM teacher foring mode (black line)
2. LSTM infer mode (gray line)
3. SandGlass + Batch Normalization (green line)

Test model detail:
sentence length : 32 (without cutoff, in order to compare these models on the same footing.)
LSTM:
one layer Bi-RNN
hidden units = 320 

SangGlass: 


Loss-epoch curves 



Loss-time curves

