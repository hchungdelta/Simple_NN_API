# Extract character specific conversation
In the Japanese language, one intriguing thing is that some words can explicitly indicate the person's gender, personalities, characteristics, even though there is no masculine and feminine noun system in the Japanese language system (as in Spanish language ...etc.) 

In English, if one want to mention his of herself, just use "I".
However, in the Japanese language, there are many words can be used as "I" depended on the situation, speaker's and listener's identity, 


| "I" in Japanese language| description  | 
| :---             |     :---:         | 
| 私               |  vanilla version  | 
| 私（わたくし）    | formally          | 
| あたし           |    feminine, usually used by teenagers  | 
|  僕              |    masculine, usually used by teenagers | 
|  俺              |   masculine, rough, would has kind of negative connotation | 
|  わし            |   masculine, a little bit old-fashioned, usually used by elders | 

In my opinions, the Japanese language can serve as a fruitful, and fascinating field for the speaker-recognition system, which can further imporve the character-specific seq2seq model. 

In this project, I want to train my model to have a certain characteristic (Yes, frankly speak, younger-sister-like-characteristic).  While due to lack of personal embedding data, I adopt a short-term palliative. As aforementioned, the identities are explicitly indicated by certain words, so I only need to use the conversations with certain keywords in it and also omit those conversations involve banned words in reply, as following shows.

```
conversation = [[question],[reply]]  # e.g. question:ありがとう！　reply:どういたしまして。 
keywords=[妹、兄、あたし]
bannedwords=[俺、僕、わし]
for conversation in conversations:
    if keywords in conversation and bannedwords not in reply:
        training_data.append(conversation)
```

After this process, the training data has been reduced to only 80,000 sentences. (originally there are 660,000 sentences)
