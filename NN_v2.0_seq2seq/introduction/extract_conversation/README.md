# extract conversation from novels

The conversation can be easily extracted by punctuations 「」.
However, in many circumstances, two or more sentences extracted by this approach may not be related.
In other words, these are not a conversation between people.

In order to extract "the conversation between people", of course the-state-of-art machine learning is possible to archive this.
However, it is beyond the score of my current focus.
Therefore I adopt a rough way to solve this. I set up a threshold for "sentence distance" between sentences in the novel. 
It is common to imagine that if two sentences appear in a row in the novel, it is highly likely that these sentences are conversations between two or more people.
The following is an example of this approach:



<img src="sample_extract.gif" width="920">



In the example, two conversations are extracted (green and orange underlines), by rearranging the conversation, they can now serve as bi-directional conversation training data. As what follows.



<img src="convsation.gif" width="760">


In my current work, as to reduce the calculational cost, I set an upper limit of the length of sentence by 20 dictionary units*. If a sentence is longer then this limit, it will be separated into a front part and an end part, while the mid part will be deleted. The front part is used to replace the original sentence in interlocutor B (decoder part), and the end part is used to replace the original sentence in interlocutor A(encoder part). The notion is that, at the beginning of the paragraph, it is more likely that the speaker is trying to "answer" someone's question. And the listener would like to reply to the speaker based on the information provided at the end of the paragraph.

For example, if the original conversation is as follows:
```
A:「Bさん！ご意見お聞かせて。」
B:「はい、でもそれあくまで私個人の持論ですね。（～ご意見～１０００字）　お役にたてれば、幸いと思います！」
A：「ありがとうございます。勉強になりました！」
```
It can be separated into:
```
A:「Bさん！ご意見お聞かせて。」                 B:「はい、でもそれあくまで私個人の持論ですね。」
B:「お役にたてれば、幸いと思います！」           A：「ありがとうございます。勉強になりました！」
```



20 dictionary units*: I use dictionary units here since the upper limit length of sentence is somewhat ambiguous in my model.
In extreme case, 6 characters can be encoded into 1 dictionary unit.   
For example: ありがとう。is equals to ありがとう + 。 and hence is only 2 units.





