
# Idea

Restructure and train the models in the following manner.

1. create melspecs for the _entire_ song for each song.
2. modify my data generator to _continuously_ sample 2-second intervals from
   these melspecs.

This idea feels a lot better than restricting the model to only seeing 2 seconds
from each clip. This way, each time the generator loops through the dataset
it sees different parts of each song and trains based on that.

I think this could either make the models perform a lot better, since the amount
of data is more vast, or it could make the models train a lot worse, since the
models won't be seeing the same 2-second chunk during each epoch. Still, this
might be a great idea to get around the 2-second training barrier.

