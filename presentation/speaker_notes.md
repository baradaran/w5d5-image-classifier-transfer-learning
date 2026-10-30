# Speaker Notes: Which Flower Is It?

Presented by Amir · about 5 minutes

## 1. Which Flower Is It?
*15 s, until 0:15*

Hi, I'm Amir. Eva and I worked on this project together. In the next five minutes, I'll show you how a garden centre can pick the right care guide from a customer's photo, with only 100 photos per flower.

## 2. The problem
*25 s, until 0:40*

At the moment, a member of staff has to look at every single photo. We want a model that takes over the easy cases, and we want to know when the model should not be trusted. And 100 photos per flower is very little data for a neural network.

## 3. The data
*30 s, until 1:10*

Our three classes are balanced, so accuracy is a fair way to measure the model. The photos themselves are very varied, with close-ups, whole fields and bouquets.
The colour plot shows that sunflowers are mostly yellow and are often photographed against a blue sky. Roses and tulips, on the other hand, have almost the same colours, mostly red and pink. So to tell those two apart, the model has to learn their shape, not their colour.

## 4. Looking deeper at the data (Eva)
*25 s, until 1:35*

Before we started modelling, Eva looked at the data more closely. She found that all three flowers have the same photo shapes, so resizing every photo to a square is fair. She also found that colour separates the sunflowers well, but that roses and tulips are almost identical in colour, saturation and brightness. That is why we expected the model to need the shape of the flower for these two.

## 5. Our approach: transfer learning
*30 s, until 2:05*

Instead of training a network from scratch, we reused MobileNetV2. It was trained on 1.4 million ImageNet photos, so it already knows how to detect edges, textures and shapes. We kept that part frozen and only trained a small new head for our three flowers. That means only 3,843 weights had to be learnt from our photos, instead of millions.

## 6. What we tried: overview
*20 s, until 2:25*

This chart shows every model we trained, measured on photos they never saw during training. The light blue bars are the validation set with 60 photos, and the dark blue bars are the test set with 30 photos.
Guessing gives 33 percent, and a CNN trained from scratch reaches 63 percent. Every transfer learning model, mine and Eva's, is far above that, between 80 and 97 percent.
What is interesting is that my model is best on validation, while Eva's models are best on test. But the differences are only one to four photos, so that is just noise from having so little data.
On the next two slides, I'll show you what each of us tried and what worked.

## 7. Experiments (1/2): Amir · 160 x 160 pixels
*30 s, until 2:55*

Let me start with my part. As a baseline, I trained a CNN from scratch. It reached 63 percent and simply memorised the training photos. Transfer learning with a frozen MobileNetV2 and a small head reached 87 percent on validation and 90 percent on test.
I also tried data augmentation and early stopping. They reduced overfitting, but they did not get more photos right, so the simple model stayed our choice.

## 8. Experiments (2/2): Eva · 224 x 224 pixels
*25 s, until 3:20*

Eva trained her models at 224 by 224 pixels, which is the original size of MobileNetV2, and she put the base inside the model. Her dense head got 48 of 60 validation photos and 29 of 30 test photos right.
Her augmentation also included a vertical flip, which I had decided against. It reduced overfitting, but it did not change the accuracy at all, so it neither helped nor hurt.
Then she let Keras Tuner try 30 different settings. The best one has no hidden layer, little dropout and a small learning rate, so once again the simplest head won. It also has the lowest validation loss of all our models, which means that its probabilities are the most honest.
Her strong augmentation by hand reached the same accuracy, but with a higher loss.

## 9. What worked, what didn't, and what we decided against
*30 s, until 3:50*

So, what worked and what didn't? Transfer learning is clearly the big win. Augmentation, early stopping and the tuner helped against overfitting and gave more honest probabilities, but they changed the accuracy by only about one photo. Our own comparison and Keras Tuner both ended up with the simplest head.
Eva and I disagreed about the vertical flip, so she tried it, and it made no difference. We left out colour changes on purpose.
The lesson is that with 60 validation photos and 30 test photos, a difference of a few photos is just noise. So we choose the model on the validation set and keep the simplest one.

## 10. Final result on the test set
*25 s, until 4:15*

We used the test set only once, at the very end, and the model got 27 of the 30 photos right.
What surprised us is that we expected roses and tulips to be confused, but two of the three mistakes are actually sunflowers, the class that looked easiest.

## 11. When can the garden centre trust it?
*30 s, until 4:45*

The three wrong photos have one thing in common. None of them is a close-up of a single flower. There is a bouquet, a shop display and a garden photographed from far away.
The model is also less sure when it is wrong, because two of the three mistakes are below 70 percent. So our rule is that every answer below 70 percent is checked by a member of staff. On the test set, that catches two of the three mistakes, and only 2 of the 30 photos need a person. One confident mistake at 90 percent still slips through.
Eva's best model shows the same pattern. Its two test mistakes, a rose far away in a landscape and an orange rose, came with only 41 and 58 percent, so the staff check would catch both of them.

## 12. Live demo
*40 s, until 5:25*

[Switch to the app. It is already running, and dense_head.keras is selected. Upload data/test/sunflower, the first file.]
Let me show you the app. First, I upload a clear photo of a sunflower. The model is 96 percent sure, so the care guide would be sent automatically.
[Upload data/test/rose, the first file.]
Now I upload a rose bouquet from the test set. This time the model is only 62 percent sure, and it cannot decide between a tulip and a rose. So it asks a member of staff to check the photo, which is exactly the rule from the previous slide.
[If the demo fails, show the screenshots instead.]

## 13. Recommendation
*15 s, until 5:40*

To sum up, transfer learning took us from 63 to 90 percent, and we know when the model should not be trusted. Thank you for listening, and I'm happy to take your questions.

## 14. Backup: our decisions
*only if asked*

This slide is only for questions. Every decision and the reason for it is also in our notebook, in the section Methods and Decisions at a Glance.
