# Speaker Notes: Which Flower Is It?

Presented by Amir · about 5 minutes

## 1. Which Flower Is It?
*15 s, until 0:15*

Hi, I'm Amir. Eva and I worked on this project together. In the next five minutes I show how a garden centre can pick the right care guide from a customer's photo, with only 100 photos per flower.

## 2. The problem
*25 s, until 0:40*

Today every photo needs a person. We want a model that takes over the easy cases, and we want to know when it should not be trusted. Only 100 photos per flower is very little data for a neural network.

## 3. The data
*30 s, until 1:10*

The classes are balanced, so accuracy is a fair metric. The photos are very varied: close-ups, fields, bouquets.
The colour plot shows: sunflowers are yellow and often photographed against blue sky. Roses and tulips have almost the same colours, red and pink. So for those two the model needs shape, not colour.

## 4. Looking deeper at the data (Eva)
*25 s, until 1:35*

Before modelling, Eva looked at the data more closely. All flowers have the same photo shapes, so resizing them to a square is fair. Colour separates sunflowers, but roses and tulips are almost identical in colour, saturation and brightness. So we expected the model to need shape for these two.

## 5. Our approach: transfer learning
*30 s, until 2:05*

Instead of training from scratch, we reuse MobileNetV2. It was trained on 1.4 million ImageNet photos and already detects edges, textures and shapes. We keep that part frozen and only train a small new head for our three flowers: 3,843 weights instead of millions.

## 6. What we tried: overview
*20 s, until 2:25*

Here is every model we trained, on photos they never trained on: light blue is the validation set (60 photos), dark blue the test set (30 photos).
Chance is 33 percent. A CNN from scratch reaches 63. Every transfer learning model, mine and Eva's, is far above that: 80 to 97 percent.
Interesting: my model is best on validation, Eva's are best on test. The differences are 1 to 4 photos, so that is noise from small data.
The next two slides show what each of us tried, and what worked.

## 7. Experiments (1/2): Amir · 160 x 160 pixels
*30 s, until 2:55*

My part: I started with a CNN from scratch as a baseline: 63 percent, and it memorised the training photos. Transfer learning with a frozen MobileNetV2 and a small head reached 87 percent on validation, 90 on test.
Augmentation and early stopping reduced overfitting, but did not get more photos right. So the simple model stayed our choice.

## 8. Experiments (2/2): Eva · 224 x 224 pixels
*25 s, until 3:20*

Eva trained at 224 x 224, the original MobileNetV2 size, with the base inside the model: 48 of 60 validation photos, 29 of 30 test photos.
Her augmentation included a vertical flip, which we had decided against. It reduced overfitting, but did not change the accuracy at all, so it neither helped nor hurt.
Then she let Keras Tuner try 30 settings. The best one has no hidden layer, little dropout and a small learning rate: the simplest head won again. It has the lowest validation loss of all our models, so its probabilities are the most honest.
Her strong augmentation by hand reached the same accuracy, with a higher loss.

## 9. What worked, what didn't, and what we decided against
*30 s, until 3:50*

Summary. Transfer learning is the big win. Augmentation, early stopping and the tuner helped with overfitting and with honest probabilities, but changed the accuracy by only one photo or so. Both our own comparison and Keras Tuner ended up with the simplest head.
We disagreed about the vertical flip: Eva tried it, and it made no difference. Colour changes we left out on purpose.
The lesson: with 60 validation and 30 test photos, differences of a few photos are noise, so we choose by validation and keep the simplest model.

## 10. Final result on the test set
*25 s, until 4:15*

The test set was used only once, at the very end. 27 of 30 are right.
Surprise: we expected roses and tulips to be confused, but two of the three mistakes are sunflowers, the class that looked easiest.

## 11. When can the garden centre trust it?
*30 s, until 4:45*

The three wrong photos have one thing in common: none is a close-up of a single flower.
And the model is less sure when it is wrong: two of the three mistakes are below 70 percent. So the rule is: below 70 percent, a member of staff checks. On the test set that catches two of three mistakes, and only 2 of 30 photos need a person.
One confident mistake at 90 percent still slips through.
Eva's best model shows the same pattern: its two test mistakes, a rose far away in a landscape and an orange rose, come with only 41 and 58 percent, so the staff check would catch both.

## 12. Live demo
*40 s, until 5:25*

Switch to the app now: uv run streamlit run app.py, dense_head.keras is selected automatically.
Upload the clear sunflower: green box.
Upload the rose bouquet (data/test/rose, first file): yellow box, staff check.
If the demo fails: show screenshots.

## 13. Recommendation
*15 s, until 5:40*

To sum up: transfer learning took us from 63 to 90 percent, and we know when not to trust the model. Thank you, I'm happy to take questions.

## 14. Backup: our decisions
*only if asked*

Only for questions. Every decision with its reason is also in the notebook, section "Methods and Decisions at a Glance".
