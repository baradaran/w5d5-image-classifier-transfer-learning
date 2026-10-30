---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 26px; }
  h1 { color: #2b6a3f; }
  h2 { color: #2b6a3f; }
  table { font-size: 20px; }
  .small { font-size: 18px; color: #555; }
  .big { font-size: 64px; font-weight: bold; color: #2b6a3f; }
---

<!-- _paginate: false -->

# Which Flower Is It?

## Choosing the right care guide from a photo with transfer learning

Eva & Amir · Flower Classification Project · Week 5

<!--
Hi, I'm Amir. Eva and I worked on this project together. In the next five minutes, I'll show you how a garden centre can pick the right care guide from a customer's photo, with only 100 photos per flower.
-->

---

## The problem

- A garden centre sends customers a **care guide** for the flower in their photo
- Today a **member of staff** looks at **every** photo and picks the guide by hand
- Start with the three most requested guides: **rose, sunflower, tulip**
- Only **100 training photos per flower**

**Question:** can a model pick the guide, and **when can the garden centre trust it?**

<!--
At the moment, a member of staff has to look at every single photo. We want a model that takes over the easy cases, and we want to know when the model should not be trusted. And 100 photos per flower is very little data for a neural network.
-->

---

## The data

![w:500](images/flower_grid.png) ![w:560](images/colours.png)

- 300 training, 60 validation, 30 **test** photos (locked until the end), balanced
- **Sunflowers:** yellow, plus blue sky · **Roses and tulips:** almost the same colours → the model must learn **shape**

<!--
Our three classes are balanced, so accuracy is a fair way to measure the model. The photos themselves are very varied, with close-ups, whole fields and bouquets.
The colour plot shows that sunflowers are mostly yellow and are often photographed against a blue sky. Roses and tulips, on the other hand, have almost the same colours, mostly red and pink. So to tell those two apart, the model has to learn their shape, not their colour.
-->

---

## Looking deeper at the data (Eva)

<style scoped> table { font-size: 19px; } </style>

| Check | Finding | What it means for the model |
|---|---|---|
| Width, height and aspect ratio **per flower** | Almost all photos are landscape (4:3); all three flowers have the **same shapes** | Resizing to a square squeezes every flower the same way: no false hint like "squeezed = tulip" |
| Hue of all coloured pixels | Sunflowers: yellow + a **blue sky** peak · roses and tulips: red and magenta | Sunflowers are easy by colour; the model may also use the sky |
| **Dominant colour per photo** | Roses and tulips very similar; tulips more often **dominantly green** (stems and leaves) | Colour barely separates roses from tulips |
| **Saturation and brightness** per flower | Roses and tulips almost identical | The model has to learn **shape**: open, layered roses against closed tulip cups |

<!--
Before we started modelling, Eva looked at the data more closely. She found that all three flowers have the same photo shapes, so resizing every photo to a square is fair. She also found that colour separates the sunflowers well, but that roses and tulips are almost identical in colour, saturation and brightness. That is why we expected the model to need the shape of the flower for these two.
-->

---

## Our approach: transfer learning

```text
photo  ->  MobileNetV2 base (frozen)  ->  1,280 features  ->  small head  ->  rose / sunflower / tulip
           pretrained on 1.4 million photos                    trained by us
           2,257,984 weights, not changed                      3,843 weights
```

- 300 photos are too few to train a network from scratch
- **Reuse** a network that already learnt to see: keep its base, replace its 1,000-class top
- **Freeze** the base: protects what it learnt, and only 3,843 weights have to be learnt from our photos

<!--
Instead of training a network from scratch, we reused MobileNetV2. It was trained on 1.4 million ImageNet photos, so it already knows how to detect edges, textures and shapes. We kept that part frozen and only trained a small new head for our three flowers. That means only 3,843 weights had to be learnt from our photos, instead of millions.
-->

---

## What we tried: overview

![w:900](images/what_we_tried.png)

<!--
This chart shows every model we trained, measured on photos they never saw during training. The light blue bars are the validation set with 60 photos, and the dark blue bars are the test set with 30 photos.
Guessing gives 33 percent, and a CNN trained from scratch reaches 63 percent. Every transfer learning model, mine and Eva's, is far above that, between 80 and 97 percent.
What is interesting is that my model is best on validation, while Eva's models are best on test. But the differences are only one to four photos, so that is just noise from having so little data.
On the next two slides, I'll show you what each of us tried and what worked.
-->

---

## Experiments (1/2): Amir · 160 x 160 pixels

<style scoped> table { font-size: 18px; } </style>

| Experiment | Idea | Validation (60) | Test (30) | Worked? |
|---|---|---|---|---|
| CNN from scratch | Baseline: all weights learnt from our 300 photos | 0.633 (38) | 0.633 (19) | ❌ overfits strongly, but **proves** we need transfer learning |
| **Frozen MobileNetV2, features once + dense head** | Reuse the pretrained base, train only 3,843 weights | **0.867 (52)** | **0.900 (27)** | ✅ **chosen**: best and simplest, trains in seconds |
| + data augmentation | Flip, rotate, zoom to reduce overfitting | 0.850 (51) | 0.867 (26) | ⚠️ less overfitting (gap 0.13 → 0.10), not more accuracy |
| + early stopping | Stop at the lowest validation loss | 0.833 (50) | – | ⚠️ more honest probabilities, 2 photos fewer |

<!--
Let me start with my part. As a baseline, I trained a CNN from scratch. It reached 63 percent and simply memorised the training photos. Transfer learning with a frozen MobileNetV2 and a small head reached 87 percent on validation and 90 percent on test.
I also tried data augmentation and early stopping. They reduced overfitting, but they did not get more photos right, so the simple model stayed our choice.
-->

---

## Experiments (2/2): Eva · 224 x 224 pixels

<style scoped> table { font-size: 18px; } </style>

| Experiment | Idea | Validation (60) | Test (30) | Worked? |
|---|---|---|---|---|
| Dense head, base inside the model | The same idea at the original MobileNetV2 size | 0.800 (48) | 0.967 (29) | ⚠️ 4 validation photos fewer than at 160, but the best test score |
| + augmentation | Flips (horizontal **and** vertical), rotation, zoom | 0.800 (48) | 0.967 (29) | ⚠️ less overfitting, same accuracy: the vertical flip neither helped nor hurt |
| **Keras Tuner (Hyperband)** | 30 settings: augmentation strength, dropout, hidden layer, L2, learning rate; objective: validation **loss** | **0.817 (49)** | 0.933 (28) | ✅ **Eva's best**: lowest validation loss (0.47) |
| Strong augmentation, by hand | + shift, brightness, random erasing; hidden layer 64, learning-rate decay, early stopping | 0.817 (49) | 0.933 (28) | ⚠️ same accuracy as the tuner, higher validation loss (0.57) |

**The tuner's choice:** no hidden layer, dropout 0.2, a small learning rate, light augmentation: **the simplest head won**, as in our own comparison.

<!--
Eva trained her models at 224 by 224 pixels, which is the original size of MobileNetV2, and she put the base inside the model. Her dense head got 48 of 60 validation photos and 29 of 30 test photos right.
Her augmentation also included a vertical flip, which I had decided against. It reduced overfitting, but it did not change the accuracy at all, so it neither helped nor hurt.
Then she let Keras Tuner try 30 different settings. The best one has no hidden layer, little dropout and a small learning rate, so once again the simplest head won. It also has the lowest validation loss of all our models, which means that its probabilities are the most honest.
Her strong augmentation by hand reached the same accuracy, but with a higher loss.
-->

---

## What worked, what didn't, and what we decided against

<style scoped> table { font-size: 19px; } </style>

| ✅ Worked | ⚠️ Helped a little / no effect | ❌ Did not work / decided against |
|---|---|---|
| **Transfer learning**: 0.63 → 0.80–0.87 on validation | Augmentation (both of us): less overfitting, not more accuracy | CNN from scratch: too little data |
| Freezing the base: only 3,843 weights to learn | Early stopping, Keras Tuner: more honest probabilities, ± 1 photo | **Colour changes**: colour is a key clue for sunflowers |
| **The simplest head**: our comparison **and** the tuner chose it | 224 instead of 160 pixels: not better on validation | Cropping to a square: could cut off the flower |
| Scaling **inside** the model: the app gets the same preparation | **Vertical flip** (Eva tried it): no effect, 48 of 60 with and without | |

**Lesson:** with 300 photos, the **pretrained features do most of the work**. Our model is best on validation, Eva's on test, by **1 to 4 photos**: that is noise, so we choose by validation and keep the simplest model.

<!--
So, what worked and what didn't? Transfer learning is clearly the big win. Augmentation, early stopping and the tuner helped against overfitting and gave more honest probabilities, but they changed the accuracy by only about one photo. Our own comparison and Keras Tuner both ended up with the simplest head.
Eva and I disagreed about the vertical flip, so she tried it, and it made no difference. We left out colour changes on purpose.
The lesson is that with 60 validation photos and 30 test photos, a difference of a few photos is just noise. So we choose the model on the validation set and keep the simplest one.
-->

---

## Final result on the test set

<div class="big">27 / 30 correct</div>

**90 % test accuracy**

![bg right:42% w:460](images/confusion_matrix.png)

- 30 photos the model **never saw** during any decision
- Validation was a fair estimate: 0.867 → 0.900 (one photo = 3.3 points)
- All 10 tulips correct · **2 of the 3 mistakes are sunflowers**, not roses vs tulips

<!--
We used the test set only once, at the very end, and the model got 27 of the 30 photos right.
What surprised us is that we expected roses and tulips to be confused, but two of the three mistakes are actually sunflowers, the class that looked easiest.
-->

---

## When can the garden centre trust it?

![w:540](images/wrong_photos.png) ![w:540](images/confidence.png)

- Every mistake is a **bouquet, a display or a far-away shot**, never a close-up of one flower
- **Below 70 % sure → a member of staff checks:** catches 2 of 3 mistakes, only 2 of 30 photos need a person
- Eva's best model confirms it: its 2 test mistakes (a far-away rose, an orange rose) are at **41 % and 58 %**

<!--
The three wrong photos have one thing in common. None of them is a close-up of a single flower. There is a bouquet, a shop display and a garden photographed from far away.
The model is also less sure when it is wrong, because two of the three mistakes are below 70 percent. So our rule is that every answer below 70 percent is checked by a member of staff. On the test set, that catches two of the three mistakes, and only 2 of the 30 photos need a person. One confident mistake at 90 percent still slips through.
Eva's best model shows the same pattern. Its two test mistakes, a rose far away in a landscape and an orange rose, came with only 41 and 58 percent, so the staff check would catch both of them.
-->

---

## Live demo

```text
customer uploads a photo  ->  model
                                |-- at least 70 % sure:  care guide sent automatically
                                '-- below 70 % sure:     a member of staff checks the photo
```

1. A clear **sunflower** → green: *"This is a sunflower (96 %)"*
2. The **rose bouquet** → yellow: *"Not sure (62 %): a tulip or a rose? A member of staff will check"*

<!--
[Switch to the app. It is already running, and dense_head.keras is selected. Upload data/test/sunflower, the first file.]
Let me show you the app. First, I upload a clear photo of a sunflower. The model is 96 percent sure, so the care guide would be sent automatically.
[Upload data/test/rose, the first file.]
Now I upload a rose bouquet from the test set. This time the model is only 62 percent sure, and it cannot decide between a tulip and a rose. So it asks a member of staff to check the photo, which is exactly the rule from the previous slide.
[If the demo fails, show the screenshots instead.]
-->

---

## Recommendation

- ✅ Use the model: **9 of 10** right, against 1 of 3 by guessing
- 📸 Ask customers for a **close-up of one flower**
- 👩‍🌾 **Staff check** every answer below **70 %**
- ℹ️ Say which flowers are covered: a daisy still gets one of the three labels

**Next:** more photos, fine-tuning the top of MobileNetV2, confirm the 70 % threshold on more data

<!--
To sum up, transfer learning took us from 63 to 90 percent, and we know when the model should not be trusted. Thank you for listening, and I'm happy to take your questions.
-->

---

<!-- _paginate: false -->

## Backup: our decisions

<style scoped> table { font-size: 15px; } th, td { padding: 4px 10px; } </style>

| Decision | Our choice | Why |
|---|---|---|
| Image size | 160 x 160 | Accepted by MobileNetV2, half the computing time of 224 x 224 |
| Batch size | 32 | Keras default, about 10 updates per epoch |
| Base | MobileNetV2, ImageNet, `include_top=False`, frozen | Small and fast, built for apps; 300 photos too few to change 2.26 M weights |
| Head | Dropout 0.2 + Dense(3, softmax) | Dropout against overfitting, softmax gives probabilities |
| Training | Adam 0.001, 30 epochs, features extracted once | Standard settings from class; trains in seconds |
| Augmentation | Horizontal flip, rotation ≤ 36°, zoom ≤ 10 % | Keeps the label true; no vertical flip (sky above, ground below), no colour changes |
| Model choice | Validation accuracy; on a tie the simpler model | 60 validation photos: one photo = 0.017 |
| Test set | Used once, at the end | An honest final score |
| Staff check | Below 70 % | Catches 2 of 3 mistakes on the test set; to confirm on more photos |

<!--
This slide is only for questions. Every decision and the reason for it is also in our notebook, in the section Methods and Decisions at a Glance.
-->
