# Repo for paper
This is the repository associated with the paper "Interpreting atypical conditions in systems with deep conditional Autoencoders: the case of electrical consumption" submitted at ECML-PKDD 2019.

Through this repository, you can find:
- the code use to learn CVAEs, retrieve features and create tensorboard projections
- the core dataset which was used and some additional instances that were labeled during our experiments
- typical notebooks that were used during our experiments
- the tensorboard projections of all the CVAE models that were learnt and used for the paper

# Interactive Repo
Launch myBinder here to use this repo inetractively in any web navigator:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marota/Autoencoder_Embedding_Expert_Caracteristion_/branch_clement)

It will open a jupyter session in your web navigator. You can start by inspecting and running the notebooks which illustrate the exper iments of the paper and the use of the code.
Then you can explore the projections in tensorboard. Go into the projector folder. Click on New -> Tensorboard. The projector opens under Tensorboard in a new window and you are ready to start explore all projections! If it doesn't display, look for projector view on the right of Tensorboard.

# Illustration of projectors

Here is an illustration of the projections you can explore and the features/labels you can visualize. The goal is to help an expert caracterize a daily electrical load curve given features of interest for the expert. 

![Alt text](images/ProjectionProfilConso_Temp.png?raw=true "Title")


Moreover, you can identify days with similar load curves.

![Alt text](images/ProjectionSlectionJour.png?raw=true "Title")

To do so, launch the project under Binder. 

# Features to explore in projectors
To caracterize our data, here are different expert features that could be important:
- Max Temperature: maximum temperature over the day
- Min Temperature: minimum temperature over the day
- Month: month of the day
- Weeday: the day of the week
- is_Weekday: weeday or weekend
- Holiday: is the day a holiday ?

# Existing projectors
1. A simple 3 components PCA to use as baseline. But you can not condition your problem here along some features whereas Autoencoders can.
2. A simple 4 components Variational AUtoEncoder. The representation/projection can be compared to the PCA. When you spherize your data in tensorboard, this projection is a lot easier to navigate in and probably more interprtable. You then notice that is_WeekDay and Temperature are the most important fatures in those projections.
3. A 4 components Conditional Variational AutoEncoder by conditioning on Weedays which is an important feature as you can deduce with the previous VAE. In the projection, you can notice that it is agnostic to is_Weekday as it was pass as a condition whereas Temperature is still an important feature.
4. A 4 components Conditional Variational AutoEncoder by conditioning on Temperature with an embedding. Similar conclusion to 4. by switching Temperature and is_weeday
5. A 4 components Conditional Variational AutoEncoder by conditioning on is_weekday and Temperature with an embedding. The projection is agnostic now to those prominent features and you can explore which other features or examples are important. Especially you should notice that holidays look quite distinct now!

# Licence information
Copyright 2018 RTE(France)

RTE: http://www.rte-france.com

This Source Code is subject to the terms of the GNU Lesser General Public License v3.0. If a copy of the LGPL-v3 was not distributed with this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.fr.html.


