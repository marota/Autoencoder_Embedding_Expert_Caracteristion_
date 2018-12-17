Launch myBinder here to use this repo inetractively in any web navigator:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/marota/Autoencoder_Embedding_Expert_Caracteristion_/master)

# Autoencoder_Embedding_Expert_Caracteristion_

In this repository, you will find several projectors that can be visualised in Tensorboard and which helps an expert caracterize a daily electrical load curve given features of interest for the expert. 

![Alt text](images/ProjectionProfilConso_Temp.png?raw=true "Title")


Moreover, you can identify days with similar load curves.

![Alt text](images/ProjectionSlectionJour.png?raw=true "Title")

To do so, launch the project under Binder. It will open a jupyter session in your web navigator. Then go into a folder of a projector you want to explore. Finally, click on New -> Tensorboard. The projector opens under Tensorboard in a new window and you are ready to start explore! 

# Features to explore in projectors
- Max Temperature: maximum temperature over the day
- Min Temperature: minimum temperature over the day
- Month: month of the day
- Weeday: the day of the week
- is_Weekday: weeday or weekend
- Holiday: is the day a holiday ?

# Existing projectors
1. A simple 5 components PCA to use as baseline. But you can not condition your problem here along some features whereas Autoencoders can.
2. A simple 5 components Variational AUtoEncoder. The representation/projection can be compared to the PCA. When you spherize your data in tensorboard, this projection is a lot easier to navigate in and probably more interprtable. You then notice that is_WeekDay and Temperature are the most important fatures in those projections.
3. A 5 components Conditional Variational AutoEncoder by conditioning on Weedays which is an important feature as you can deduce with the previous VAE. In the projection, you can notice that it is agnostic to is_Weekday as it was pass as a condition whereas Temperature is still an important feature.
4. A 5 components Conditional Variational AutoEncoder by conditioning on Temperature with an embedding. Similar conclusion to 4. by switching Temperature and is_weeday
5. A 5 components Conditional Variational AutoEncoder by conditioning on is_weekday and Temperature with an embedding. The projection is agnostic now to those prominent features and you can explore which other features or examples are important. Especially you should notice that holidays look quite distinct now!

# Licence information
Copyright 2018 RTE(France)

RTE: http://www.rte-france.com

This Source Code is subject to the terms of the GNU Lesser General Public License v3.0. If a copy of the LGPL-v3 was not distributed with this file, You can obtain one at https://www.gnu.org/licenses/lgpl-3.0.fr.html.


