# UFC-MMA-Fight-Predictor

## Overview

In this project I build a Mixed Martial Arts Fight Predictor from the ground up, i.e. from scraping data to evaluating a model.

## Manual

- For the Infrastructure check [docker/README.md](https://github.com/ManosL/UFC-MMA-Fight-Predictor/blob/master/docker/README.md).
- In order to run the Machine Learning Experiments and Predictor(aka Demo) check [src/machine_learning/README.md](https://github.com/ManosL/UFC-MMA-Fight-Predictor/blob/master/src/machine_learning/README.md).

## Real-Time predictions

At 05/02/2022, with the data that I could acquire until now, I tested my model in two upcoming events "UFC Fight Night: Hermansson vs. Strickland" and "UFC 271" in order to see how it will do on real-time. In order to do that I run the demo and wrote the appropriate Match-up dataset which is in `./src/data/Matchups.csv` file. The results were the following:

#### UFC Fight Night: Hermansson vs. Strickland

The predictions done along with the actual results are the following:

```
1 . Jack Hermansson vs Sean Strickland
        Jack Hermansson: 40.0
        Sean Strickland: 60.0
        draw: 0.0

Actual Winner: Sean Strickland

2 . Punahele Soriano vs Nick Maximov
        Punahele Soriano: 36.67
        Nick Maximov: 63.33
        draw: 0.0

Actual Winner: Nick Maximov

3 . Shavkat Rakhmonov vs Carlston Harris
        Shavkat Rakhmonov: 63.33
        Carlston Harris: 36.67
        draw: 0.0

Actual Winner: Shavkat Rakhmonov

4 . Sam Alvey vs Brendan Allen
        Sam Alvey: 30.67
        Brendan Allen: 68.67
        draw: 0.67

Actual Winner: Brendan Allen

5 . Tresean Gore vs Bryan Battle
        Tresean Gore: 48.67
        Bryan Battle: 50.67
        draw: 0.67

Actual Winner: Bryan Battle

6 . Julian Erosa vs Steven Peterson
        Julian Erosa: 52.0
        Steven Peterson: 46.67
        draw: 1.33

Actual Winner: Julian Erosa

7 . Miles Johns vs John Castaneda
        Miles Johns: 68.67
        John Castaneda: 29.33
        draw: 2.0

Actual Winner: John Castaneda

8 . Hakeem Dawodu vs Michael Trizano
        Hakeem Dawodu: 56.67
        Michael Trizano: 40.67
        draw: 2.67

Actual Winner: Hakeem Dawodu

9 . Chidi Njokuani vs Marc-Andre Barriault
        Chidi Njokuani: 55.33
        Marc-Andre Barriault: 44.67
        draw: 0.0

Actual Winner: Chidi Njokuani

10 . Alexis Davis vs Julija Stoliarenko
        Alexis Davis: 36.67
        Julija Stoliarenko: 61.33
        draw: 2.0

Actual Winner: Alexis Davis

11 . Jailton Almeida vs Danilo Marques
        Jailton Almeida: 52.0
        Danilo Marques: 46.67
        draw: 1.33

Actual Winner: Jailton Almeida

12 . Jason Witt vs Phil Rowe
        Jason Witt: 45.33
        Phil Rowe: 54.0
        draw: 0.67

Actual Winner: Phil Rowe

13 . Malcolm Gordon vs Denys Bondar
        Malcolm Gordon: 47.33
        Denys Bondar: 52.0
        draw: 0.67

Actual Winner: Malcolm Gordon
```

This means that my predictor achieved approximately 76% accuracy on this event, by predicting correctly 10 out of 13 fights which is pretty good and probably its better than the performance of a casual fan like me, who cannot even predict correctly fights between two relatively unknown fighters.

#### UFC 271

The predictions done are the following(**results are pending because the event is on 12/02**):
```
1. Israel Adesanya vs Robert Whittaker
        Israel Adesanya: 64.0
        Robert Whittaker: 36.0
        draw: 0.0

Actual Winner: Israel Adesanya

2. Derrick Lewis vs Tai Tuivasa
        Derrick Lewis: 47.33
        Tai Tuivasa: 50.0
        draw: 2.67

Actual Winner: Tai Tuivasa

3. Jared Cannonier vs Derek Brunson
        Jared Cannonier: 38.67
        Derek Brunson: 61.33
        draw: 0.0

Actual Winner: Jared Cannonier

4. Kyler Phillips vs Marcelo Rojo
        Kyler Phillips: 72.0
        Marcelo Rojo: 28.0
        draw: 0.0

Actual Winner: Kyler Philips

5. Bobby Green vs Nasrat Haqparast
        Bobby Green: 40.0
        Nasrat Haqparast: 58.0
        draw: 2.0

Actual Winner: Bobby Green

6. Andrei Arlovski vs Jared Vanderaa
        Andrei Arlovski: 50.67
        Jared Vanderaa: 48.0
        draw: 1.33

Actual Winner: Andrei Arlovski

7. Roxanne Modafferi vs Casey O'Neill
        Roxanne Modafferi: 21.33
        Casey O'Neill: 78.67
        draw: 0.0

Actual Winner: Casey O'Neill

8. Alex Perez vs Matt Schnell
        Alex Perez: 64.0
        Matt Schnell: 36.0
        draw: 0.0

Actual Winner: N/A

9. William Knight vs Maxim Grishin
        William Knight: 59.33
        Maxim Grishin: 39.33
        draw: 1.33

Actual Winner: Maxim Grishin

10. Mana Martinez vs Ronnie Lawrence
        Mana Martinez: 49.33
        Ronnie Lawrence: 49.33
        draw: 1.33

Actual Winner: Ronnie Lawrence

11. Alexander Hernandez vs Renato Moicano
        Alexander Hernandez: 38.0
        Renato Moicano: 62.0
        draw: 0.0

Actual Winner: Renato Moicano

12. Carlos Ulberg vs Fabio Cherant
        Carlos Ulberg: 52.67
        Fabio Cherant: 45.33
        draw: 2.0

Actual Winner: Carlos Ulberg

13. AJ Dobson vs Jacob Malkoun
        AJ Dobson: 52.0
        Jacob Malkoun: 47.33
        draw: 0.67

Actual Winner: Jacob Malkoun

14. Douglas Silva de Andrade vs Sergey Morozov
        Douglas Silva de Andrade: 32.0
        Sergey Morozov: 68.0
        draw: 0.0

Actual Winner: Douglas Silva de Andrade

15. Jeremiah Wells vs Blood Diamond
        Jeremiah Wells: 42.67
        Blood Diamond: 56.0
        draw: 1.33

Actual Winner: Jeremiah Wells
```

This means that my predictor achieved approximately 57% accuracy on this event, by predicting correctly 8 out of 14 fights which is not so good, especially considering the performance on last event.

This means that our model is generally unstable considering its real-time performance in these two events. However, it is consoling that in most of the wrong prediction our model selects as winner the favorite fighter, given the odds of a bookmaker.

**DISCLAIMER**: This project was done for educational purposes ONLY and does not support any betting activity that will result in addiction and property losing. However, if you want to apply this model in the real world, I suggest use that in order to level up in [Verdict App](https://verdictmma.com/).