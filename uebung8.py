import numpy as np


# Bitte diese Funktion selber implementieren
def viterbi( logLike, logPi, logA ):
    ...
    return stateSequence, pStar


def limLog(x):
    """
    Log of x.

    :param x: numpy array.
    :return: log of x.
    """
    MINLOG = 1e-100
    return np.log(np.maximum(x, MINLOG))



if __name__ == "__main__":
    # Vektor der initialen Zustandswahrscheinlichkeiten
    logPi = limLog([ 0.9, .0, 0.1 ])

    # Matrix der Zustandsübergangswahrscheinlichkeiten
    logA  = limLog([
      [ 0.8,  .0, 0.2 ], 
      [ 0.4, 0.4, 0.2 ], 
      [ 0.3, 0.2, 0.5 ] 
    ]) 

    # Beobachtungswahrscheinlichkeiten für "Regen", "Sonne", "Schnee" 
    # B = [
    #     {  2: 0.1,  3: 0.1,  4: 0.2,  5: 0.5,  8: 0.1 },
    #     { -1: 0.1,  1: 0.1,  8: 0.2, 10: 0.2, 15: 0.4 },
    #     { -3: 0.2, -2: 0.0, -1: 0.8,  0: 0.0 }
    # ]




    # gemessene Temperaturen (Beobachtungssequenz): [ 2, -1, 8, 8 ]
    # ergibt folgende Zustands-log-Likelihoods
    logLike = limLog([
      [ 0.1,  .0,  .0 ],
      [  .0, 0.1, 0.8 ],
      [ 0.1, 0.2,  .0 ],
      [ 0.1, 0.2,  .0 ]
    ])

    # erwartetes Ergebnis: [0, 2, 1, 1], -9.985131541576637
    print( viterbi( logLike, logPi, logA ) )


    # verlängern der Beobachtungssequenz um eine weitere Beobachung 
    # mit der gemessenen Temperatur 4
    # neue Beobachtungssequenz: [ 2, -1, 8, 8, 4 ]
    logLike = np.vstack( ( logLike, limLog([ 0.2, 0, 0 ]) ) )

    # erwartetes Ergebnis: [0, 2, 0, 0, 0], -12.105395077776727
    print( viterbi( logLike, logPi, logA ) )
