#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:48:42 2026

@author: jonathandimitri32
"""

import pygame
import numpy as np
import random
import time


# ============================================================
# V1 - PIANO ALÉATOIRE DANS UNE GAMME
# ============================================================

NOMBRE_TEMPS = 50

TEMPO_MIN = 60
TEMPO_MAX = 180

# Zone du piano utilisée : Do4 → Do6
NOTE_MIN = 60
NOTE_MAX = 84

FREQUENCE_ECHANTILLONNAGE = 44100


# ============================================================
# GAMME
# ============================================================

# Gamme de Do majeur :
#
# Do - Ré - Mi - Fa - Sol - La - Si
#
# Les valeurs représentent les positions dans une octave MIDI.
#
# Do  = 0
# Ré  = 2
# Mi  = 4
# Fa  = 5
# Sol = 7
# La  = 9
# Si  = 11

GAMME = [0, 2, 4, 5, 7, 9, 11]

NOM_GAMME = "Do majeur"


# ============================================================
# PROBABILITÉS DU NOMBRE DE NOTES SIMULTANÉES
# ============================================================

# 60 % : une seule note
# 25 % : deux notes
# 15 % : trois notes

PROBABILITES = [0.60, 0.25, 0.15]


# ============================================================
# INITIALISATION DU SON
# ============================================================

pygame.mixer.init(
    frequency=FREQUENCE_ECHANTILLONNAGE,
    size=-16,
    channels=1,
    buffer=512
)

# Plusieurs sons doivent pouvoir être joués simultanément.
pygame.mixer.set_num_channels(16)


# ============================================================
# CONVERSION MIDI -> FRÉQUENCE
# ============================================================

def midi_vers_frequence(note):

    return 440 * (2 ** ((note - 69) / 12))


# ============================================================
# CRÉATION DU SON
# ============================================================

def creer_note(note_midi, duree):

    frequence = midi_vers_frequence(note_midi)

    t = np.linspace(
        0,
        duree,
        int(FREQUENCE_ECHANTILLONNAGE * duree),
        endpoint=False
    )

    # Fréquence fondamentale + harmoniques
    signal = (
        np.sin(2 * np.pi * frequence * t)
        + 0.5 * np.sin(2 * np.pi * frequence * 2 * t)
        + 0.25 * np.sin(2 * np.pi * frequence * 3 * t)
        + 0.12 * np.sin(2 * np.pi * frequence * 4 * t)
    )

    # Décroissance du son
    enveloppe = np.exp(-3 * t / duree)

    signal = signal * enveloppe

    # Normalisation
    signal = signal / np.max(np.abs(signal))

    # Volume volontairement réduit,
    # car plusieurs notes peuvent être jouées simultanément
    signal = (signal * 32767 * 0.30).astype(np.int16)

    return pygame.sndarray.make_sound(signal)


# ============================================================
# NOMS DES NOTES
# ============================================================

NOMS_NOTES = [
    "Do",
    "Do#",
    "Ré",
    "Ré#",
    "Mi",
    "Fa",
    "Fa#",
    "Sol",
    "Sol#",
    "La",
    "La#",
    "Si"
]


def nom_note(note):

    nom = NOMS_NOTES[note % 12]

    octave = (note // 12) - 1

    return f"{nom}{octave}"


# ============================================================
# CONSTRUCTION DES NOTES AUTORISÉES
# ============================================================

notes_autorisees = []

for note in range(NOTE_MIN, NOTE_MAX + 1):

    position_dans_octave = note % 12

    if position_dans_octave in GAMME:

        notes_autorisees.append(note)


print()
print("Notes autorisées :")

for note in notes_autorisees:

    print(nom_note(note), end=" ")

print()
print()


# ============================================================
# CHOIX DU TEMPO
# ============================================================

tempo = random.randint(TEMPO_MIN, TEMPO_MAX)

duree_note = 60 / tempo


print("==============================")
print("       PIANO ALÉATOIRE V1")
print("==============================")
print()

print(f"Gamme : {NOM_GAMME}")
print(f"Tempo : {tempo} BPM")
print(f"Durée d'un temps : {duree_note:.2f} seconde(s)")

print()


# ============================================================
# MORCEAU
# ============================================================

for i in range(NOMBRE_TEMPS):

    # --------------------------------------------------------
    # CHOIX DU NOMBRE DE NOTES
    # --------------------------------------------------------

    nombre_notes = random.choices(
        population=[1, 2, 3],
        weights=PROBABILITES,
        k=1
    )[0]


    # --------------------------------------------------------
    # CHOIX DES NOTES
    # --------------------------------------------------------

    notes = random.sample(
        notes_autorisees,
        nombre_notes
    )

    # On les classe de la plus grave à la plus aiguë
    notes.sort()


    # --------------------------------------------------------
    # AFFICHAGE
    # --------------------------------------------------------

    noms = [nom_note(note) for note in notes]

    if nombre_notes == 1:

        type_evenement = "NOTE"

    elif nombre_notes == 2:

        type_evenement = "2 NOTES"

    else:

        type_evenement = "3 NOTES"


    print(
        f"{i + 1:02d} | "
        f"{type_evenement:8s} | "
        + " + ".join(noms)
    )


    # --------------------------------------------------------
    # CRÉATION DES SONS
    # --------------------------------------------------------

    sons = []

    for note in notes:

        son = creer_note(
            note,
            duree_note
        )

        sons.append(son)


    # --------------------------------------------------------
    # ON JOUE TOUTES LES NOTES EN MÊME TEMPS
    # --------------------------------------------------------

    for son in sons:

        son.play()


    # On attend un temps
    time.sleep(duree_note)


    # --------------------------------------------------------
    # ARRÊT DES NOTES
    # --------------------------------------------------------

    for son in sons:

        son.stop()


# ============================================================
# FIN
# ============================================================

pygame.mixer.quit()

print()
print("Morceau terminé.")