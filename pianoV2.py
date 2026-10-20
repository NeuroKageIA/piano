#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:05:06 2026

@author: jonathandimitri32
"""

import pygame
import numpy as np
import random
import time




# ============================================================
# V2 - PIANO AVEC ACCORDS ET PROGRESSIONS MUSICALES
# ============================================================

NOMBRE_CYCLES = 4

TEMPO_MIN = 70
TEMPO_MAX = 130

NOTE_MIN = 48      # Do3
NOTE_MAX = 84      # Do6

FREQUENCE_ECHANTILLONNAGE = 44100


# ============================================================
# INITIALISATION AUDIO
# ============================================================

pygame.mixer.init(
    frequency=FREQUENCE_ECHANTILLONNAGE,
    size=-16,
    channels=1,
    buffer=512
)

pygame.mixer.set_num_channels(32)


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
# MIDI -> FRÉQUENCE
# ============================================================

def midi_vers_frequence(note):

    return 440 * (2 ** ((note - 69) / 12))


# ============================================================
# CRÉATION D'UNE NOTE
# ============================================================

def creer_note(note_midi, duree):

    frequence = midi_vers_frequence(note_midi)

    t = np.linspace(
        0,
        duree,
        int(FREQUENCE_ECHANTILLONNAGE * duree),
        endpoint=False
    )

    # Son composé de plusieurs harmoniques
    signal = (
        np.sin(2 * np.pi * frequence * t)
        + 0.45 * np.sin(2 * np.pi * frequence * 2 * t)
        + 0.20 * np.sin(2 * np.pi * frequence * 3 * t)
        + 0.08 * np.sin(2 * np.pi * frequence * 4 * t)
    )

    # Attaque très rapide
    attaque = np.minimum(t / 0.01, 1)

    # Décroissance progressive
    enveloppe = attaque * np.exp(-3 * t / duree)

    signal *= enveloppe

    signal /= np.max(np.abs(signal))

    # Volume réduit car plusieurs notes peuvent être superposées
    signal = (signal * 32767 * 0.25).astype(np.int16)

    return pygame.sndarray.make_sound(signal)


# ============================================================
# ACCORDS DE DO MAJEUR
# ============================================================

# Les nombres correspondent ici aux classes MIDI.
#
# Do = 0
# Ré = 2
# Mi = 4
# etc.

ACCORDS = {

    "Do majeur": [0, 4, 7],

    "Ré mineur": [2, 5, 9],

    "Mi mineur": [4, 7, 11],

    "Fa majeur": [5, 9, 0],

    "Sol majeur": [7, 11, 2],

    "La mineur": [9, 0, 4]
}


# ============================================================
# PROGRESSIONS D'ACCORDS
# ============================================================

PROGRESSIONS = [

    [
        "Do majeur",
        "Fa majeur",
        "Sol majeur",
        "Do majeur"
    ],

    [
        "Do majeur",
        "La mineur",
        "Fa majeur",
        "Sol majeur"
    ],

    [
        "Ré mineur",
        "Sol majeur",
        "Do majeur",
        "Do majeur"
    ],

    [
        "Do majeur",
        "Sol majeur",
        "La mineur",
        "Fa majeur"
    ],

    [
        "La mineur",
        "Fa majeur",
        "Do majeur",
        "Sol majeur"
    ]
]


# ============================================================
# RÉCUPÉRER TOUTES LES NOTES D'UN ACCORD
# ============================================================

def notes_de_accord(nom_accord):

    classes = ACCORDS[nom_accord]

    notes = []

    for note in range(NOTE_MIN, NOTE_MAX + 1):

        if note % 12 in classes:

            notes.append(note)

    return notes


# ============================================================
# CRÉER UNE VERSION JOUABLE DE L'ACCORD
# ============================================================

def choisir_notes_accord(nom_accord, nombre_notes):

    classes = ACCORDS[nom_accord]

    notes_choisies = []

    # On choisit une octave centrale
    octave_base = random.choice([3, 4])

    note_base = (octave_base + 1) * 12

    # On récupère une note pour chacune des trois notes
    # constituant l'accord
    candidats = []

    for classe in classes:

        note = note_base + classe

        # Certaines notes doivent être ramenées
        # dans l'octave correcte.
        while note < NOTE_MIN:
            note += 12

        while note > NOTE_MAX:
            note -= 12

        candidats.append(note)

    # Évite d'avoir par exemple Fa5, La5, Do5
    # dans un ordre bizarre.
    candidats.sort()

    # Si l'écart est trop étrange, on relève certaines notes
    for i in range(1, len(candidats)):

        while candidats[i] <= candidats[i - 1]:

            candidats[i] += 12

    candidats = [
        note
        for note in candidats
        if NOTE_MIN <= note <= NOTE_MAX
    ]

    # Choix de 1, 2 ou 3 notes parmi l'accord
    nombre_notes = min(nombre_notes, len(candidats))

    notes_choisies = random.sample(
        candidats,
        nombre_notes
    )

    notes_choisies.sort()

    return notes_choisies


# ============================================================
# JOUER PLUSIEURS NOTES SIMULTANÉMENT
# ============================================================

def jouer_notes(notes, duree):

    sons = []

    for note in notes:

        son = creer_note(
            note,
            duree
        )

        sons.append(son)

    # Toutes les notes démarrent presque simultanément
    for son in sons:

        son.play()

    time.sleep(duree)

    for son in sons:

        son.stop()


# ============================================================
# TEMPO
# ============================================================

tempo = random.randint(
    TEMPO_MIN,
    TEMPO_MAX
)

duree_temps = 60 / tempo


# ============================================================
# CHOIX D'UNE PROGRESSION
# ============================================================

progression = random.choice(PROGRESSIONS)


print()
print("======================================")
print("           PIANO MUSICAL V2")
print("======================================")
print()

print(f"Tempo : {tempo} BPM")

print()
print("Progression choisie :")

print(
    " -> ".join(progression)
)

print()


# ============================================================
# MORCEAU
# ============================================================

for cycle in range(NOMBRE_CYCLES):

    print()
    print(
        f"-------- CYCLE {cycle + 1} --------"
    )

    for nom_accord in progression:

        print()
        print(f"Accord : {nom_accord}")

        # ----------------------------------------------------
        # Chaque accord dure 4 temps
        # ----------------------------------------------------

        for temps in range(4):

            # -----------------------------------------------
            # Choix du nombre de notes
            # -----------------------------------------------

            nombre_notes = random.choices(
                population=[1, 2, 3],
                weights=[0.35, 0.25, 0.40],
                k=1
            )[0]

            # -----------------------------------------------
            # IMPORTANT :
            #
            # Les notes ne sont PLUS choisies
            # dans toute la gamme.
            #
            # Elles doivent appartenir
            # à l'accord actuel.
            # -----------------------------------------------

            notes = choisir_notes_accord(
                nom_accord,
                nombre_notes
            )

            noms = [
                nom_note(note)
                for note in notes
            ]

            print(
                f"Temps {temps + 1} : "
                + " + ".join(noms)
            )

            # -----------------------------------------------
            # Jouer
            # -----------------------------------------------

            jouer_notes(
                notes,
                duree_temps
            )


# ============================================================
# FIN
# ============================================================

pygame.mixer.quit()

print()
print("==============================")
print("Morceau terminé.")
print("==============================")