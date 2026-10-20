import pygame
import numpy as np
import random
import time


# ============================================================
# V0 - PIANO ALÉATOIRE
# ============================================================

NOMBRE_NOTES = 50

TEMPO_MIN = 60
TEMPO_MAX = 180

# Notes MIDI entre Do4 et Do6
NOTE_MIN = 60
NOTE_MAX = 84

FREQUENCE_ECHANTILLONNAGE = 44100


# ============================================================
# INITIALISATION DU SON
# ============================================================

pygame.mixer.init(
    frequency=FREQUENCE_ECHANTILLONNAGE,
    size=-16,
    channels=1,
    buffer=512
)


# ============================================================
# CONVERSION NOTE MIDI -> FRÉQUENCE
# ============================================================

def midi_vers_frequence(note):
    return 440 * (2 ** ((note - 69) / 12))


# ============================================================
# CRÉATION D'UN SON "PIANO"
# ============================================================

def creer_note(note_midi, duree):

    frequence = midi_vers_frequence(note_midi)

    t = np.linspace(
        0,
        duree,
        int(FREQUENCE_ECHANTILLONNAGE * duree),
        endpoint=False
    )

    # Fondamentale + quelques harmoniques
    signal = (
        np.sin(2 * np.pi * frequence * t)
        + 0.5 * np.sin(2 * np.pi * frequence * 2 * t)
        + 0.25 * np.sin(2 * np.pi * frequence * 3 * t)
        + 0.12 * np.sin(2 * np.pi * frequence * 4 * t)
    )

    # Le son diminue progressivement, comme une touche de piano
    enveloppe = np.exp(-3 * t / duree)

    signal = signal * enveloppe

    # Normalisation
    signal = signal / np.max(np.abs(signal))

    signal = (signal * 32767 * 0.5).astype(np.int16)

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
# CHOIX DU TEMPO
# ============================================================

tempo = random.randint(TEMPO_MIN, TEMPO_MAX)

# Chaque note est une noire
duree_note = 60 / tempo


print()
print("==============================")
print("       PIANO ALÉATOIRE V0")
print("==============================")
print()

print(f"Tempo choisi : {tempo} BPM")
print(f"Durée d'une note : {duree_note:.2f} seconde(s)")
print()


# ============================================================
# MORCEAU
# ============================================================

for i in range(NOMBRE_NOTES):

    note = random.randint(NOTE_MIN, NOTE_MAX)

    print(
        f"{i + 1:02d} | "
        f"{nom_note(note):4s} | "
        f"MIDI {note}"
    )

    son = creer_note(note, duree_note)

    son.play()

    time.sleep(duree_note)

    son.stop()


# ============================================================
# FIN
# ============================================================

pygame.mixer.quit()

print()
print("Morceau terminé.")