#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIANO IA V4 — apprentissage des notes, des durées, du rythme et du tempo (CPU).

Installation : python3 -m pip install numpy mido pygame
Entraîner : python3 pianoV4.py --train --dataset ./maestro_midi --epochs 20
Composer : python3 pianoV4.py --generate
Son exportable : python3 pianoV4.py --generate --sans-son --wav morceau.wav
Rendu audio : python3 pianoV4.py --generate --volume 1.0 --resonance 1.2
Comparer : --seed 12 --tempo 100 --tempo-fixe --rythme-fixe

L'ancien modele_piano_regles.npz (format 1) est repris : poids ET états Adam
notes/progressions conservés. Les quatre réseaux temporels sont neufs ; ils
nécessitent un entraînement sur les MIDI. Avant la première migration, une
copie .avant_rythme_tempo.npz est créée. Le format 2 contient tous les réseaux.

Règles conservées : six triades, cinq progressions répétées, une à trois notes
par attaque, deux octaves. Influence des notes : 65 % règles + 35 % réseau.
Les nouveaux réseaux choisissent attaque/repos, durée (0.25 à 4 noires), tempo
initial (40 à 200 BPM par pas de 5) et variations de tempo par mesure.
Une case sans attaque laisse finir les notes précédentes. Les durées de touche
apprises sont conservées, puis le son décroît avec une résonance réglable
(1.2 seconde par défaut). --resonance 0 produit un rendu sec. Le niveau d'écoute
est remis à niveau sur le morceau entier ; --volume règle ensuite son intensité.
Ces réglages concernent la lecture Pygame et le WAV. Le MIDI conserve les
durées de touche apprises ; son rendu dépend de l'instrument qui le lit.
Le fichier modèle et ses signatures sont inchangés : aucun réentraînement
n'est nécessaire pour cette correction du rendu sonore.

Les MIDI sont simplifiés sur une grille de doubles croches, en groupes de
quatre noires ; ni les autres métriques, ni toutes les voix/ornements ne sont
reproduits fidèlement. Les durées sont quantifiées. La pédale source est ignorée.

TEMPO : toutes les commandes set_tempo sont lues. --source-tempo auto conserve
les changements explicites. Si le MIDI n'a qu'un tempo de référence fixe (cas
fréquent de MAESTRO), une pulsation est estimée à partir des attaques, par
autocorrélation locale et suivi dynamique inspiré d'Ellis (2007). Ce tempo est
une estimation : ambiguïtés moitié/double possibles. Si la confiance est trop
faible, retour au tempo MIDI. --source-tempo midi désactive cette estimation.
Les changements sont ensuite ramenés à un tempo moyen par groupe de 4 noires.

L'extraction n'utilise que les MIDI fournis. Séparation train/validation par
fichiers avant extraction, doublons exacts retirés ; les rôles des anciens
fichiers connus sont conservés. Le meilleur état de CHAQUE réseau est retenu.
Les entrées communes sont stockées en float16 pour la mémoire, les calculs en
float32/64. --max-files 0 signifie tous les fichiers, sous-dossiers compris.

Référence de l'estimation : https://www.ee.columbia.edu/~dpwe/pubs/Ellis07-beattrack.pdf
MIDI/tempo : https://mido.readthedocs.io/en/latest/files/midi.html
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import time

# De petites matrices n'ont pas besoin de dizaines de threads BLAS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("Installer NumPy : python3 -m pip install numpy") from exc


# ------------------------- Vos règles initiales -------------------------
NOMBRE_CYCLES = 4
TEMPO_MIN, TEMPO_MAX = 70, 130
NOTE_MIN, NOTE_MAX = 48, 84
SUSTAIN = 3.5  # Durée de touche de secours / --rythme-fixe, en temps.
RESONANCE_SECONDES = 1.2  # Résonance APRÈS le relâchement de la touche.
FREQUENCE_ECHANTILLONNAGE = 44100
NOMS_NOTES = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
ACCORDS = {
    "Do majeur": [0, 4, 7], "Ré mineur": [2, 5, 9],
    "Mi mineur": [4, 7, 11], "Fa majeur": [5, 9, 0],
    "Sol majeur": [7, 11, 2], "La mineur": [9, 0, 4],
}
NOMS_ACCORDS = list(ACCORDS)
PROGRESSIONS = [
    ["Do majeur", "Fa majeur", "Sol majeur", "Do majeur"],
    ["Do majeur", "La mineur", "Fa majeur", "Sol majeur"],
    ["Ré mineur", "Sol majeur", "Do majeur", "Do majeur"],
    ["Do majeur", "Sol majeur", "La mineur", "Fa majeur"],
    ["La mineur", "Fa majeur", "Do majeur", "Sol majeur"],
]
PROGRESSIONS_IDS = [[NOMS_ACCORDS.index(nom) for nom in p] for p in PROGRESSIONS]
PROBABILITES_NOMBRE = np.array([0.35, 0.25, 0.40], dtype=np.float64)
HISTORIQUE = 4
NB_NOTES = NOTE_MAX - NOTE_MIN + 1
NB_FEATURES = 6 + 7 + HISTORIQUE * NB_NOTES + 4 + 2
FORMAT_MODELE = 2
ECHELLE_CORRECTION = 1.0
SCRIPT_DIR = Path(__file__).resolve().parent

# Sept sous-ensembles d'une triade, dans chacune des deux octaves centrales.
SOUS_ENSEMBLES = [c for k in (1, 2, 3) for c in itertools.combinations(range(3), k)]
ACTIONS = [(octave, indices) for octave in (3, 4) for indices in SOUS_ENSEMBLES]
PRIOR_NOTES = np.array([
    0.5 * PROBABILITES_NOMBRE[len(indices) - 1] / math.comb(3, len(indices))
    for _, indices in ACTIONS
], dtype=np.float64)
PRIOR_PROGRESSIONS = np.full(len(PROGRESSIONS), 1.0 / len(PROGRESSIONS))


def nom_note(note: int) -> str:
    return f"{NOMS_NOTES[note % 12]}{note // 12 - 1}"


def notes_action(accord: int, action: int) -> list[int]:
    octave, indices = ACTIONS[action]
    triade = sorted((octave + 1) * 12 + pc for pc in ACCORDS[NOMS_ACCORDS[accord]])
    return [triade[i] for i in indices]


def contexte(accord, precedent, historique, temps, tempo) -> np.ndarray:
    """Même représentation pour apprendre et composer : quatre temps passés."""
    x = np.zeros(NB_FEATURES, dtype=np.float32)
    if accord is not None:
        x[accord] = 1
    x[6 + (6 if precedent is None else precedent)] = 1
    offset = 13
    historique = ([[]] * HISTORIQUE + list(historique))[-HISTORIQUE:]
    for i, notes in enumerate(historique):
        for note in notes:
            if NOTE_MIN <= note <= NOTE_MAX:
                x[offset + i * NB_NOTES + note - NOTE_MIN] = 1
    offset += HISTORIQUE * NB_NOTES
    x[offset + temps % 4] = 1
    x[-2] = (np.clip(tempo, TEMPO_MIN, TEMPO_MAX) - 100) / 30
    x[-1] = len(historique[-1]) / 3
    return x


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=-1, keepdims=True)


class CorrectionNeuronale:
    """MLP dont la sortie corrige les logits des règles ; le mélange les ancre."""

    def __init__(self, prior, hidden=64, seed=42, features=NB_FEATURES):
        self.prior = np.asarray(prior, dtype=np.float64)
        self.prior /= self.prior.sum()
        self.log_prior = np.log(self.prior)
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, 1 / np.sqrt(features), (features, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            # Initialiser seulement la SORTIE à zéro, pas tout le réseau.
            "W2": np.zeros((hidden, len(prior)), dtype=np.float32),
            "b2": np.zeros(len(prior), dtype=np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.step = 0

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        h = np.tanh(x @ self.params["W1"] + self.params["b1"])
        u = h @ self.params["W2"] + self.params["b2"]
        p = softmax(self.log_prior + ECHELLE_CORRECTION * u)
        return p, (x, h, u)

    def probabilities(self, x, influence=1.0):
        p, _ = self.forward(x)
        return (1 - influence) * self.prior + influence * p

    def loss_and_gradients(self, x, targets, anchor=0.20):
        p, (x, h, u) = self.forward(x)
        # CE des données + CE du prior : ancrage KL(prior || réseau).
        cible = (1 - anchor) * targets + anchor * self.prior
        loss = -np.mean(np.sum(cible * np.log(np.maximum(p, 1e-15)), axis=1))
        dz = ((p - cible) / len(x)).astype(np.float32)
        du = dz * ECHELLE_CORRECTION
        dh = (du @ self.params["W2"].T) * (1 - h * h)
        grads = {
            "W2": h.T @ du, "b2": du.sum(axis=0),
            "W1": x.T @ dh, "b1": dh.sum(axis=0),
        }
        return float(loss), grads

    def update(self, grads, learning_rate):
        norm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values()))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        self.step += 1
        for k in self.params:
            g = grads[k] * scale
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            m_hat = self.m[k] / (1 - 0.9 ** self.step)
            v_hat = self.v[k] / (1 - 0.999 ** self.step)
            self.params[k] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

    def train_epoch(self, x, y, batch_size, learning_rate, anchor, rng):
        if not len(x):
            return
        order = rng.permutation(len(x))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            _, grads = self.loss_and_gradients(x[indices], y[indices], anchor)
            self.update(grads, learning_rate)

    def nll(self, x, y):
        if not len(x):
            return None
        total = 0.0
        for start in range(0, len(x), 1024):
            p, _ = self.forward(x[start:start + 1024])
            total -= float(np.sum(y[start:start + 1024] * np.log(np.maximum(p, 1e-15))))
        return total / len(x)


def signature_regles():
    regles = [ACCORDS, PROGRESSIONS, NOTE_MIN, NOTE_MAX, HISTORIQUE,
              PROBABILITES_NOMBRE.tolist(), ACTIONS, ECHELLE_CORRECTION]
    return hashlib.sha256(json.dumps(regles, ensure_ascii=False).encode()).hexdigest()



# ------------------- Rythme et tempo appris (format 2) -------------------
SUBDIVISIONS = 4  # Une case = une double croche = 1/4 de noire.
PAS = 1.0 / SUBDIVISIONS
DUREES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
TEMPOS = np.arange(40.0, 201.0, 5.0)
RATIOS_TEMPO = np.array([0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2])
NB_FEATURES_TEMPS = NB_FEATURES + 24
TACHES = ("notes", "progressions", "rythme", "durees", "tempo_initial", "tempos")
CLASSES = dict(notes=14, progressions=5, rythme=2, durees=len(DUREES),
               tempo_initial=len(TEMPOS), tempos=len(RATIOS_TEMPO))
ALGORITHME_TEMPO = "onsets-autocorrelation-dp-v1"


def signature_temporelle():
    return hashlib.sha256(json.dumps([SUBDIVISIONS, DUREES.tolist(), TEMPOS.tolist(),
        RATIOS_TEMPO.tolist(), NB_FEATURES_TEMPS, ALGORITHME_TEMPO]).encode()).hexdigest()


class EtatRythme:
    def __init__(self):
        self.attaques = deque([0] * 16, maxlen=16)
        self.durees = deque([1.0] * 4, maxlen=4)
        self.variations = deque([1.0] * 3, maxlen=3)
        self.fin_note = 0.0


def contexte_temporel(accord, precedent, historique, beat, tempo, etat):
    base = contexte(accord, precedent, historique, int(beat) % 4, tempo)
    extra = np.zeros(24, np.float32)
    extra[int(round(beat * SUBDIVISIONS)) % SUBDIVISIONS] = 1
    extra[4 + (int(beat) // 4) % 4] = 1
    extra[8] = np.log2(max(tempo, 1) / 100)
    extra[9:12] = np.log2(list(etat.variations))
    extra[12:16] = np.log2(list(etat.durees)) / 2
    extra[16:20] = list(etat.attaques)[-4:]
    extra[20] = np.clip(etat.fin_note - beat, 0, 4) / 4
    extra[21] = sum(etat.attaques) / 16
    extra[22] = float(beat == 0)
    extra[23] = min(beat / 64, 1)
    return np.concatenate((base, extra))


def entree_reseau(nom, x):
    x = np.asarray(x, np.float32)
    if nom in ("notes", "progressions"):
        x = x[:, :NB_FEATURES].copy()
        if nom == "progressions":
            x[:, :6] = 0  # Progression encore inconnue, comme à la composition.
    elif nom == "tempo_initial":
        # Le tempo à prédire ne doit jamais être présent dans son entrée.
        initial = contexte_temporel(None, None, [], 0, 100, EtatRythme())
        x = np.broadcast_to(initial, (len(x), len(initial))).copy()
    return x


class PianoIA:
    def __init__(self, seed=42):
        self.notes = CorrectionNeuronale(PRIOR_NOTES.copy(), hidden=64, seed=seed)
        self.progressions = CorrectionNeuronale(PRIOR_PROGRESSIONS.copy(), hidden=32, seed=seed + 1)
        self.rythme = CorrectionNeuronale([0.75, 0.25], hidden=48, seed=seed + 2, features=NB_FEATURES_TEMPS)
        self.durees = CorrectionNeuronale(np.full(len(DUREES), 1 / len(DUREES)),
                                          hidden=48, seed=seed + 3, features=NB_FEATURES_TEMPS)
        self.tempo_initial = CorrectionNeuronale(np.full(len(TEMPOS), 1 / len(TEMPOS)),
                                                 hidden=32, seed=seed + 4, features=NB_FEATURES_TEMPS)
        prior_tempo = np.full(len(RATIOS_TEMPO), 0.025)
        prior_tempo[np.argmin(abs(RATIOS_TEMPO - 1))] = 0.8
        self.tempos = CorrectionNeuronale(prior_tempo, hidden=48, seed=seed + 5, features=NB_FEATURES_TEMPS)
        self.vues = {nom: np.zeros(CLASSES[nom], np.int64) for nom in TACHES}
        self.metadata = {"trained_epochs": 0, "epochs_par_reseau": {nom: 0 for nom in TACHES}}

    def reseaux(self):
        return {nom: getattr(self, nom) for nom in TACHES}

    def save(self, path, etats=None):
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.metadata, format=FORMAT_MODELE, rules_signature=signature_regles(),
                    temporal_signature=signature_temporelle())
        arrays = {"metadata": np.array(json.dumps(meta, ensure_ascii=False))}
        for nom, net in self.reseaux().items():
            etat = etats[nom] if etats is not None else capturer(net)
            meta[f"{nom}_step"] = etat["step"]
            for kind in ("params", "m", "v"):
                prefix = "" if kind == "params" else kind + "_"
                for k, value in etat[kind].items():
                    arrays[f"{nom}_{prefix}{k}"] = value
            arrays[f"{nom}_vues"] = self.vues[nom]
        # Les anciens noms de compteurs restent lisibles dans les métadonnées.
        meta["note_step"], meta["progression_step"] = meta["notes_step"], meta["progressions_step"]
        arrays["metadata"] = np.array(json.dumps(meta, ensure_ascii=False))
        fd, temporary = tempfile.mkstemp(prefix=".piano-", suffix=".npz", dir=path.parent)
        os.close(fd)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            version = meta.get("format")
            if version not in (1, FORMAT_MODELE) or meta.get("rules_signature") != signature_regles():
                raise ValueError("Modèle incompatible avec ces règles musicales.")
            if version == FORMAT_MODELE and meta.get("temporal_signature") != signature_temporelle():
                raise ValueError("Modèle incompatible avec cette grille de rythme/tempo.")
            noms = ("notes", "progressions") if version == 1 else TACHES
            for nom in noms:
                net = getattr(model, nom)
                for k, expected in net.params.items():
                    for kind, dest in (("", net.params), ("m_", net.m), ("v_", net.v)):
                        value = np.asarray(data[f"{nom}_{kind}{k}"], dtype=np.float32)
                        if value.shape != expected.shape or not np.isfinite(value).all():
                            raise ValueError(f"Poids invalides : {nom}/{kind}{k}.")
                        dest[k] = value.copy()
                key = ("note_step" if nom == "notes" else "progression_step") if version == 1 else f"{nom}_step"
                net.step = int(meta[key])
                if net.step < 0:
                    raise ValueError("Compteur d'apprentissage invalide.")
                if version != 1:
                    seen = np.asarray(data[f"{nom}_vues"], dtype=np.int64)
                    if seen.shape != (CLASSES[nom],) or (seen < 0).any():
                        raise ValueError("Classes observées invalides.")
                    model.vues[nom] = seen.copy()
        if version == 1:
            epochs = int(meta.get("trained_epochs", 0))
            meta["epochs_par_reseau"] = {nom: epochs if nom in ("notes", "progressions") else 0 for nom in TACHES}
            meta["migration_depuis_format"] = 1
        model.metadata = meta
        return model


def capturer(net):
    return {"step": net.step, **{kind: {k: v.copy() for k, v in getattr(net, kind).items()}
                               for kind in ("params", "m", "v")}}


def restaurer(net, state):
    net.step = state["step"]
    for kind in ("params", "m", "v"):
        setattr(net, kind, {k: v.copy() for k, v in state[kind].items()})


def require_mido():
    try:
        import mido
    except ImportError as exc:
        raise ValueError("Installer Mido : python3 -m pip install mido") from exc
    return mido


@dataclass
class NoteMIDI:
    start: float
    end: float
    pitch: int
    velocity: int


class CarteTempo:
    """Conversion exacte par morceaux entre noires MIDI et secondes."""
    def __init__(self, points):
        unique = {0.0: 120.0}
        for beat, bpm in points:
            if beat < 0 or not np.isfinite(beat) or not 0 < bpm <= 2000:
                raise ValueError("Carte de tempo invalide.")
            unique[float(beat)] = float(bpm)
        self.beats = np.array(sorted(unique), dtype=np.float64)
        self.bpms = np.array([unique[b] for b in self.beats], dtype=np.float64)
        self.secondes_cumulees = np.r_[0, np.cumsum(np.diff(self.beats) * 60 / self.bpms[:-1])]

    def secondes(self, beats):
        b = np.asarray(beats, dtype=np.float64)
        indices = np.maximum(0, np.searchsorted(self.beats, b, side="right") - 1)
        result = self.secondes_cumulees[indices] + (b - self.beats[indices]) * 60 / self.bpms[indices]
        return float(result) if result.ndim == 0 else result

    def bpm_moyen(self, debut, fin):
        return (fin - debut) * 60 / max(self.secondes(fin) - self.secondes(debut), 1e-9)


@dataclass
class MorceauMIDI:
    notes: list[NoteMIDI]
    tempo: CarteTempo
    source_tempo: str = "midi"
    confiance: float = 1.0


def lire_midi(path):
    mido = require_mido()
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("MIDI de type 2 ou résolution SMPTE non pris en charge.")
    programs, actives = defaultdict(int), defaultdict(deque)
    notes, tempos = [], [(0, 120)]
    tick = 0
    for msg in mido.merge_tracks(midi.tracks):
        tick += msg.time
        beat = tick / midi.ticks_per_beat
        if beat > 20000:
            raise ValueError("MIDI trop long : plus de 20 000 noires.")
        if msg.type == "set_tempo":
            if msg.tempo <= 0:
                raise ValueError("Tempo MIDI nul.")
            tempos.append((beat, 60_000_000 / msg.tempo))
        elif msg.type == "program_change":
            programs[msg.channel] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            if msg.channel != 9 and programs[msg.channel] <= 7:
                actives[(msg.channel, msg.note)].append((beat, msg.velocity))
        elif msg.type in ("note_off", "note_on"):
            key = (msg.channel, msg.note)
            if actives[key]:
                start, velocity = actives[key].popleft()
                if beat > start:
                    notes.append(NoteMIDI(start, beat, msg.note, velocity))
    for (_, pitch), pending in actives.items():
        for start, velocity in pending:
            if beat > start:
                notes.append(NoteMIDI(start, beat, pitch, velocity))
    if not notes:
        raise ValueError("Aucune note de piano exploitable.")
    return MorceauMIDI(sorted(notes, key=lambda n: n.start), CarteTempo(tempos))


def estimer_pulsation(starts, velocities, fin):
    """Pulsation indicative : autocorrélation locale + suivi par programmation dynamique.

    Adaptation aux attaques MIDI, inspirée d'Ellis (2007), sans dépendance audio.
    Une subdivision peut être prise pour un temps : ce n'est pas une partition.
    """
    sr = 50.0
    if len(starts) < 12 or fin < 6:
        return None, 0.0
    n = int(math.ceil(fin * sr)) + 1
    strength = np.zeros(n, np.float64)
    indices = np.clip(np.rint(starts * sr).astype(int), 0, n - 1)
    np.add.at(strength, indices, np.sqrt(np.asarray(velocities) / 127))
    strength = np.sqrt(strength)
    strength = np.convolve(strength, np.array([.15, .6, 1, .6, .15]), mode="same")
    candidates = np.arange(40.0, 201.0, 2)
    lags = 60 * sr / candidates
    centers = np.arange(0, fin + 8, 8)
    observations, confidences = [], []
    for center in centers:
        a, b = max(0, int((center - 10) * sr)), min(n, int((center + 10) * sr))
        segment = strength[a:b].copy()
        if len(segment) < 160 or not np.any(segment):
            observations.append(np.zeros(len(candidates)))
            confidences.append(0.0)
            continue
        segment -= segment.mean()
        segment *= np.hanning(len(segment))
        size = 1 << (2 * len(segment) - 1).bit_length()
        spectrum = np.fft.rfft(segment, size)
        ac = np.fft.irfft(abs(spectrum) ** 2, size)[:len(segment)]
        ac /= max(ac[0], 1e-9)
        axis = np.arange(len(ac))
        raw = sum(w * np.interp(k * lags, axis, ac, right=0)
                  for k, w in ((1, 1), (2, .5), (3, .25)))
        # Préférence large de pulsation, pas une étiquette certaine.
        raw *= np.exp(-0.5 * (np.log2(candidates / 100) / .9) ** 2)
        confidences.append(float(max(raw.max(), 0)))
        observations.append(raw / max(raw.max(), 1e-9))
    observations = np.asarray(observations)
    transition = 25 * np.log(candidates[:, None] / candidates[None, :]) ** 2
    scores, back = observations[0].copy(), []
    for obs in observations[1:]:
        options = scores[:, None] - transition
        parents = options.argmax(axis=0)
        scores = obs + options[parents, np.arange(len(candidates))]
        back.append(parents)
    path = [int(scores.argmax())]
    for parents in reversed(back):
        path.append(int(parents[path[-1]]))
    local_bpm = candidates[np.array(path[::-1])]
    confidence = float(np.median(confidences))
    if confidence < .035:
        return None, confidence
    expected = 60 * sr / np.interp(np.arange(n) / sr, centers, local_bpm)
    strength /= max(float(np.std(strength)), 1e-8)
    score = np.zeros(n)
    previous = np.full(n, -1, np.int32)
    for i in range(n):
        period = expected[i]
        lo = max(15, int(.65 * period))
        hi = min(75, int(1.45 * period), i)
        if hi < lo:
            score[i] = strength[i]
            continue
        delays = np.arange(lo, hi + 1)
        options = score[i - delays] - 100 * np.log(delays / period) ** 2
        j = int(options.argmax())
        if options[j] > 0:
            previous[i] = i - delays[j]
            score[i] = strength[i] + .98 * options[j]
        else:
            score[i] = strength[i]
    tail = max(0, n - int(2 * expected[-1]))
    cursor = tail + int(score[tail:].argmax())
    beats = []
    while cursor >= 0:
        beats.append(cursor / sr)
        cursor = int(previous[cursor])
    beats = np.array(beats[::-1])
    if len(beats) < 8:
        return None, confidence
    # Couvrir les deux extrémités pour que l'interpolation n'écrase aucune note.
    first_period = float(np.median(np.diff(beats[:min(9, len(beats))])))
    last_period = float(np.median(np.diff(beats[-min(9, len(beats)):])))
    before, after = [], []
    t = beats[0] - first_period
    while t > -first_period:
        before.append(t)
        t -= first_period
    before.append(t)
    t = beats[-1] + last_period
    while t < fin + last_period:
        after.append(t)
        t += last_period
    after.append(t)
    return np.r_[before[::-1], beats, after], confidence


def preparer_tempo(morceau, source="auto"):
    varie = len(np.unique(np.round(morceau.tempo.bpms, 5))) > 1
    if source == "midi" or (source == "auto" and varie):
        return morceau
    starts = morceau.tempo.secondes([n.start for n in morceau.notes])
    ends = morceau.tempo.secondes([n.end for n in morceau.notes])
    anchors, confidence = estimer_pulsation(starts, [n.velocity for n in morceau.notes], float(max(ends)))
    if anchors is None:
        morceau.source_tempo, morceau.confiance = "midi (estimation incertaine)", confidence
        return morceau
    grid = np.arange(len(anchors), dtype=np.float64)
    new_starts, new_ends = np.interp(starts, anchors, grid), np.interp(ends, anchors, grid)
    notes = [NoteMIDI(float(a), float(b), n.pitch, n.velocity)
             for n, a, b in zip(morceau.notes, new_starts, new_ends)]
    bpms = 60 / np.diff(anchors)
    return MorceauMIDI(notes, CarteTempo(list(enumerate(bpms))), "estime", confidence)


# ----------------------- Exemples issus des MIDI -----------------------
def choisir_transposition(notes):
    poids = np.zeros(12)
    for note in notes:
        poids[note.pitch % 12] += min(note.end - note.start, 8) * note.velocity
    gamme = {0, 2, 4, 5, 7, 9, 11}
    scores = [sum(poids[pc] for pc in range(12) if (pc + shift) % 12 in gamme) for shift in range(12)]
    return int(np.argmax(scores))


def action_observee(accord, groupe):
    pcs = sorted(ACCORDS[NOMS_ACCORDS[accord]])
    retenues = [note for note in groupe if note.pitch % 12 in pcs]
    if not retenues or len(retenues) < .5 * len(groupe):
        return None
    indices = tuple(i for i, pc in enumerate(pcs) if any(n.pitch % 12 == pc for n in retenues))
    centres = {pc: float(np.median([n.pitch for n in retenues if n.pitch % 12 == pc]))
               for pc in pcs if any(n.pitch % 12 == pc for n in retenues)}
    octave = min((3, 4), key=lambda o: sum(abs((o + 1) * 12 + pc - c) for pc, c in centres.items()))
    return ACTIONS.index((octave, indices))


def classe_proche(values, value):
    return int(np.argmin(abs(np.log(values / max(float(value), 1e-9)))))


@dataclass
class Exemples:
    x: np.ndarray
    y: dict
    stats: dict

    def indices(self, nom):
        target = self.y[nom]
        mask = target.sum(axis=1) > 0 if nom == "progressions" else target >= 0
        return np.flatnonzero(mask)


def extraire_exemples(path, transpose=True, max_examples=512, seed=42, source_tempo="auto"):
    morceau = preparer_tempo(lire_midi(path), source_tempo)
    shift = choisir_transposition(morceau.notes) if transpose else 0
    shift = shift if shift <= 6 else shift - 12
    notes = [NoteMIDI(n.start, n.end, n.pitch + shift, n.velocity) for n in morceau.notes]
    # Retirer seulement les mesures entièrement silencieuses avant le jeu.
    offset = 4 * int(min(n.start for n in notes) // 4)
    notes = [NoteMIDI(n.start - offset, n.end - offset, n.pitch, n.velocity) for n in notes]
    fin = max(n.end for n in notes)
    n_beats = max(1, int(math.ceil(fin)))
    n_bars = (n_beats + 3) // 4
    n_cells = n_beats * SUBDIVISIONS
    chromas = np.zeros((n_bars, 12))
    clusters = defaultdict(list)
    for note in notes:
        first = int(note.start // 4)
        last = min(n_bars - 1, int(max(note.start, note.end - 1e-9) // 4))
        for bar in range(first, last + 1):
            overlap = max(0, min(note.end, 4 * (bar + 1)) - max(note.start, 4 * bar))
            chromas[bar, note.pitch % 12] += overlap * note.velocity
        cell = min(n_cells - 1, max(0, int(math.floor(note.start * SUBDIVISIONS + .5))))
        clusters[cell].append(note)
    scores = np.column_stack([chromas[:, ACCORDS[name]].sum(axis=1) for name in NOMS_ACCORDS])
    chords = scores.argmax(axis=1)
    quality = scores.max(axis=1) / np.maximum(chromas.sum(axis=1), 1e-9)
    bpms = np.array([morceau.tempo.bpm_moyen(offset + 4 * b, offset + 4 * (b + 1)) for b in range(n_bars)])
    bpms = np.clip(bpms, TEMPOS[0], TEMPOS[-1])
    rng = np.random.default_rng(seed)
    # Tirage UNIFORME des cases, y compris sans attaque : ne pas apprendre
    # un faux taux de silences en suréchantillonnant les notes.
    indices = np.arange(n_cells)
    if n_cells > max_examples:
        indices = np.sort(rng.choice(n_cells, max_examples, replace=False))
        if 0 not in indices:
            indices[0] = 0
            indices.sort()
    selected = set(indices.tolist())
    xs = []
    ys = {nom: [] for nom in TACHES}
    history = deque([[]] * HISTORIQUE, maxlen=HISTORIQUE)
    previous, last_in_beat = None, []
    etat = EtatRythme()
    current_tempo = float(bpms[0])
    for cell in range(n_cells):
        beat = cell * PAS
        bar = int(beat) // 4
        chord = int(chords[bar])
        audible_chord = chord if quality[bar] >= .55 else None
        bar_start = cell % (4 * SUBDIVISIONS) == 0
        # Contexte CAUSAL : le nouveau tempo n'est pas encore connu du réseau.
        x = contexte_temporel(audible_chord, previous, history, beat, current_tempo, etat) if cell in selected else None
        labels = {nom: -1 for nom in TACHES}
        labels["progressions"] = np.zeros(len(PROGRESSIONS), np.float32)
        if cell == 0:
            labels["tempo_initial"] = classe_proche(TEMPOS, bpms[0])
        if bar_start:
            if bar > 0:
                ratio = float(bpms[bar] / current_tempo)
                labels["tempos"] = classe_proche(RATIOS_TEMPO, ratio)
                etat.variations.append(ratio)
            current_tempo = float(bpms[bar])
            if bar % 4 == 0 and bar + 4 <= n_bars:
                segment = chords[bar:bar + 4]
                matches = np.array([sum(a == b for a, b in zip(segment, p)) for p in PROGRESSIONS_IDS])
                if quality[bar:bar + 4].min() >= .55 and matches.max() >= 3:
                    labels["progressions"] = softmax(2.0 * matches).astype(np.float32)
        group = clusters.get(cell, [])
        labels["rythme"] = int(bool(group))
        if group:
            # Une attaque représentative par double croche ; nuances et voix
            # indépendantes restent simplifiées dans ce petit modèle CPU.
            duration = float(np.median([n.end - n.start for n in group]))
            labels["durees"] = classe_proche(DUREES, duration)
            etat.durees.append(float(DUREES[labels["durees"]]))
            etat.fin_note = max(etat.fin_note, beat + float(DUREES[labels["durees"]]))
            action = action_observee(chord, group) if quality[bar] >= .55 else None
            if action is not None:
                labels["notes"] = action
                last_in_beat = notes_action(chord, action)
        etat.attaques.append(labels["rythme"])
        if cell in selected:
            xs.append(x)
            for nom in TACHES:
                ys[nom].append(labels[nom])
        if cell % SUBDIVISIONS == SUBDIVISIONS - 1:
            history.append(last_in_beat)
            last_in_beat = []
            if int(beat) % 4 == 3:
                previous = chord if quality[bar] >= .55 else None
    # float16 en stockage seulement ; calculs et gradients restent float32.
    x = np.asarray(xs, np.float16).reshape((-1, NB_FEATURES_TEMPS))
    y = {nom: np.asarray(vals, np.float32 if nom == "progressions" else np.int16)
         for nom, vals in ys.items()}
    stats = {"source_tempo": morceau.source_tempo, "confiance": morceau.confiance,
             "bpm_min": float(bpms.min()), "bpm_max": float(bpms.max()),
             "bpm_initial": float(bpms[0]), "transposition": shift,
             "cases_totales": n_cells, "attaques_totales": len(clusters)}
    return Exemples(x, y, stats)


def trouver_midis(folder, max_files, seed):
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise ValueError(f"Dossier introuvable : {folder}")
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (".mid", ".midi"))
    if not paths:
        raise ValueError(f"Aucun .mid ou .midi dans {folder}")
    np.random.default_rng(seed).shuffle(paths)
    unique, hashes = [], set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in hashes:
            hashes.add(digest)
            unique.append(path)
            if max_files and len(unique) == max_files:
                break
    return unique


def separer_fichiers(paths, metadata):
    # Garder les rôles des fichiers déjà connus, même après déplacement du dossier.
    train_names = {Path(p).name for p in metadata.get("train_files", [])}
    val_names = {Path(p).name for p in metadata.get("validation_files", [])}
    if train_names & val_names:
        raise ValueError("Un même nom de MIDI figure dans train et validation du modèle.")
    train, val, unknown = [], [], []
    for path in paths:
        (val if path.name in val_names else train if path.name in train_names else unknown).append(path)
    n_val = max(1, round(.2 * len(unknown))) if len(unknown) >= 2 else 0
    val += unknown[:n_val]
    train += unknown[n_val:]
    if not train:
        raise ValueError("Aucun fichier disponible pour apprendre ; élargir --dataset/--max-files.")
    return train, val


def collecter(paths, args, label, seed):
    chunks, targets = [], {nom: [] for nom in TACHES}
    valid, stats = [], []
    for i, path in enumerate(paths):
        try:
            data = extraire_exemples(path, not args.sans_transposition, args.max_examples_per_file,
                                    seed + i, args.source_tempo)
            chunks.append(data.x)
            for nom in TACHES:
                targets[nom].append(data.y[nom])
            valid.append(str(path.resolve()))
            stats.append(data.stats)
            counts = "/".join(str(len(data.indices(n))) for n in ("notes", "rythme", "durees", "tempos"))
            s = data.stats
            print(f"  [{label} {i+1}/{len(paths)}] {path.name} | exemples notes/rythme/durées/tempo {counts} | "
                  f"BPM {s['bpm_min']:.0f}-{s['bpm_max']:.0f} ({s['source_tempo']})", flush=True)
        except (ValueError, OSError, EOFError, KeyError) as exc:
            print(f"  [{label}] Ignoré : {path.name} ({exc})", flush=True)
    x = np.concatenate(chunks) if chunks else np.empty((0, NB_FEATURES_TEMPS), np.float16)
    y = {nom: np.concatenate(parts) if parts else np.empty((0, 5), np.float32) if nom == "progressions"
         else np.empty(0, np.int16) for nom, parts in targets.items()}
    return Exemples(x, y, {"fichiers": stats}), valid


def cibles(data, nom, ids):
    values = data.y[nom][ids]
    if nom == "progressions":
        return values.astype(np.float32)
    target = np.zeros((len(ids), CLASSES[nom]), np.float32)
    target[np.arange(len(ids)), values] = 1
    return target


def evaluer(net, nom, data, indices):
    if not len(indices):
        return None
    total = 0.0
    for start in range(0, len(indices), 1024):
        ids = indices[start:start + 1024]
        p, _ = net.forward(entree_reseau(nom, data.x[ids]))
        total -= float(np.sum(cibles(data, nom, ids) * np.log(np.maximum(p, 1e-15))))
    return total / len(indices)


def entrainer(model, args):
    require_mido()
    seed = 42 if args.seed is None else args.seed
    paths = trouver_midis(args.dataset, args.max_files, seed)
    train_paths, val_paths = separer_fichiers(paths, model.metadata)
    print(f"\n{len(paths)} MIDI uniques : {len(train_paths)} train / {len(val_paths)} validation.")
    print("Lecture des attaques, des durées, des silences et des tempos ; aucun son n'est joué.")
    train, train_files = collecter(train_paths, args, "train", seed)
    val, val_files = collecter(val_paths, args, "val", seed + 10000)
    if not len(train.x):
        raise ValueError("Aucun exemple d'entraînement exploitable.")
    itrain = {nom: train.indices(nom) for nom in TACHES}
    ival = {nom: val.indices(nom) for nom in TACHES}
    for nom in TACHES:
        print(f"  {nom:14s} : {len(itrain[nom])} exemples train / {len(ival[nom])} validation")
        if len(itrain[nom]) and nom != "progressions":
            model.vues[nom] += np.bincount(train.y[nom][itrain[nom]], minlength=CLASSES[nom])
    if not len(ival["rythme"]):
        print("Validation indisponible : suivi sur les exemples d'entraînement, sans mesure de généralisation.")
    evaluations = {nom: (val, ival[nom]) if len(ival[nom]) else (train, itrain[nom]) for nom in TACHES}
    starts = dict(model.metadata["epochs_par_reseau"])
    best = {nom: evaluer(net, nom, *evaluations[nom]) for nom, net in model.reseaux().items()}
    states = {nom: capturer(net) for nom, net in model.reseaux().items()}
    model.metadata.update(train_files=train_files, validation_files=val_files,
        data_reduction="sixteenth-note grid, separate gates, per-bar tempo; shared float16 inputs",
        seed=seed, source_tempo=args.source_tempo, tempo_algorithm=ALGORITHME_TEMPO,
        transposition=not args.sans_transposition, anchor=args.anchor, anchor_temporel=args.anchor_temporel,
        evaluation_par_reseau={nom: "validation" if len(ival[nom]) else "train" for nom in TACHES},
        tempo_sources=dict((source, sum(s["source_tempo"] == source for s in train.stats["fichiers"]))
                          for source in {s["source_tempo"] for s in train.stats["fichiers"]}))
    if args.model.exists():
        with np.load(args.model, allow_pickle=False) as old:
            old_format = json.loads(str(old["metadata"].item())).get("format")
        if old_format == 1:
            import shutil
            backup = args.model.with_name(args.model.stem + ".avant_rythme_tempo.npz")
            if not backup.exists():
                shutil.copy2(args.model, backup)
            print(f"Ancien modèle conservé : {backup}")
    model.save(args.model, states)
    rng, stale = np.random.default_rng(seed), 0
    try:
        for epoch in range(1, args.epochs + 1):
            improved, texts = False, []
            for nom, net in model.reseaux().items():
                ids = itrain[nom]
                if not len(ids):
                    continue
                order = rng.permutation(ids)
                anchor = args.anchor if nom in ("notes", "progressions") else args.anchor_temporel
                for start in range(0, len(order), args.batch_size):
                    batch = order[start:start + args.batch_size]
                    _, grads = net.loss_and_gradients(entree_reseau(nom, train.x[batch]), cibles(train, nom, batch), anchor)
                    net.update(grads, args.learning_rate)
                score = evaluer(net, nom, *evaluations[nom])
                texts.append(f"{nom} {score:.3f}")
                if best[nom] is None or score < best[nom] - 1e-5:
                    best[nom], states[nom], improved = score, capturer(net), True
                    model.metadata["epochs_par_reseau"][nom] = starts[nom] + epoch
            model.metadata["trained_epochs"] = max(model.metadata["epochs_par_reseau"].values())
            model.metadata["meilleures_nll"] = best
            model.save(args.model, states)
            print(f"Époque {epoch}/{args.epochs} | NLL " + " | ".join(texts), flush=True)
            stale = 0 if improved else stale + 1
            if stale >= args.patience:
                print("Arrêt anticipé : aucun réseau ne s'améliore.")
                break
    except KeyboardInterrupt:
        print("\nInterruption : les meilleurs poids déjà sauvegardés sont conservés.")
    finally:
        for nom, net in model.reseaux().items():
            restaurer(net, states[nom])
    print(f"Modèle sauvegardé : {args.model}")
    print("La NLL évalue la prédiction des exemples simplifiés ; la qualité musicale reste à écouter.")


# ------------------------- Composer puis écouter -------------------------
@dataclass
class Evenement:
    beat: float
    accord: int
    notes: list[int]
    duree: float = SUSTAIN


@dataclass
class Composition:
    evenements: list[Evenement]
    tempos: list[tuple[float, float]]
    fin: float
    progression: int
    probabilites_progressions: np.ndarray


def probabilites_observees(model, nom, x, influence=1):
    net = getattr(model, nom)
    p = net.probabilities(np.asarray(x)[None, :], influence)[0]
    if nom not in ("notes", "progressions") and model.vues[nom].sum():
        p *= model.vues[nom] > 0
        p /= p.sum()
    return p


def composer(model, cycles=4, tempo=None, influence=.35, seed=None, rythme_fixe=False, tempo_fixe=False):
    rng = np.random.default_rng(seed)
    history = deque([[]] * HISTORIQUE, maxlen=HISTORIQUE)
    etat = EtatRythme()
    epochs = model.metadata["epochs_par_reseau"]
    if tempo is None:
        if epochs["tempo_initial"]:
            x = contexte_temporel(None, None, [], 0, 100, etat)
            tempo = float(TEMPOS[rng.choice(len(TEMPOS), p=probabilites_observees(model, "tempo_initial", x))])
        else:
            tempo = float(np.random.default_rng(seed).integers(TEMPO_MIN, TEMPO_MAX + 1))
    current_tempo = float(tempo)
    prog_p = model.progressions.probabilities(contexte(None, None, history, 0, tempo), influence)[0]
    index = int(rng.choice(len(PROGRESSIONS), p=prog_p))
    progression = PROGRESSIONS_IDS[index]
    events, changes = [], [(0.0, current_tempo)]
    previous, last_in_beat = None, []
    total_beats = cycles * 16
    learned_rhythm = bool(epochs["rythme"]) and not rythme_fixe
    learned_durations = bool(epochs["durees"]) and not rythme_fixe
    for cell in range(total_beats * SUBDIVISIONS):
        beat = cell * PAS
        bar = int(beat) // 4
        chord = progression[bar % 4]
        x = contexte_temporel(chord, previous, history, beat, current_tempo, etat)
        if cell and cell % (4 * SUBDIVISIONS) == 0 and not tempo_fixe and epochs["tempos"]:
            action = int(rng.choice(len(RATIOS_TEMPO), p=probabilites_observees(model, "tempos", x)))
            ratio = float(RATIOS_TEMPO[action])
            nouveau = float(np.clip(current_tempo * ratio, TEMPOS[0], TEMPOS[-1]))
            etat.variations.append(nouveau / current_tempo)
            if abs(nouveau - current_tempo) > .01:
                changes.append((beat, nouveau))
            current_tempo = nouveau
        if learned_rhythm:
            attack = bool(rng.choice(2, p=probabilites_observees(model, "rythme", x)))
        else:
            attack = cell % SUBDIVISIONS == 0
        pitches, duration = [], PAS
        if attack:
            note_x = x[:NB_FEATURES]
            action = int(rng.choice(len(ACTIONS), p=model.notes.probabilities(note_x, influence)[0]))
            pitches = notes_action(chord, action)
            if learned_durations:
                duration = float(DUREES[rng.choice(len(DUREES), p=probabilites_observees(model, "durees", x))])
            else:
                duration = SUSTAIN
            etat.durees.append(duration)
            etat.fin_note = max(etat.fin_note, beat + duration)
            last_in_beat = pitches
        events.append(Evenement(beat, chord, pitches, duration))
        etat.attaques.append(int(attack))
        if cell % SUBDIVISIONS == SUBDIVISIONS - 1:
            history.append(last_in_beat)
            last_in_beat = []
            if int(beat) % 4 == 3:
                previous = chord
    fin = max(total_beats, max((e.beat + e.duree for e in events if e.notes), default=0))
    return Composition(events, changes, fin, index, prog_p)


def intervalles_notes(composition):
    """Une même touche rejouée ferme sa note précédente, en MIDI ET en audio."""
    intervals, last = [], {}
    for e in composition.evenements:
        for pitch in e.notes:
            if pitch in last:
                intervals[last[pitch]][1] = min(intervals[last[pitch]][1], e.beat)
            last[pitch] = len(intervals)
            intervals.append([e.beat, e.beat + e.duree, pitch])
    return intervals


def exporter_midi(composition, output):
    mido = require_mido()
    resolution = 960
    midi = mido.MidiFile(ticks_per_beat=resolution)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.Message("program_change", program=0, channel=0, time=0))
    messages = []
    for beat, bpm in composition.tempos:
        messages.append((round(beat * resolution), -1, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
    for start, end, pitch in intervalles_notes(composition):
        messages.append((round(start * resolution), 1, pitch, mido.Message("note_on", note=pitch, velocity=72)))
        messages.append((round(end * resolution), 0, pitch, mido.Message("note_off", note=pitch, velocity=0)))
    last = 0
    for tick, _, _, message in sorted(messages, key=lambda item: item[:3]):
        track.append(message.copy(time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=max(0, round(composition.fin * resolution) - last)))
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output)
    return output


def signal_note(note_midi, duree, release=RESONANCE_SECONDES):
    """Séparer la durée de touche apprise de la décroissance acoustique du son."""
    if not np.isfinite(duree) or not np.isfinite(release) or release < 0:
        raise ValueError("Durée ou résonance audio invalide.")
    if duree <= 0:
        return np.zeros(1, np.float32)
    frequence = 440 * 2 ** ((note_midi - 69) / 12)
    t = np.arange(max(1, round(FREQUENCE_ECHANTILLONNAGE * (duree + release)))) / FREQUENCE_ECHANTILLONNAGE
    # Les harmoniques aiguës s'amortissent plus vite : le son garde du corps
    # après l'attaque, sans transformer la résonance en un bourdonnement fixe.
    decay = float(np.clip(1.5 * (220 / frequence) ** .12, .8, 2.2))
    signal = sum(amplitude * np.sin(2 * np.pi * frequence * harmonic * t)
                 * np.exp(-t / (decay / np.sqrt(harmonic)))
                 for harmonic, amplitude in ((1, 1), (2, .45), (3, .20), (4, .08)))
    envelope = np.minimum(t / .012, 1)
    if release > 0:
        envelope *= np.exp(-4.6 * np.maximum(0, t - duree) / release)
    # Fin à zéro, même lorsque --resonance 0 est demandé.
    fade = min(.012, (duree + release) / 3)
    envelope *= np.clip((duree + release - t) / max(fade, 1e-6), 0, 1)
    signal *= envelope
    signal /= max(float(np.max(np.abs(signal))), 1e-12)
    return (signal * .12).astype(np.float32)


def niveau_sortie(audio, volume=1.0):
    """Un gain global : préserver les nuances et les silences du morceau.

    La mesure utilise les fenêtres actives de 100 ms, pour que de longues
    pauses ne conduisent pas à amplifier exagérément les notes restantes.
    Une limitation progressive agit seulement sur les plus grosses crêtes.
    """
    if not np.isfinite(volume) or not 0 <= volume <= 2:
        raise ValueError("Le volume doit être compris entre 0 et 2.")
    audio = np.asarray(audio, np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if volume == 0 or peak < 1e-8:
        return np.zeros_like(audio)
    window = FREQUENCE_ECHANTILLONNAGE // 10
    squared = np.square(audio, dtype=np.float64)
    starts = np.arange(0, len(audio), window)
    sizes = np.minimum(window, len(audio) - starts)
    energy = np.add.reduceat(squared, starts) / sizes
    active = energy > max(float(energy.max()) * .01, 1e-12)
    if not np.any(active):
        return audio * volume
    rms = math.sqrt(float(np.sum(energy[active] * sizes[active]) / np.sum(sizes[active])))
    gain = min(20.0, .16 / max(rms, 1e-8)) * volume
    y = audio * gain
    # Courbe continue, pente 1 au seuil ; aucune coupure brutale à +/-32767.
    magnitude = np.abs(y)
    high = magnitude > .8
    if np.any(high):
        y[high] = np.sign(y[high]) * (.8 + .15 * (-np.expm1(-(magnitude[high] - .8) / .15)))
    return y


def rendre_audio(composition, resonance=RESONANCE_SECONDES, volume=1.0):
    if not np.isfinite(resonance) or not 0 <= resonance <= 5:
        raise ValueError("La résonance doit être comprise entre 0 et 5 secondes.")
    carte = CarteTempo(composition.tempos)
    length = int(math.ceil((carte.secondes(composition.fin) + resonance + .02) * FREQUENCE_ECHANTILLONNAGE))
    if length > 30 * 60 * FREQUENCE_ECHANTILLONNAGE:
        raise ValueError("Rendu audio limité à 30 minutes ; exporter le MIDI ou réduire --cycles.")
    audio = np.zeros(length, np.float32)
    cache = {}
    for start, end, pitch in intervalles_notes(composition):
        begin = carte.secondes(start)
        duration = carte.secondes(end) - begin
        key = (pitch, round(duration * FREQUENCE_ECHANTILLONNAGE))
        if key not in cache:
            if len(cache) >= 64:
                del cache[next(iter(cache))]
            cache[key] = signal_note(pitch, duration, release=resonance)
        sound = cache[key]
        index = int(round(begin * FREQUENCE_ECHANTILLONNAGE))
        n = min(len(sound), len(audio) - index)
        if n > 0:
            audio[index:index + n] += sound[:n]
    audio = niveau_sortie(audio, volume)
    return np.ascontiguousarray((audio * 32767).astype(np.int16))


def exporter_wav(audio, path):
    import wave
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(FREQUENCE_ECHANTILLONNAGE)
        f.writeframes(audio.astype("<i2", copy=False).tobytes())
    return path


def decrire_evenement(event, carte):
    bpm = carte.bpms[max(0, np.searchsorted(carte.beats, event.beat, side="right") - 1)]
    text = " + ".join(nom_note(n) for n in event.notes) if event.notes else "aucune nouvelle attaque"
    duration = f" | durée {event.duree:g} temps" if event.notes else ""
    return f"{event.beat:6.2f} temps | {NOMS_ACCORDS[event.accord]} | {text}{duration} | {bpm:.1f} BPM"


def jouer(composition, audio=None):
    try:
        import pygame
    except ImportError:
        print("Audio indisponible : python3 -m pip install pygame. Le MIDI reste disponible.")
        return
    audio = rendre_audio(composition) if audio is None else audio
    carte = CarteTempo(composition.tempos)
    try:
        pygame.mixer.init(frequency=FREQUENCE_ECHANTILLONNAGE, size=-16, channels=1, buffer=1024)
        sound = pygame.sndarray.make_sound(audio)
        sound.play()
        origin = time.perf_counter()
        for event in composition.evenements:
            if not event.notes and int(round(event.beat * SUBDIVISIONS)) % SUBDIVISIONS:
                continue
            target = origin + carte.secondes(event.beat)
            time.sleep(max(0, target - time.perf_counter()))
            print(decrire_evenement(event, carte), flush=True)
        time.sleep(max(0, origin + len(audio) / FREQUENCE_ECHANTILLONNAGE - time.perf_counter()))
    except pygame.error as exc:
        print(f"Audio indisponible ({exc}). Le fichier MIDI reste disponible.")
    finally:
        pygame.mixer.quit()


def parser_arguments():
    parser = argparse.ArgumentParser(description="Piano IA V4 : notes, rythme, durées et tempo appris sur MIDI.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Continuer l'apprentissage et sauvegarder.")
    mode.add_argument("--generate", action="store_true", help="Composer avec le modèle, sans réentraînement (défaut).")
    parser.add_argument("--dataset", type=Path, default=SCRIPT_DIR / "maestro_midi")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "modele_piano_regles.npz")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "composition_IA.mid")
    parser.add_argument("--wav", type=Path, help="Exporter aussi le son synthétisé dans un WAV.")
    parser.add_argument("--resonance", type=float, default=RESONANCE_SECONDES,
                        help="Résonance après relâchement, en secondes (0-5 ; défaut 1.2).")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="Volume audio/WAV : 0 = muet, 1 = normal, 2 = renforcé.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=0, help="Maximum de MIDI ; 0 = tous (défaut).")
    parser.add_argument("--max-examples-per-file", type=int, default=512, help="Cases temporelles par MIDI (mémoire CPU).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--anchor", type=float, default=.20)
    parser.add_argument("--anchor-temporel", type=float, default=.02)
    parser.add_argument("--influence", type=float, default=.35, help="Influence du réseau sur les notes/progressions.")
    parser.add_argument("--cycles", type=int, default=NOMBRE_CYCLES)
    parser.add_argument("--tempo", type=float, help="Tempo INITIAL imposé (40-200 BPM) ; sinon il est appris.")
    parser.add_argument("--tempo-fixe", action="store_true", help="Garder le tempo initial pendant tout le morceau.")
    parser.add_argument("--rythme-fixe", action="store_true", help="Comparer à l'ancien rythme : une attaque par temps.")
    parser.add_argument("--source-tempo", choices=("auto", "midi", "estime"), default="auto",
        help="auto : changements MIDI explicites, sinon estimation de la pulsation depuis les attaques.")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sans-son", "--no-audio", dest="sans_son", action="store_true")
    parser.add_argument("--sans-transposition", action="store_true")
    parser.add_argument("--reinitialiser", action="store_true", help="Repartir de réseaux neufs.")
    args = parser.parse_args()
    for name in ("epochs", "max_examples_per_file", "batch_size", "cycles", "patience"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} doit être strictement positif.")
    if args.max_files < 0:
        parser.error("--max-files doit être positif ou nul.")
    if not np.isfinite(args.resonance) or not 0 <= args.resonance <= 5:
        parser.error("--resonance doit être entre 0 et 5 secondes.")
    if not np.isfinite(args.volume) or not 0 <= args.volume <= 2:
        parser.error("--volume doit être entre 0 et 2.")
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in (args.anchor, args.anchor_temporel, args.influence)):
        parser.error("Les influences et ancrages doivent être compris entre 0 et 1.")
    if not np.isfinite(args.learning_rate) or not 0 < args.learning_rate <= .1:
        parser.error("--learning-rate doit être entre 0 (exclu) et 0.1.")
    if args.tempo is not None and (not np.isfinite(args.tempo) or not TEMPOS[0] <= args.tempo <= TEMPOS[-1]):
        parser.error("--tempo doit être entre 40 et 200 BPM.")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed doit être positif ou nul.")
    args.model = args.model.expanduser().resolve()
    return args


def main():
    args = parser_arguments()
    model = PianoIA.load(args.model) if args.model.exists() and not args.reinitialiser else PianoIA()
    print("\n========== PIANO IA V4 : NOTES + RYTHME + TEMPO ==========", flush=True)
    epochs = model.metadata["epochs_par_reseau"]
    if args.model.exists() and not args.reinitialiser:
        print(f"Modèle chargé : {args.model}")
        print("Époques retenues : " + " | ".join(f"{k}={v}" for k, v in epochs.items()))
    if args.train:
        entrainer(model, args)
        return
    if not epochs["rythme"] or not epochs["tempo_initial"]:
        print("Rythme/tempo pas encore appris : entraîner cette nouvelle V4 pour les activer.")
    comp = composer(model, args.cycles, args.tempo, args.influence, args.seed, args.rythme_fixe, args.tempo_fixe)
    print(f"Tempo initial : {comp.tempos[0][1]:.1f} BPM | changements : {len(comp.tempos)-1} | influence notes : {args.influence:.0%}")
    print("Progression : " + " -> ".join(PROGRESSIONS[comp.progression]))
    print(f"Composition MIDI : {exporter_midi(comp, args.output)}")
    print(f"Rendu audio : résonance {args.resonance:g} s | volume {args.volume:.0%}")
    audio = rendre_audio(comp, args.resonance, args.volume) if args.wav or not args.sans_son else None
    if args.wav:
        print(f"Composition WAV : {exporter_wav(audio, args.wav)}")
    if args.sans_son:
        carte = CarteTempo(comp.tempos)
        for event in comp.evenements[:32]:
            print(decrire_evenement(event, carte))
    else:
        jouer(comp, audio)
    print("Morceau terminé.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nArrêt demandé.")
    except (ValueError, OSError, EOFError, KeyError) as exc:
        raise SystemExit(f"Erreur : {exc}") from exc

