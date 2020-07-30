# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Notizen
# %% [markdown]
# ## Vor 22.04.2020
# %% [markdown]
# ### Implementiert
# - IMP (aus dem originalen LT-Paper)
# - Pruning-Methoden
#     - Magnitude
#     - Delta-Bewegung high/low ("Gradient")
# - Neuronen-Pruning
#     - Lösche alle Neuronen bei denen die Inputgewichte alle 0 sind (d.h. die Zeilen der Gewichtsmatrizen)
#     - Vorteil: Trainingszeit wird kürzer da Netzwerk kleiner wird
#     - Nachteil: Deutlich höherer Trainingsaufwand
#     - **Zu Zeigen**: Vorteil von der Pruning-Methode im Vergleich zu einem Netz das direkt in der Größe initialisiert wurde
# - Pruning-Consitency: Prüfe ob bei gleicher Initialisierung die selben Gewichte "genullt" werden
# 
# %% [markdown]
# ### Beobachtungen
# %% [markdown]
# - IMP
#     - Ergebnisse konnten reproduziert werden
#     - Delta high scheint besser zu funktionieren als Magnitude
#     - Muster mit Delta deutlich besser erkennbar und deutlich mehr Zeilenvektoren == 0
#         - Beziehung zwischen Verhalten und Methode??
#         - Prüfe weitere Methoden!!
# - Neuronen-Pruning funktioniert prinzipiell. Noch nicht gezeigt dass es besser als direkte Initialisierung ist
# - Es wird nicht konsistent gepruned!
#     - Frage: Warum?? Gleiche Daten, Reihenfolge, Initialisierung und trotzdem unterschiedliches Ergebnis...
# %% [markdown]
# # TODO
# %% [markdown]
# - Weitere Pruning-Methoden (siehe "Deconstructing LT"-Paper)
# - Vergleiche Optimierer (SGD, Adam, RMSProp)
#     - Performance allgemein
#     - Verhalten bei vorinitalisierten Gewichten
# - Einfluss/Veränderung des Bias
# - Untersuche Beziehung zwischen Gewichtsinitialisierung und "Lottery Ticket"
#     - Welchen Einfluss hat die Initialisierung bei LTs?
# - Weitere Datensätze
# - Warum werden Gewichte genullt, die sich weniger bewegen?
# 
# - Bias bei gestrichenen Neuronen? Größe? Verähltnis Bias zu incoming Gewichten
# - Training Mask Backprop / other optimization techniques
# - Weight freezing instead of masking

# %%



