# Présentation

## Organisation du dossier

- **Projet de recherche détaillé :** fichier pdf contenant le compte-rendu complet de la recherche menée en autonomie (sujet, cadre théorique, méthodologie, problématique, résultats, discussions, conclusion)
- **Notebook détaillé :** fichier Jupyter Notebook contenant le code détaillé et commenté du projet
- **Représentations graphiques :** ensemble de représentations graphiques produites via Python et à l'aide du site Data Wrapper

## Résumé synthétique de la recherche

- **Question de recherche :** la rhétorique de guerre devient-elle plus importante dans les discours présidentiels français depuis le 11 septembre 2001 ?

- **Hypothèses :** la guerre est une thématique importante comparée aux autres sur 25 ans, l'importance de la rhétorique de guerre est croissante ces dernières années, la rhétorique de guerre s'étend de plus en plus à d'autres sujets.

- **Données :** Corpus issu de Vie-Publique.fr, 5 554 discours présidentiels français depuis le 11 septembre 2001, aspirés via l'application Web Scraper au format CSV. Chaque individu représente un discours. Le corpus est trop volumineux pour être partagé (85mo au format CSV, 26mo au format ZIP)

- **Méthodologie :** web scraping, nettoyage et normalisation, fréquence pour 10 000 mots, TF-IDF, cooccurrence, analyse statistique (anova, coefficients de corrélation, p-value).

- **Résultats :** La guerre est une thématique importante comparée à d'autres sujets, la rhétorique de guerre est croisante de manière positive et significative ces dernières années, la rhétorique de guerre s'étend de plus en plus à d'autres thématiques (la relation est positive et significative).

- **Discussions :** l'analyse de réseau semble être la piste la plus pertinente pour mesurer l'évolution globale des discours présidentiels français, nécessitant un encadrement par un Master spécialisé
