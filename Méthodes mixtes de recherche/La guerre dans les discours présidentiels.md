# **La guerre dans les discours présidentiels français**

##### Importation des libraries 


```python
# Pour la gestion du dataframe
import numpy as np
import pandas as pd

# Pour le TALN (NLP)
import spacy
import nltk

# Pour la gestion des dates en français
import locale
locale.setlocale(locale.LC_ALL, "fr_FR")

# Pour les représentations graphiques simples
import matplotlib.pyplot as plt

# Pour le calcul du TF-IDF
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
```


```python
# Spécification concernant spaCy : utilisation de la langue française
nlp = spacy.load("fr_core_news_sm")
```

## **1-) Introduction**
### (Importation de la base de donnée)

**Importation des libraries et chargement de la bibliothèque française de spaCy :**


```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

Importation de la base de donnée web scrappée sur vie-publique.fr : les discours présidentiels français depuis le 11 septembre 2001 (5554 documents). Lien vers la source des discours : https://www.vie-publique.fr/discours/recherche?search_api_fulltext_discours=&sort_by=field_date_prononciation_discour&field_intervenant_title=&field_intervenant_qualite=&field_date_prononciation_discour_interval%5Bmin%5D=2001-09-10&field_date_prononciation_discour_interval%5Bmax%5D=2026-01-01&field_type_emetteur%5B9340%5D=9340


```python
df = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/analyse-discours-presidentiels.csv")
```


```python
# Vérification du dataframe
print(df.head())
```

      web-scraper-order                              web-scraper-start-url  \
    0      1768752907-1  https://www.vie-publique.fr/discours/recherche...   
    1      1768752909-2  https://www.vie-publique.fr/discours/recherche...   
    2      1768752912-3  https://www.vie-publique.fr/discours/recherche...   
    3      1768752915-4  https://www.vie-publique.fr/discours/recherche...   
    4      1768752921-5  https://www.vie-publique.fr/discours/recherche...   
    
                                    discours-individuels  \
    0  Discours de M. Jacques Chirac, Président de la...   
    1  Discours de M. Jacques Chirac, Président de la...   
    2  Intervention de M. Jacques Chirac, Président d...   
    3  Intervention télévisée de M. Jacques Chirac, P...   
    4  Lettre de M. Jacques Chirac, Président de la R...   
    
                               discours-individuels-href  \
    0  https://www.vie-publique.fr/discours/195774-di...   
    1  https://www.vie-publique.fr/discours/194083-di...   
    2  https://www.vie-publique.fr/discours/196432-in...   
    3  https://www.vie-publique.fr/discours/194790-in...   
    4  https://www.vie-publique.fr/discours/194906-le...   
    
                                                   titre     thématique  \
    0  Discours de M. Jacques Chirac, Président de la...            NaN   
    1  Discours de M. Jacques Chirac, Président de la...       Économie   
    2  Intervention de M. Jacques Chirac, Président d...  International   
    3  Intervention télévisée de M. Jacques Chirac, P...        Société   
    4  Lettre de M. Jacques Chirac, Président de la R...  International   
    
                 détails     intervenant  \
    0  10 septembre 2001  Jacques Chirac   
    1  11 septembre 2001  Jacques Chirac   
    2  11 septembre 2001  Jacques Chirac   
    3  11 septembre 2001  Jacques Chirac   
    4  11 septembre 2001  Jacques Chirac   
    
                                                   corps  \
    0  Monsieur le Maire de Saint-Brieuc, mon cher am...   
    1  Monsieur le président LEMETAYER,\nMonsieur le ...   
    2  Monsieur le Président,\nMesdames, messieurs,\n...   
    3  Mes chers compatriotes,\nLes attentats qui ont...   
    4  Monsieur le Président,\nC'est avec une immense...   
    
                       mots-cles  \
    0       Situation économique   
    1             Vie économique   
    2  Relations internationales   
    3                   Sécurité   
    4  Relations internationales   
    
                                                   pages  
    0  https://www.vie-publique.fr/discours/recherche...  
    1  https://www.vie-publique.fr/discours/recherche...  
    2  https://www.vie-publique.fr/discours/recherche...  
    3  https://www.vie-publique.fr/discours/recherche...  
    4  https://www.vie-publique.fr/discours/recherche...  
    

## **2-) Nettoyage de la base de donnée**
### (Suppression des NA & normalisation du texte : tokenisation, suppression des stop words, lemmatisation)

#### **Gestion des NA**


```python
# Vérification des NA pour les corps de texte
print(f"Nombre de valeurs manquantes : {df["corps"].isna().sum()}")
print(f"Type de données : {df["corps"].dtype}")
```

    Nombre de valeurs manquantes : 0
    Type de données : object
    


```python
# Suppression des NA pour les corps de texte
df["corps"] = df["corps"].fillna("")
df["corps"] = df["corps"].astype(str)
```


```python
# Vérification des NA pour les intervenants & détails
print(f"Nombre de valeurs manquantes : {df["intervenant"].isna().sum()}")
print(f"Type de données : {df["intervenant"].dtype}")

# Détails
print(f"Nombre de valeurs manquantes : {df["détails"].isna().sum()}")
print(f"Type de données : {df["détails"].dtype}")
```

    Nombre de valeurs manquantes : 0
    Type de données : object
    Nombre de valeurs manquantes : 0
    Type de données : object
    


```python
# Suppression des NA pour les intervenant & date
df["intervenant"] = df["intervenant"].fillna("")
df["intervenant"] = df["intervenant"].astype(str)

# Détails
df["détails"] = df["détails"].fillna("")
```


```python
# Vérification des intervenants & détails restants
df["intervenant"].value_counts()
df["détails"].value_counts()
```




    détails
                        155
    22 février 2016      12
    25 février 2016       8
    16 novembre 2014      8
    2 septembre 2002      8
                       ... 
    7 octobre 2001        1
    14 août 2007          1
    11 octobre 2001       1
    12 octobre 2001       1
    4 novembre 2021       1
    Name: count, Length: 3570, dtype: int64




```python
présidents = ["François Hollande", "Jacques Chirac", "Emmanuel Macron", "Nicolas Sarkozy"]
```


```python
df = df[df["intervenant"].isin(présidents)]
```

#### **Nettoyage de la date**


```python
df["détails"].head()
type(df["détails"])
```




    pandas.core.series.Series




```python
df["date"] = pd.to_datetime(df["détails"], format="%d %B %Y", dayfirst=True)
```


```python
df["date"].head()
```




    0   2001-09-10
    1   2001-09-11
    2   2001-09-11
    3   2001-09-11
    4   2001-09-11
    Name: date, dtype: datetime64[ns]




```python
print(f"Nombre de valeurs manquantes : {df["date"].isna().sum()}")
print(f"Type de données : {df["date"].dtype}")
```

    Nombre de valeurs manquantes : 154
    Type de données : datetime64[ns]
    


```python
# Date
df["date"] = df["date"].dropna
```


```python
# Année
df["année"] = pd.to_datetime(df["date"], format="%Y")
```

#### **Normalisation (tokenisation, suppression des stop-words, lemmatisation)**


```python
# Importation des stop-words et ajout de la ponctuation à la liste des stop-words par défaut
stops = nlp.Defaults.stop_words
nlp.Defaults.stop_words.add(".")
nlp.Defaults.stop_words.add(",")
nlp.Defaults.stop_words.add("-")
nlp.Defaults.stop_words.add(";")
nlp.Defaults.stop_words.add(":")
nlp.Defaults.stop_words.add(":")
nlp.Defaults.stop_words.add("*")
```

Cellule suivante écrite à partir d'un thread issu de Stack Overflow : https://stackoverflow.com/questions/45605946/how-to-do-text-pre-processing-using-spacy


```python
# Définition d'un algorithme de normalisation du texte
def normalisation(token, lowercase, remove_stopwords):
    if lowercase:
        token = token.lower()
    token = nlp(token)
    lemmatisation = list()
    for word in token:
        lemme = word.lemma_.strip()
        if lemme:
            if not remove_stopwords or (remove_stopwords and lemme not in stops):
                lemmatisation.append(lemme)
    return " ".join(lemmatisation)
```

Application de l'algorithme de normalisation :


```python
df["corpus-nettoyé"] = df["corps"].apply(normalisation, lowercase=True, remove_stopwords=True)
```

Vérification que l'algorithme a bien fonctionné :


```python
print(df["corpus-nettoyé"].head())
```

    0    Monsieur maire saint-brieuc cher ami Monsieur ...
    1    Monsieur président lemetayer Monsieur ministre...
    2    Monsieur président madame monsieur venir ici g...
    3    cher compatriote attentat frapper aujourd'hui ...
    4    Monsieur président immense émotion france appr...
    Name: corpus-nettoyé, dtype: object
    


```python
discours_n5553 = df.loc[5430, "corpus-nettoyé"]
```


```python
print(discours_n5553)
```

    altesse sérénissime madame monsieur chef état gouvernementorgane collégial composer ministre ministre secrétaire état charger exécution loi direction politique national Monsieur président conseil Monsieur secrétaire général ocde madame monsieur ministre mesdamer ambassadeur monsieur ambassadeur madame monsieur parlementaire Monsieur maire nice madame monsieur grade qualité cher ami heureux sincèremer rendez-vous obliger savez accompagner année sujet mer océan cher retrouver fois cohorte visage ami formidable explorateur explorateur scientifique engager activiste militant travers continent porter cause inspir continuer inspirer croire venir voir mot instant harrison ford excuser pouvoir hier soir vouloir cas venir passer instant côtés.hier soir 21h tour eiffel illuminer honneur océan fois existence porter oeil monde message attention particulière sos océan incarner préparer conférence essentiel vouloir gratitude fondation oceano azul organisateur précieux moment sos océan remerciement président directeur général josé soare tiago pitta e cunha.l' urgence croire ici besoin plaider convaincre mobilisation fond commencer 2022 ensembl époque ministre antónio costa unoc 2 lisbonne beaucoup chose lancer ensuite one ocean summit brest occasion présidence français union européen continuité mouvement pouvoir relancer conclusion accord gouvernance haute-mer fameu bbnj revenir accompagner acteur transport maritime engagement trouver carburant durable annoncer grand aire marin protégées.depui mobilisation cesser cop15 cadre mondial biodiversité objectif commun valider protéger 30 pourcent terre mer fin 2022 annoncer engagement fort france faire interdire exploitation minière grand fonds marin connaissance scientifique nécessaire trésor biodiversité devoir prix protéger savoir engagement nécessité domaine maritime monde responsabilité incomber france matière recherche matière exigence pouvoir avancer.c' esprit décider costa rica accueillir ensemble nice 2025 sommet nation unir océan grand sommet nation unir précéder grand forum économie bleu organiser monaco infiniment monseigneur mobilisation aller rassembler entreprendre monde finance service économie bleu feuille route clair innover investir économie bleu durable protéger écosystème marin grand rôle régulateur climat océan celer fidèle engagement principauté aïeul monaco avant-poste combat connaissance protection biodiversité océan reconnaître organiser sommet.nous ensuite choisir donner rendez-vous quinzaine juin bord méditerranée nice cher christian magnifique métropole sommet enneigé mer méditerraner caractère unique monde nice savoir transformer dernier année méditerranée porteur enjeu ici beaucoup entrer engager 1 pourcent surface mer 10 pourcent biodiversité 25 pourcent passage transport maritime mondial 500 million riverain mer ébullition affecter changement climatique pollution.tenir sommet nice important conférence nation unie france accueill sol cop21 2015 unoc 3 devoir promesse restauration nature protection vivant rendez-vous important vouloir ici remercier remercier femme homme porter mobilisation fond rendez-vous servir cristalliser résultat accélérer chose travail scientifique exploratrice explorateur formidable aventurier ong entreprise fondation ici mobiliser évidemment réseau diplomatique.alors aller aller essayer faire nice ici lancer mobilisation accompagner soutenir so objectif moi.numéro 1 haute mer évoquer bbnj 64 pourcent surface océan global moitié surface planète océan pouvoir zone non-droit finir parachever négociation sein nation unie mars 2023 traité biodiversité au-delà juridiction national bbnj finaliser nation unie adopter 19 juin année 110 état signer traité majeur survie océan objectif nice 60 ratification luire permettre entrer vigueur jour 31 mars 2025 21 pays ratifier traité gros travail faire semaine venir devoir redoubler effort mobiliser largement calcul simple voir beaucoup état européen vouloir potentiel ami antonio beaucoup partenaire toucher conséquence devoir obtenir résultat appeler état côtier façade littorale ailleurs espace gestion durable haute mer ressource travers création aire marine protéger étude impact environnemental activité engager haute mer etc. point clé objectif sommet.le pêche durable mer nourrit devoir garder équilibre soutenabilité océan cœur souveraineté alimentaire loi français consacrer pêche intérêt majeur nation bien légitime beaucoup faire regagner souveraineté reconnaissance obliger constat clair mettre fin surpêche niveau mondial faire face risque majeur effondremer stock mettre péril vivre métier dépendre partie mer nourrir europe prendre part faire progrès majeur france participe 25 an 10 pourcent poisson débarquer issu stock gérer durablement 2023 poisson venir stock bon état reconstituable beaucoup travail faire devoir mobilisation pêcheur scientifique vouloir remercier collaboration fructueux indispensable pouvoir relever défi continuer agir pêcheur réduire impact pêche écosystème faire européen interdire dernier année pêche électrique utilisation engin fond écosystème marin vulnérable grand profondeur continuer obtenir résultat concret porter scientifique lien pêcheur améliorer justement résultat permettre mieux protéger fonds marin particulier aire marine protégées.pour faire face défi devoir avancer manière décisif sujet lutte contre pêche illégal illicite non déclarer entrer 10 20 pourcent production poisson issu évidemment inacceptable organisation mondial commerce faire sorte accord mettre fin subvention pêche illégal ratifier nice objectif égard penser accord cap port stat measure agreement appeler fao mobiliser état pratique pêche durable ensemble devoir développer pêche proximité non destructeur écosystème circuit court aquaculture durable penser transition flotte arriver ici pêche iam surpêche maîtrisée.le objectif atteindre cas continuer remobiliser 30 pourcent protection océan sujet processus multilatéral battre monter coalition savez objectif kunming montréal poser protéger 30 pourcent océan ici 2030 bien loin poser objectif mobiliser état membre aujourd'hui 8,5 pourcent appeler pays annoncer nice aire marin protéger fixon but atteindre 12 pourcent protection zone économique exclusif voire 15 pourcent ici unoc france faire part création aire marin protéger rappelon france ore aire marine protéger top 10 grand aire marin mondial voir fondation semaine président aire formidable faire açore continuer mobilisation feron renforcer niveau protection aire marine protéger existant mieux encadrer activité humain fort impact biodiversité transport maritime pêche pollution tellurique aire marine protéger berceau vie marine bénéfice économique évident permettre reconstitution stock poisson constituer barrière naturel contre hausse niveau mer renforcer potentiel séquestration océan rappelle puits carbone mondial.quatrièm objectif décarbonation transport maritime espérer nice pouvoir résultat vraiment tangible attendre impatience résultat négociation organisation maritime international savoir semaine aboutir mesure concret mettre œuvre ambitieux cibl total neutralité carbone 2050 sujet essentiel mobiliser année g7 biarritz transporteur maritime travail important faire devoir européen investir massivemer soutenir transition secteur carburant durable électrification port transformation flotte aller carburant durable transport hydrogène révolution secteur vouloir saluer travail faire cluster maritime français égard absolument remarquable monde présenter feuille route décarbonation grand armateur engager côté cop 29 accélérer décarbonation joue beaucoup omi prochain semaine souhaiter objectif cler unoc.cinquièm objectif lutte contre pollution plastique lucider dernier grand rendez-vous international déception beaucoup blocage beaucoup anti-jeu pouvoir faire sorte nice sou conduite ministre agnès pannier runacher homologue costa rica mobiliser état ambitieux matière lutte contre plastique savoir ravage instruire plusieur entrer responsabilité fond plastique arriver mer trop tard microplastique catastrophe biodiversité santé animal humain bien prévention transformation usage plastique filière amont falloir penser organiser beaucoup bouger loi dernier année europe france fois heurter beaucoup résistance raison faire résultat falloir véritablement réussir bouger chose plan international celer falloir poursuivre négociation traité plastique négociation début mois août genève rendez-vous important espérer unoc permettre accélérer celer reprendre travail.j' demander celer échelle méditerranée mobilisation particulière engagement état riverain méditerranée préfigurer pouvoir accord genèv état méditerranéen responsabilité particulière mobiliser mobiliser citoyen industriel collectivité faire sorte méditerranée mer polluer monde prévenir justement pollution plastique supprimer usage plastique usage unique progressivement sortir chaîne permettre recyclage préserver mer pollution chimique tellurique devoir cœur objectif nice penser sargasse rappeler ici territoire ultramarin cœur solution devoir présenter pays savoir particulier mexique prêt engager fortement sujet oublier bien sûr espèce marin menacer travailler lien iucn ong protéger.le objectif mobilisation financement travailler mobiliser financement philanthropique construire économie bleu durable continuité monaco nice ocde particulier secrétaire général rapport présenter matin éclaire bien économie océan 2050 secteur priver largement engager fondation avant-gardister ici présent solution préserver océan trouver mode financement initiative océan support crucial investissement tourisme bleu énergie régénération biotechnologie transport développement vélique port pêcherie durable compter cible travail devoir passer logique compensation logique régénération intégré valoriser service écosystème marin modèle économique égard travail préparer ocde devoir consacrer nice mobiliser financements.le objectif lutte contre changement climatique action local déployer brest 2022 précieux soutien plateform océan climat rassembler nombre ville mondial toucher phénomène élévation niveau mer soutien rénover plateform océan climat confier christian estrosi mission créer grand coalition ville région affecter sujet montée eau 7 juin maire nice accueillir 500 homologue maire gouverneur élu représentant 800 million citoyen toucher ici fin siècle dramatique conséquence réchauffement thermique mer fonte calotte glaciaire groenland antarctique mobilisation essentiel aller permettre déclencher coopération mécanisme adaptation financement adaptés.je mentionner cadre régional essentiel faire avancer cause océan union européen action local mobiliser union européen cher antonio présence hier sein engagement océan devoir boussole Monsieur président président commission mobiliser ensemble service pacte européen océan présenter permettre réaffirmer océan atout stratégique compétitif état membre devoir faire bon usage montrer chemin présenterez nice 9 juin président commission oublier travail faire dernier année cher pascal mobilisation chercheur financement européen jumeau numérique faire dernier année sorte fondation pacte aller venir parachever.ce pact bleu devoir pacte investissement protection océan investir carburant durable européen créer filière permettre remplacer énergie fossile investir électrification port recherche innovation donner communauté scientifique moyen service connaissance océan devoir pacte réciprocité pêcheur devoir trouver protection légitime face concurrence production respecter critère durabilité exemple matière lutte contre capture accidentel tortue pêche crevette prendre exemple beaucoup insister point oublier avis cler maintenir objectif durée réciprocité effort climatique durabilité effort climatique perdre adhésion producteur moment demande effort agriculteur industriel pêcheur devoir protéger assurer sorte créer standard aller progressivemer opposable tiers écrire passer décennie venir compte attitude quelques-uns reculer sou pression industrie réguler chose produire importer bilan carbon destruction biodiversité faire ailleurs critère réciprocité clé.enfin enjeu défense science nice savoir rappeler attachement travail scientifique action public fonder résultat science retrouver ici beaucoup ami organisme scientifique français international redi ici vivre période beaucoup grand puissance stopper financement organisme recherche public contester véracité résultat établir scientifiquemer aller dur faire financement sujet mot « biodiversité » interdire programme recherche « diversité poisson » interdire programme recherche train passer expliquer jour conseil présidentiel scientifique financement américain mot programme recherche aujourd'hui responsabilité européen ici véritablement maintenir financement recherche académique libre bien commun permettre établir scientifiquemer résultat permettre base établir changer pratique réussir faire faire mettre autour table exemple pêcheur scientifique vouloir pouvoir parachever aire marine protéger faire scientifique autour table scientifique autour table confronter chose établir scientifiquement nice devoir rappeler engagement -là devoir action concerner océan éclairer science.avec appui ifremer cnrs unoc 3 commencera 3 6 juin one ocean science congress organisme recherche évidemment mobiliser autour conférence réunir 2 000 scientifique représentant chercheur centaine pays croire adosser conférence nation unir remercier particulièrement bruno david excellent clair rapport exploitation grand fonds marin confort position exprimer an groupe grandissant état mobiliser sujet sujet activiste sou contrôle convaincre consolider science justifier continuer mobiliser prendre véritablement — choisir terme clair clair expression france — " pause précaution " " moratoire " exploitation grand fonds marin connaître suffisamment endosser code minier nécessité porter position assemblée général autorité international grand fonds marin kingston jamaïque mois unoc fond priorité priorité scientifique devoir connaître devoir comprendre devoir protéger ensuite pouvoir créer cadre pouvoir créer cadre comprendre exploré.c' combat inséparable connaissance scientifique soutien continuer apporter explorateur grand aventurier aller continuer lancer mission excellence grand fonds grand pôle savoir soutien unoc proposer rencontre mobiliser chef état gouvernementorgane collégial composer ministre ministre secrétaire état charger exécution loi direction politique national présent partenaire justement scientifiques.voilà madame monsieur objectif unoc exhaustif beaucoup action rendez-vous vouloir autour principal objectif pouvoir mobiliser énergie bien heure trouver solution fond intelligence collectif capacité coopérer volonté penser avenir humanité fuir mars pays arriver gérer kilomètre carré bande côtier loin ici convaincre neptune bon aventure permettre mobiliser chercheur planète b. aller faire continuer battr vivable vivable permettre ordre international respectueux humanisme éclairer apprendre vivre planète nature diversité faire durée apprendre connaître sommet unoc résolument oui cas neptun mars pouvoir faire pouvoir autre.en cas aujourd'hui lancer pari so océan choix hauteur savoir attendre france attendre côté costa rica remercier fois ensemble état membre nation unie océan cœur pouvoir compter engagement plein entier cas remercier infiniment travail mobilisation énergie faire compte nice au-delà pouvoir compter
    


```python
# Faire en sorte que le corpus soit un texte et non pas une liste de tokens
df["corpus-nettoyé"] = df["corpus-nettoyé"].astype(str)
```


```python
# Création d'une colonne au format tokens
df["corpus-tokens"] = df["corpus-nettoyé"].apply(lambda x: x.split())

# Vérification
print("Type :", type(df["corpus-tokens"].iloc[0]))
print("Exemple :", df["corpus-tokens"].iloc[0][:10])
```

    Type : <class 'list'>
    Exemple : ['Monsieur', 'maire', 'saint-brieuc', 'cher', 'ami', 'Monsieur', 'président', 'conseil', 'régional', 'Monsieur']
    

Sauvegarde du texte nettoyé, exportation du dataframe au format CSV :


```python
df.to_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/analyse-discours-presidentiels-nettoyes.csv")
```

## **3-) Représentation temporelle**
### Occurences simples du lexique de la guerre dans les discours présidentiels français, représenation temporelle, occurrence pour 10 000 mots, TF-IDF & Co-occurrence

### Comptage brut

#### Comptage de la guerre sans analyse temporelle

On commencer par importer la base de donnée nettoyée et sauvegardée. On vérifie ensuite son contenu.


```python
# Importation de la base de donnée (avec les années en français)
df = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/analyse-discours-presidentiels-nettoyes.csv")
```


```python
# Faire en sorte que le corpus soit un texte et non pas une liste de tokens
df["corpus-nettoyé"] = df["corpus-nettoyé"].astype(str)
```


```python
# Vérification du contenu
print(df.head())
```

      web-scraper-order                              web-scraper-start-url  \
    0      1768752907-1  https://www.vie-publique.fr/discours/recherche...   
    1      1768752909-2  https://www.vie-publique.fr/discours/recherche...   
    2      1768752912-3  https://www.vie-publique.fr/discours/recherche...   
    3      1768752915-4  https://www.vie-publique.fr/discours/recherche...   
    4      1768752921-5  https://www.vie-publique.fr/discours/recherche...   
    
                                    discours-individuels  \
    0  Discours de M. Jacques Chirac, Président de la...   
    1  Discours de M. Jacques Chirac, Président de la...   
    2  Intervention de M. Jacques Chirac, Président d...   
    3  Intervention télévisée de M. Jacques Chirac, P...   
    4  Lettre de M. Jacques Chirac, Président de la R...   
    
                               discours-individuels-href  \
    0  https://www.vie-publique.fr/discours/195774-di...   
    1  https://www.vie-publique.fr/discours/194083-di...   
    2  https://www.vie-publique.fr/discours/196432-in...   
    3  https://www.vie-publique.fr/discours/194790-in...   
    4  https://www.vie-publique.fr/discours/194906-le...   
    
                                                   titre     thématique  \
    0  Discours de M. Jacques Chirac, Président de la...            NaN   
    1  Discours de M. Jacques Chirac, Président de la...       Économie   
    2  Intervention de M. Jacques Chirac, Président d...  International   
    3  Intervention télévisée de M. Jacques Chirac, P...        Société   
    4  Lettre de M. Jacques Chirac, Président de la R...  International   
    
                 détails     intervenant  \
    0  10 septembre 2001  Jacques Chirac   
    1  11 septembre 2001  Jacques Chirac   
    2  11 septembre 2001  Jacques Chirac   
    3  11 septembre 2001  Jacques Chirac   
    4  11 septembre 2001  Jacques Chirac   
    
                                                   corps  \
    0  Monsieur le Maire de Saint-Brieuc, mon cher am...   
    1  Monsieur le président LEMETAYER,\nMonsieur le ...   
    2  Monsieur le Président,\nMesdames, messieurs,\n...   
    3  Mes chers compatriotes,\nLes attentats qui ont...   
    4  Monsieur le Président,\nC'est avec une immense...   
    
                       mots-cles  \
    0       Situation économique   
    1             Vie économique   
    2  Relations internationales   
    3                   Sécurité   
    4  Relations internationales   
    
                                                   pages  \
    0  https://www.vie-publique.fr/discours/recherche...   
    1  https://www.vie-publique.fr/discours/recherche...   
    2  https://www.vie-publique.fr/discours/recherche...   
    3  https://www.vie-publique.fr/discours/recherche...   
    4  https://www.vie-publique.fr/discours/recherche...   
    
                                          corpus-nettoyé        date  \
    0  Monsieur maire saint-brieuc cher ami Monsieur ...  2001-09-10   
    1  Monsieur président lemetayer Monsieur ministre...  2001-09-11   
    2  Monsieur président madame monsieur venir ici g...  2001-09-11   
    3  cher compatriote attentat frapper aujourd'hui ...  2001-09-11   
    4  Monsieur président immense émotion france appr...  2001-09-11   
    
                                           corpus-tokens  
    0  ['Monsieur', 'maire', 'saint-brieuc', 'cher', ...  
    1  ['Monsieur', 'président', 'lemetayer', 'Monsie...  
    2  ['Monsieur', 'président', 'madame', 'monsieur'...  
    3  ['cher', 'compatriote', 'attentat', 'frapper',...  
    4  ['Monsieur', 'président', 'immense', 'émotion'...  
    

On va ensuite compter le nombre d'occurrences totales du mot "guerre" dans tout le corpus, puis produire un ratio occurrences/discours.


```python
# Compteur d'occurrence du mot guerre
occurrence_guerre = 0

# Pour tous les discours du corpus nettoyé, compter le nombre d'occurrence du mot guerre
for mot in df["corpus-nettoyé"]:
    occurrence_guerre += mot.count("guerre")

# Afficher le nombre d'occurrences
print(f"Occurence du mot guerre : {occurrence_guerre}")
print(f"Nombre de discours comptés : {df["corpus-nettoyé"].count()}")

# Calculer un ratio mention de la guerre / discours
nombre_discours = df["corpus-nettoyé"].count()
print(f"Occurrence du mot guerre par discours : {occurrence_guerre/nombre_discours}")
```

    Occurence du mot guerre : 5377
    Nombre de discours comptés : 5524
    Occurrence du mot guerre par discours : 0.973388848660391
    

On veut ensuite voir l'évolution de la mention de la guerre à travers le temps.


```python
def compter_guerre(mot):
    compte = mot.count("guerre")
    return compte
```


```python
df["nombre_mention_guerre"] = df["corpus-nettoyé"].apply(compter_guerre)
```


```python
df["nombre_mention_guerre"].head()
```




    0    0
    1    1
    2    0
    3    0
    4    0
    Name: nombre_mention_guerre, dtype: int64



#### Analyse temporelle de la guerre


```python
# Représentation graphique en fonction du temps
df.plot.line(x="date", y="nombre_mention_guerre")
```




    <Axes: xlabel='date'>




    
![png](output_54_1.png)
    



```python
#Filtrage de la date par année simplement
df["année"] = df["date"].dt.year
print(df["année"].head())
```

    0    2001.0
    1    2001.0
    2    2001.0
    3    2001.0
    4    2001.0
    Name: année, dtype: float64
    


```python
print(f"Nombre de valeurs manquantes : {df["date"].isna().sum()}")
```

    Nombre de valeurs manquantes : 154
    


```python
mention_mots_année = pd.DataFrame(mention_guerre_année)
```


```python
mention_guerre_année = df.groupby("année")["nombre_mention_guerre"].sum()
```


```python
print(mention_guerre_année)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[48], line 1
    ----> 1 print(mention_guerre_année)
    

    NameError: name 'mention_guerre_année' is not defined



```python
print(mention_mots_année.head())
```

            nombre_mention_guerre
    année                        
    2001.0                     68
    2002.0                    134
    2003.0                    303
    2004.0                    132
    2005.0                     78
    


```python
mention_mots_année.plot.line(y="nombre_mention_guerre")
```




    <Axes: xlabel='année'>




    
![png](output_61_1.png)
    


#### Comptage du lexique de la guerre et représentation temporelle


```python
# Choix d'un mot
print("Choisir un mot à compter")
mot_choisi = input()
print("On a choisi le mot :" + " " + mot_choisi)
```

    Choisir un mot à compter
    

     Terrorisme
    

    On a choisi le mot : Terrorisme
    


```python
# Fonction de comptage de mots choisis
def compter_mot(discours):
    compte = discours.count(mot_choisi)
    return compte
```


```python
print(mention_mots_année.head())
```

            nombre_mention_guerre
    année                        
    2001.0                     68
    2002.0                    134
    2003.0                    303
    2004.0                    132
    2005.0                     78
    


```python
# Choix d'un mot
print("Choisir un mot à compter")
mot_choisi = input()
print("On a choisi le mot :" + " " + mot_choisi)

# Fonction de comptage de mots au choix
def compter_mot(discours):
    compte = discours.count(mot_choisi)
    return compte

# Comptage du mot choisi
print("Comptage du mot choisi")
df["nombre_mentions_mot_choisi"] = df["corpus-nettoyé"].apply(compter_mot)

# Regroupement par année
print("Regroupement par année")
mentions_mot_choisi_année = df.groupby("année")["nombre_mentions_mot_choisi"].sum()

# Ajout au dataframe de comptage
print("Ajout au Dataframe de comptage")
mention_mots_année[mot_choisi] = mentions_mot_choisi_année

# Validation des résultats
print("La fonction a fonctionné youpi !")
```

    Choisir un mot à compter
    

     gaspard
    

    On a choisi le mot : gaspard
    Comptage du mot choisi
    Regroupement par année
    Ajout au Dataframe de comptage
    La fonction a fonctionné youpi !
    


```python
print(mention_mots_année.head())
```

            nombre_mention_guerre  terrorisme  Melissa  gaspard
    année                                                      
    2001.0                     68         310        0        0
    2002.0                    134         220        0        0
    2003.0                    303         165        0        0
    2004.0                    132         206        0        0
    2005.0                     78          98        0        0
    


```python
mention_mots_année.plot.line(y=["gaspard"])
```




    <Axes: xlabel='année'>




    
![png](output_68_1.png)
    



```python
mention_mots_année = mention_mots_année.drop("terrorisme", axis = 1)
```


```python
mention_mots_année.to_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_brut.csv")
```

### Occurrence pour 1000 mots


```python
mention_10000 = pd.DataFrame()
```


```python
# Choix d'un mot
print("Choisir un mot à compter")
mot_choisi = input()
print("On a choisi le mot :" + " " + mot_choisi)

# Fonction de comptage de mots au choix
def compter_mot_10000(discours):
    compte = discours.count(mot_choisi)
    longueur = len(discours)
    résultat = (compte / longueur) * 10000
    return résultat

# Application
print("Comptage")
df["compte-10000"] = df["corpus-tokens"].apply(compter_mot_10000)
print("Regroupement")
mention_10000[mot_choisi] = df.groupby("année")["compte-10000"].mean()
print(mention_10000[mot_choisi].mean())
print("Fini")
```

    Choisir un mot à compter
    

     combat
    

    On a choisi le mot : combat
    Comptage
    Regroupement
    9.823445198941402
    Fini
    


```python
print(mention_10000.head())
```

               guerre       paix   sécurité  mobilisation  réarmement     armée  \
    année                                                                         
    2001.0  11.310295  25.886105  25.944289      3.587053    0.000000  5.890807   
    2002.0   7.231691  16.959144  30.243960      4.432326    0.097677  9.784633   
    2003.0  10.507901  22.136598  22.905702      5.734779    0.000000  3.657114   
    2004.0   5.574556  27.608531  20.647349      5.421994    0.000000  6.556760   
    2005.0   3.864744  27.959190  20.104472      6.341646    0.000000  4.233997   
    
             conflit  mobiliser    défense     front  attaquer  bataille  \
    année                                                                  
    2001.0  8.253164   6.877781   7.565359  1.583265  2.229691  1.259448   
    2002.0  4.107639   6.019574  11.302255  0.681147  0.522536  1.197492   
    2003.0  6.453131   5.791511   7.986109  1.065969  1.141838  0.215260   
    2004.0  5.068190   8.215026   5.884646  0.731813  0.983879  1.517508   
    2005.0  6.217995   8.036844  10.087769  0.257866  0.400539  1.024138   
    
             attaque   riposte  déploiement  stratégique  tactique  combattre  \
    année                                                                       
    2001.0  5.406173  3.132152     0.250105     1.377579  0.000000   4.661113   
    2002.0  1.743114  0.030307     0.368573     2.094152  0.046675   3.243085   
    2003.0  0.231212  0.058761     0.734919     2.000383  0.244531   1.210583   
    2004.0  0.267492  0.000000     0.287138     3.100584  0.054882   2.810315   
    2005.0  0.207150  0.000000     0.198416     3.588168  0.068809   4.325326   
    
               combat  
    année              
    2001.0  16.366117  
    2002.0   9.380159  
    2003.0   5.856501  
    2004.0  15.185504  
    2005.0   6.602530  
    


```python
mention_10000.to_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_guerre_v20.csv")
```


```python
mention_10000.to_clipboard()
```

### TF-IDF

Maintenant qu'on a identifié des périodes clés dans le recours au lexique de la guerre, on va essayer d'affiner l'analyse avec de meilleurs indicateurs (TF-IDF). Cela permet d'identifier des discours dans lesquels l'usage de la guerre est particulièrement marquant relativement au document et l'ensemble du corpus.


```python
type(df["corpus-tokens"])
```




    pandas.core.series.Series




```python
print(df["corpus-tokens"].head())
```

    0    [Monsieur, maire, saint-brieuc, cher, ami, Mon...
    1    [Monsieur, président, lemetayer, Monsieur, min...
    2    [Monsieur, président, madame, monsieur, venir,...
    3    [cher, compatriote, attentat, frapper, aujourd...
    4    [Monsieur, président, immense, émotion, france...
    Name: corpus-tokens, dtype: object
    


```python
# Top 10 des scores TF-IDF par discours (à partir d'Émilien Schultz & Matthias Bussonnier)
dictionnary = Dictionary(df["corpus-tokens"])
corpus_converti = [dictionnary.doc2bow(line)
                  for line in list(df["corpus-tokens"])]
modele_tfidf = TfidfModel(corpus_converti)

vectors = [modele_tfidf[i] for i in corpus_converti]

def top_10(vec, dic):
    serie = pd.Series({dic[i[0]]:i[1] for i in vec})
    return serie.sort_values()[0:10].index

tab = pd.DataFrame([top_10(vec, dictionnary) for vec in vectors],
                   index = df["date"])
tab.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-09-10</th>
      <td>france</td>
      <td>faire</td>
      <td>grand</td>
      <td>français</td>
      <td>aller</td>
      <td>pays</td>
      <td>vouloir</td>
      <td>monde</td>
      <td>ici</td>
      <td>mettre</td>
    </tr>
    <tr>
      <th>2001-09-11</th>
      <td>monde</td>
      <td>beaucoup</td>
      <td>donner</td>
      <td>moment</td>
      <td>jour</td>
      <td>venir</td>
      <td>remercier</td>
      <td>responsabilité</td>
      <td>porter</td>
      <td>besoin</td>
    </tr>
    <tr>
      <th>2001-09-11</th>
      <td>vouloir</td>
      <td>france</td>
      <td>grand</td>
      <td>français</td>
      <td>aller</td>
      <td>bien</td>
      <td>Monsieur</td>
      <td>fois</td>
      <td>falloir</td>
      <td>monsieur</td>
    </tr>
    <tr>
      <th>2001-09-11</th>
      <td>france</td>
      <td>pays</td>
      <td>pouvoir</td>
      <td>vouloir</td>
      <td>faire</td>
      <td>savoir</td>
      <td>aujourd'hui</td>
      <td>entrer</td>
      <td>français</td>
      <td>aller</td>
    </tr>
    <tr>
      <th>2001-09-11</th>
      <td>france</td>
      <td>français</td>
      <td>aller</td>
      <td>bien</td>
      <td>venir</td>
      <td>falloir</td>
      <td>contre</td>
      <td>président</td>
      <td>etat</td>
      <td>luire</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fonction de calcul de tf-idf en fonction d'un mot choisi
def score_mot_choisi(vec, dictionnary, mot_choisi):
    """
    Retourne le score TF-IDF d'un mot choisi dans un discours.
    
    Args:
        vec: liste de tuples (token_id, score_tfidf) pour un discours
        dictionnaire: l'objet Dictionary de Gensim
        mot_choisi: le mot dont on veut le score (ex: "guerre")
    
    Returns:
        Le score TF-IDF du mot choisi, ou 0.0 si absent
    """
    # 1. Trouver l'ID du mot cible
    token_id_mot_choisi = dictionnary.token2id.get(mot_choisi)

    # 2. Si le mot n'existe pas dans le dictionnaire, score = 0
    if token_id_mot_choisi is None:
        return 0.0
    
    # 3. Chercher le score dans le vecteur du discours
    # vec = [(0, 0.15), (1, 0.32), (2, 0.08), ...]
    # On cherche le tuple où le premier élément == token_id
    for token_id_score_vec, score in vec:
        if token_id_score_vec == token_id_mot_choisi:
            return score
    
    # 4. Si trouvé, retourner le score; sinon 0.0
    return 0.0
```


```python
# Dataframe du score par année d'un mot choisi
score_mots_année = pd.DataFrame()
```


```python
## Choix d'un mot
print("Choisir un mot à compter")
mot_choisi = input()
print("On a choisi le mot :" + " " + mot_choisi)

# Application de la fonction de calcul "score_mot_choisi" à la liste de vector
print("Calcul du score tf-idf du mot choisi")
score_vec_identifié = [score_mot_choisi(vec, dictionnary, mot_choisi) for vec in vectors]
df["score-mot-choisi"] = score_vec_identifié
print(df["score-mot-choisi"].head())

# Création d'un dataframe ad-hoc pour le tf-idf
score_mot_choisi_année = df.groupby("année")["score-mot-choisi"].mean()
score_mots_année[mot_choisi] = score_mot_choisi_année
```

    Choisir un mot à compter
    

     combat
    

    On a choisi le mot : combat
    Calcul du score tf-idf du mot choisi
    0    0.000000
    1    0.012942
    2    0.000000
    3    0.000000
    4    0.000000
    Name: score-mot-choisi, dtype: float64
    


```python
print(score_mots_année.tail())
```

              guerre      paix  sécurité  mobilisation  réarmement     armée  \
    année                                                                      
    2021.0  0.003980  0.005002  0.007966      0.006273    0.000175  0.016488   
    2022.0  0.033076  0.010884  0.010234      0.008686    0.001003  0.013546   
    2023.0  0.017818  0.010839  0.009831      0.006004    0.003076  0.011249   
    2024.0  0.023195  0.014673  0.014265      0.005574    0.002103  0.017219   
    2025.0  0.014845  0.027175  0.017609      0.010438    0.001121  0.022354   
    
             conflit    menace  mobiliser   défense     front  attaquer  bataille  \
    année                                                                           
    2021.0  0.002601  0.004016   0.009771  0.010758  0.001867  0.001967  0.004922   
    2022.0  0.006768  0.005458   0.010323  0.012303  0.002130  0.004047  0.003681   
    2023.0  0.006427  0.002802   0.007223  0.012917  0.002545  0.002375  0.008703   
    2024.0  0.007678  0.002688   0.007966  0.019242  0.004221  0.002616  0.004565   
    2025.0  0.005568  0.008633   0.010891  0.015613  0.002789  0.003398  0.005030   
    
             attaque   riposte  déploiement  stratégique  tactique  combattre  \
    année                                                                       
    2021.0  0.004662  0.001056     0.003519     0.010882  0.000192   0.001710   
    2022.0  0.003754  0.000405     0.006190     0.011311  0.000371   0.003267   
    2023.0  0.006462  0.000155     0.003640     0.016765  0.000106   0.001543   
    2024.0  0.009775  0.000532     0.001752     0.015306  0.000747   0.001646   
    2025.0  0.006878  0.001123     0.003025     0.014520  0.000707   0.001364   
    
              combat  
    année             
    2021.0  0.013410  
    2022.0  0.011953  
    2023.0  0.007923  
    2024.0  0.008821  
    2025.0  0.012118  
    


```python
# Moyenne des scores moyens par année
score_mots_année["tfidf_moyen"] = score_mots_année.mean(axis=1)
```


```python
score_mots_année.to_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/score.csv")
```

### Co-occurrence


```python
print(df["corpus-tokens"].iloc[5522])
```

    ['madame', 'monsieur', 'ministre', 'Monsieur', 'président', 'commission', 'affaire', 'étranger', 'assemblée', 'national', 'monsieur', 'député', 'Monsieur', 'chef', 'état', 'major', 'armée', 'Monsieur', 'ambassadeur', 'monsieur', 'officier', 'général', 'madame', 'monsieur', 'officier', 'sous-officier', 'officier', 'marinier', 'soldat', 'marin', 'aviateur', 'personnel', 'civil', 'honorer', 'veille', 'noël', 'ici', 'milieu', 'désert', 'émirat', 'arabe', 'unir', 'sein', 'prestigieu', '"', 'royal', 'pologne', '"', 'fin', 'année', 'pensée', 'frère', 'arme', 'mort', 'année', 'exercice', 'mission', 'penser', 'particulière', 'sergent', 'jimmy', 'gosselin', '7ème', 'bataillon', 'chasseur', 'alpin', 'varce', 'mort', 'durer', 'mission', 'harpie', 'guyane', 'chef', 'armée', 'vouloir', 'reconnaissance', 'profond', 'nation', 'sacrifice', 'solidarité', 'absolu', 'devoir', 'famille', 'période', 'fête', 'au-delà', 'vie', 'jour', 'engager', 'battr', 'servir', 'tomber', 'vivre', 'esprit', 'cœur', 'vivant', 'rester', 'engager', 'chemin', 'force', 'espérance', 'ensemble', 'emprunter', 'pensée', 'aller', 'blessé', 'année', 'corps', 'meurtrir', 'âme', 'empreinte', 'violence', 'déposer', 'épreuve', 'devoir', 'épreuve', 'nation', 'là.je', 'veux', 'saluer', 'ici', 'travail', 'accompagnement', 'association', 'ministèreensemble', 'service', 'état', '(', 'administration', 'central', 'service', 'déconcentrer', ')', 'placer', 'sou', 'responsabilité', 'ministre', 'armée', 'ancien', 'combattant', 'aider', 'mieux', 'prendre', 'charge', 'souffrance', 'pays', 'fort', 'pays', 'reconnaître', 'servent', 'force', 'moral', 'lien', 'fort', 'blessé', 'famille', 'famille', 'endeuiller', 'grand', 'nation', 'force', 'attention', 'louis', 'xiv', 'chef', 'état', 'protecteur', 'institution', 'national', 'invalide', 'pensionnaire', 'fraternité', 'attention', 'faire', 'nation.je', 'veux', 'ici', 'témoigner', 'reconnaissance', 'loin', 'métropole', 'loin', 'servir', 'sein', 'force', 'français', 'émirat', 'arabe', 'unir', 'affecté', 'mission', 'court', 'durée', 'remercier', 'accueil', 'chaleureux', 'féliciter', 'participer', 'organisation', 'déplacement', 'travers', 'ensemble', 'engagement', 'soldat', 'aviateur', 'marin', 'médecin', 'infirmier', 'commissaire', 'ingénieur', 'personnel', 'civil', 'ministèreensemble', 'service', 'état', '(', 'administration', 'central', 'service', 'déconcentrer', ')', 'placer', 'sou', 'responsabilité', 'ministre', 'active', 'réserve', 'vouloir', 'saluer', 'engager', 'permanence', 'dissuasion', 'opération', 'extérieurer', 'force', 'présence', 'force', 'prépositionner', 'mission', 'sentinell', 'permanence', 'veille', 'noël', 'savoir', 'détermination', 'réussite', 'mission', 'savoir', 'tribut', 'acquitter', 'réussite', 'nom', 'nation', '-en', 'remerciés.je', 'ici', 'région', 'monde', 'entremêlement', 'défi', 'temps', 'sécuritaires', 'terroriste', 'géopolitique', 'enjeux', 'rivalité', 'sécurité', 'amiral', 'base', 'opérationnel', 'avancer', 'mission', 'femme', 'homme', 'commander', 'étendre', 'canal', 'suer', 'australie', 'cœur', 'moyen-orient', 'ffeau', 'force', 'stabilité.stabilité', 'ressortissant', 'français', 'ressortissant', 'européen', 'région', 'témoigner', 'opération', 'évacuation', 'soudan', 'mener', 'avril', '2023', 'stabilité', 'côté', 'allé', 'lutter', 'contre', 'terrorisme', 'cadre', 'opération', 'chammal', 'irak', 'syrie', 'stabilité', 'immensité', 'mer', 'assurer', 'sécurité', 'maritime', 'quart', 'trafic', 'maritime', 'mondial', 'transiter', 'mer', 'rouge', 'océan', 'indien', 'point', 'passage', 'engagement', 'opération', 'union', 'européen', 'aspide', 'permettre', 'maintenir', 'trafic', 'commercial', 'défendre', 'navire', 'civil', 'transiter', 'détroit', 'bab', 'el', 'mandeb', 'sauver', 'vie', 'marin', 'danger', 'éviter', 'marée', 'noir', 'août', '2024', 'ciblage', 'houthi', 'pétrolier', 'sounion', 'pouvoir', 'citer', 'missions.depui', 'massacre', '7', 'octobre', '2023', 'israël', 'frégate', 'rang', 'atlantique', '2', 'appuyer', 'régulièrement', 'mirage', '2000', 'force', 'français', 'stationner', 'djibouti', 'répondre', 'succès', 'quotidien', 'menace', 'découlant', 'conflit', '200', 'bâtiment', 'escorter', 'dizaine', 'marin', 'sauvé', 'dizaine', 'drone', 'missile', 'interceptés.stabilité', 'océan', 'indien', 'lutter', 'contre', 'trafic', 'trafic', 'arme', 'composant', 'dual', 'alimenter', 'menace', 'houthi', 'narcotrafic', 'saisir', '17', 'tonne', 'année', 'mission', 'illustrer', 'engagement', 'conster', 'armées.a', 'heure', 'prédateur', 'devoir', 'fort', 'craint', 'particulier', 'fort', 'mer', 'conformément', 'dernier', 'loi', 'programmation', 'militaire', 'examen', 'complet', 'minutieux', 'décider', 'doter', 'france', 'porte-avions', 'décision', 'lancer', 'réalisation', 'grand', 'programme', 'prendre', 'semaine.madame', 'ministre', 'armée', 'particulièrement', 'remercier', 'qualité', 'travail', 'mener', 'état', 'major', 'dga', 'cea', 'industriel', 'chantier', 'aller', 'irriguer', 'économie', '800', 'fournisseur', '80', 'pourcent', 'pme', 'impliquer', 'construction', 'garant', 'engagement', 'faveur', 'entreprise', 'déplacer', 'chantier', 'février', 'prochain', 'rencontrer', 'porte-avions', 'illustration', 'puissance', 'nation', 'puissance', 'industrie', 'technique', 'puissance', 'service', 'liberté', 'mer', 'remous', 'temps.stabilité', 'profit', 'partenaire', 'stratégique', 'émirat', 'arabe', 'unir', 'entretenir', 'lien', 'grand', 'confiance', 'france', 'ici', 'nation', 'étranger', 'militaire', 'présent', 'famille', 'occasion', 'dernier', 'épisode', 'voir', 'péninsule', 'prise', 'cibl', 'bel', 'marque', 'confiance', 'pouvoir', 'témoigner', 'pays', 'accueille.cette', 'confiance', 'mutuel', 'inscrire', 'accord', 'défense', 'lier', 'émirat', 'arabe', 'unir', 'france', 'faire', 'démonstration', 'détermination', 'appliquer', 'clause', 'assistance', 'accord', 'prévoit.ainsi', '2022', 'abu', 'dhabi', 'cibler', 'drone', 'missile', 'houthi', 'france', 'réagi', 'immédiatement', 'rafale', 'base', 'aérien', '104', 'al', 'dhafra', 'batterie', 'crotal', 'juin', 'dernier', 'guerre', '12', 'jour', 'ffeau', 'faire', 'face', 'réactivité', 'soutenir', 'partenaire', 'émirien', 'qatarien', 'déployant', 'renfort', 'aérien', 'frégate', 'défense', 'aérien', 'confiance', 'démontrer', 'éprouver', 'moment', 'difficile', 'camarade', 'émirien', 'savoir', 'cas', 'crise', 'pouvoir', 'compter', 'france', 'réactivité', 'capacité', 'affronter', 'danger', 'menacer', 'confiance', 'réciproque', 'traduire', 'déploiement', 'matériel', 'équipement', 'haut', 'spectre', 'frégate', 'rang', 'chasseur', 'rafal', 'char', 'leclerc', 'canon', 'caesar', 'décembre', '2025', 'présence', 'renforcer', 'mise', 'place', 'compagnie', 'infanterie', 'permanent', 'a400', 'm', 'abu', 'dhabi.grâce', 'action', 'réactif', 'déterminée', 'ffeau', 'france', 'disposer', 'position', 'unique', 'moyen', 'orient', 'grâce', 'reconnaître', 'partenaire', 'fiable', 'crédible', 'fidèle.cett', 'stabilité', 'confiance', 'ffeau', 'entretiennent', 'quotidien', 'action', 'coopération', 'entraînement', 'camarade', 'émirien', '2025', 'marquer', 'avancée', 'majeur', 'domaine', 'récent', 'exercice', 'interarmer', 'quadriennal', 'gulf', '25', 'émirat', 'bel', 'démonstration', 'capacité', 'mettre', 'place', 'état', 'major', 'binational', 'interarmer', 'faire', 'opérer', 'côte', 'côte', 'composante', 'français', 'émirienne', 'terre', 'mer', 'airs.en', 'irak', 'création', 'emprise', 'français', 'camp', 'taji', 'continuer', 'formation', 'bataillon', 'armée', 'terre', 'irakien', 'sou', 'commandement', 'tactique', '5ème', 'cuir', 'aider', 'irak', 'recouvrer', 'pleine', 'souveraineté', 'assurer', 'sécurité', 'action', 'coopération', 'militaire', 'régional', 'essentiel', 'partenaire', 'déterminer', 'stratégie', 'accès', 'influence', 'région.les', 'eau', 'partenaire', 'plan', 'domaine', 'armement', 'commander', '80', 'rafal', 'avril', '2022', 'année', '2025', 'marque', 'sortie', 'production', 'rafal', 'début', 'campagne', 'essai', 'eau', 'client', 'france', 'moyen-orient', '600', 'entreprise', 'français', 'implanter', 'riche', 'coopération', 'nouer', 'domaine', 'énergétique', 'technologique', 'innovation', 'santé', 'éducation', 'implantation', 'sorbonne', 'domaine', 'culture', 'grâce', 'louvr', 'abu', 'dhabi', 'dialogue', 'stratégique', 'diversifier', 'riche', 'coopération', 'stratégique', 'tourner', 'avenir', 'co-investissement', 'majeur', 'intelligence', 'artificiel', 'visage', 'ambassadeur', 'manière', 'lien', 'amitié', 'confiance', 'entrer', 'pays', 'oui', 'compatriote', 'savoir', 'situation', 'international', 'peser', 'quotidien', 'conflit', 'ukraine', 'facteur', 'déstabilisation', 'europe', 'conflit', 'orient', 'retentissement', 'universel', 'source', 'inquiétude', 'particulier', 'région', 'passer', 'trouver', 'racine', 'remède', 'ailleurs', 'crise', 'géopolitique', 'guerre', 'trafic', 'former', 'horizon', 'péril', 'sentinelle', 'idéal', 'servir', 'ici', 'défendre', 'sauvegarde', 'intérêt', 'nation', 'sécurité', 'français', 'intérêt', 'côté', 'garant', 'celle-ci.alors', 'fin', 'année', 'ensemble', 'soyon', 'garant', 'fier', 'éclat', 'français', 'briller', 'ici', 'universel', 'place', 'ici', 'important', 'engagement', 'jour', 'ffeau', 'changer', 'perception', 'france', 'pays', 'région', 'mission', 'contribuer', 'stabilité', 'simplement', 'région', 'monde', 'venir', 'venir', 'ambition', 'continuer', 'armée', 'efficace', 'europe', 'doter', 'meilleur', 'équipement', 'poursuivre', 'modernisation', 'année', 'venir', 'celer', 'rien', 'femme', 'homme', 'faire', 'armée', 'fierté', "aujourd'hui", 'côté', 'année', 'tête', 'voir', 'durer', 'année', 'femme', 'homme', 'engager', 'patrie', 'servir', 'intérêt', 'valeur', 'prêt', 'sacrifice', 'ultime', 'jour', 'prêt', 'recommencer', 'celer', 'grand', 'force', 'oublier', 'jamais', 'grand', 'bel', 'inscrire', 'lignée', 'génération', 'permettre', 'france', 'libre', 'combat', 'guerre', 'mener', 'celer', 'fier', 'suis.vive', 'amitié', 'franco-émirienn', 'vivre', 'république', 'vivre', 'france', '!']
    


```python
# POC : compteur de co-occurrence (Deepseek) - 13 février 2026
def compter_cooccurrences(discours_tokens, mot_cible, fenetre=5):
    """
    Compte les mots qui co-occurrent avec mot_cible dans une fenêtre donnée.
    
    Args:
        discours_tokens: liste de mots (ex: ["le", "président", "parle", "de", "guerre"])
        mot_cible: le mot dont on veut les co-occurrences (ex: "guerre")
        fenetre: nombre de mots avant et après à considérer
    
    Returns:
        dictionnaire {mot_cooccurrent: nombre_occurrences}
    """
    cooc = {}
    
    # Parcourir le discours pour trouver toutes les positions de mot_cible
    for i, mot in enumerate(discours_tokens):
        if mot == mot_cible:
            # Définir la fenêtre autour de cette occurrence
            debut = max(0, i - fenetre)  # Ne pas sortir du début
            fin = min(len(discours_tokens), i + fenetre + 1)  # Ne pas sortir de la fin
            
            # Examiner tous les mots dans la fenêtre
            for j in range(debut, fin):
                if j != i:  # Ne pas compter le mot lui-même
                    mot_contextuel = discours_tokens[j]
                    cooc[mot_contextuel] = cooc.get(mot_contextuel, 0) + 1
    
    return cooc
```


```python
# Test manuel pour comprendre
test_discours = ["la", "guerre", "en", "ukraine", "est", "terrible", "mais", "nous", "résisterons", "comme", "nous", "l'avons", "toujours", "fait", ".", "Cependant", "la", "guerre", "a", "aussi", "toujours", "une", "fin", "."]
resultat = compter_cooccurrences(test_discours, "guerre", fenetre=5)
print("Co-occurrences de 'guerre' (fenêtre=2):")
for mot, count in sorted(resultat.items(), key=lambda x: x[1], reverse=True):
    print(f"  {mot}: {count} fois")
```

    Co-occurrences de 'guerre' (fenêtre=2):
      la: 2 fois
      toujours: 2 fois
      en: 1 fois
      ukraine: 1 fois
      est: 1 fois
      terrible: 1 fois
      mais: 1 fois
      fait: 1 fois
      .: 1 fois
      Cependant: 1 fois
      a: 1 fois
      aussi: 1 fois
      une: 1 fois
      fin: 1 fois
    


```python
# Appropriation du compteur
def compteur_cooccurrence(discours, mot_choisi, fenetre=5):

    # Liste de coocurrence
    cooc={}

    # Pour tout x mot contenu dans le discours, si le mot correspond au mot choisi regarder dans une fenêtre prédéfinie les mots
    for x, mot in enumerate(discours):
        if mot == mot_choisi:
            debut = max(0, x - fenetre)
            fin = min(len(discours), x + fenetre + 1)

        # Pour tout mot y contenu dans la fenêtre de x, le mettre dans la liste de cooccurrence
            for y in range(debut, fin):
                if y != x:
                    mot_cooccurrent = discours[y]
                    cooc[mot_cooccurrent] = cooc.get(mot_cooccurrent, 0) + 1
    return cooc
```


```python
# Phrase test
discours_lambda = ["Je", "pense", "que", "la", "guerre", "devrait", "être", "interdite", "pour", "le", "bien", "de", "tous", ".", "Non", "pas", "que", "la","guerre", "soit", "évitable", "mais", "que", "seul", "un", "cadre", "international", "peut", "éviter", "la", "guerre", "."]

# Choix d'un mot
print("Choisir un mot:")
mot_choisi = input()

# Application du compteur
boite_cooccurrence = compteur_cooccurrence(discours_lambda, mot_choisi, fenetre=7)

for x, count in sorted(boite_cooccurrence.items(), key=lambda x: x[1], reverse=True):
    print(f"{x}: {count} fois")
```

    Choisir un mot:
    

     guerre
    

    que: 3 fois
    la: 3 fois
    de: 2 fois
    .: 2 fois
    seul: 2 fois
    un: 2 fois
    cadre: 2 fois
    Je: 1 fois
    pense: 1 fois
    devrait: 1 fois
    être: 1 fois
    interdite: 1 fois
    pour: 1 fois
    le: 1 fois
    bien: 1 fois
    tous: 1 fois
    Non: 1 fois
    pas: 1 fois
    soit: 1 fois
    évitable: 1 fois
    mais: 1 fois
    international: 1 fois
    peut: 1 fois
    éviter: 1 fois
    


```python
# Version 2 (autonomie) : comptage de mots prédéfinis dans une fenêtre x
def compteur_cooccurrence_predefini(discours, mot_referent, mot_cible, fenetre=7):

    # Compteur de coocurrence
    compteur_cooccurrences = 0

    # Pour toute apparition du mot choisi dans le discours, regarder les mots environnants dans une fenêtre prédéfinie
    for x, mot in enumerate(discours):
        if mot == mot_referent:
            debut = max(0, x - fenetre)
            fin = min(len(discours), x + fenetre + 1)
            print("Le mot référent apparaît")
            print(f"Fenêtre de début: {debut}. Fenêtre de fin: {fin}")

        # Pour tout mot cible contenu dans la fenêtre de référence, ajouter 1 au compteur de cooccurrence
            print("Comptage du mot cible")
            for y in range(debut, fin):
                if y != x:
                    print(f"Il y a un autre mot que {mot_referent} dans la fenêtre")
                    if discours[y] == mot_cible:
                        print(f"Le mot cible {mot_cible} apparaît")
                        compteur_cooccurrences += 1
                        print(compteur_cooccurrences)
                    else: print("Mais ce n'est pas le mot cible")
    return compteur_cooccurrences
```


```python
# Phrase test
discours_lambda = ["Je", "pense", "que", "la", "guerre", "devrait", "être", "interdite", "pour", "le", "bien", "de", "tous", ".", "Non", "pas", "que", "la","guerre", "soit", "évitable", "mais", "que", "seul", "un", "cadre", "international", "peut", "éviter", "la", "guerre", "."]

# Choix du mot de référence
print("Choisir un mot de référence")
mot_referent = input()

# Choix du mot cible
print("Choisir un mot cible")
mot_cible = input()

print(f"Mot de référence: {mot_referent}. Mot cible: {mot_cible}")

# Application du compteur
cooccurrences = compteur_cooccurrence_predefini(discours_lambda, mot_referent, mot_cible, fenetre=7)
print(f"Nombre de cooccurrence de {mot_referent} et {mot_cible}: {cooccurrences}")
```

    Choisir un mot de référence
    

     guerre
    

    Choisir un mot cible
    

     devrait
    

    Mot de référence: guerre. Mot cible: devrait
    Le mot référent apparaît
    Fenêtre de début: 0. Fenêtre de fin: 12
    Comptage du mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Le mot cible devrait apparaît
    1
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Le mot référent apparaît
    Fenêtre de début: 11. Fenêtre de fin: 26
    Comptage du mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Le mot référent apparaît
    Fenêtre de début: 23. Fenêtre de fin: 32
    Comptage du mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Il y a un autre mot que guerre dans la fenêtre
    Mais ce n'est pas le mot cible
    Nombre de cooccurrence de guerre et devrait: 1
    

Application au corpus de la guerre


```python
# Version 2 (autonomie) : comptage de mots prédéfinis dans une fenêtre x
def compteur_cooccurrence_predefini_corpus(discours, mot_referent, mot_cible, fenetre=7):

    # Compteur de coocurrence
    compteur_cooccurrences = 0

    # Pour toute apparition du mot choisi dans le discours, regarder les mots environnants dans une fenêtre prédéfinie
    for x, mot in enumerate(discours):
        if mot == mot_referent:
            debut = max(0, x - fenetre)
            fin = min(len(discours), x + fenetre + 1)

        # Pour tout mot cible contenu dans la fenêtre de référence, ajouter 1 au compteur de cooccurrence
            for y in range(debut, fin):
                if y != x:
                    if discours[y] == mot_cible:
                        compteur_cooccurrences += 1
    return compteur_cooccurrences
```


```python
cooccurrences = pd.DataFrame()
```


```python
# Choix du mot de référence
print("Choisir un mot de référence")
mot_referent = input()

# Choix du mot cible
print("Choisir un mot cible")
mot_cible = input()

print(f"Mot de référence: {mot_referent}. Mot cible: {mot_cible}")

# Application du compteur
df["cooccurrence"] = df["corpus-tokens"].apply(lambda discours: compteur_cooccurrence_predefini_corpus(discours, mot_referent, mot_cible, fenetre=20))
print(df["cooccurrence"].sum())

# Attribution à un dataframe annuel
cooccurrences[f"{mot_referent}-{mot_cible}"] = df.groupby("année")["cooccurrence"].sum()
```

    Choisir un mot de référence
    

     sécurité
    

    Choisir un mot cible
    

     militaire
    

    Mot de référence: sécurité. Mot cible: militaire
    915
    


```python
cooccurrences.to_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/cooccurrences_sécurité.csv")
```


```python
# Co-occurrence inter-lexicale
# Version 3 (autonomie) : comptage de mots d'un lexique de référence et d'un lexique cible dans une fenêtre x
def compteur_cooccurrence_interlexical(discours, dictionnaire_referent, dictionnaire_cible, fenetre=7):

    # Transformation des dictionnaires en sets (Deepseek)
    set_referent = dictionnaire_referent
    set_cible = dictionnaire_cible
    
    # Compteur de coocurrence
    compteur_cooccurrences = 0

    # Pour toute apparition d'un mot du lexique de référence dans le discours, regarder les mots environnants dans une fenêtre prédéfinie
    for x, mot in enumerate(discours):
        if mot == mot in set_referent:
            debut = max(0, x - fenetre)
            fin = min(len(discours), x + fenetre + 1)

        # Pour tout mot du lexique cible contenu dans la fenêtre de référence, ajouter 1 au compteur de cooccurrence
            for y in range(debut, fin):
                if y != x and discours[y] in set_cible:
                    compteur_cooccurrences += 1
    
    return compteur_cooccurrences
```


```python
# Choix du dictionnaire de référence
print("Dictionnaire de référence:")
dictionnaire_referent = ["guerre", "paix", "sécurité", "mobilisation", "réarmement", "armée", "conflit", "menace", "mobiliser", "défense", "front", "attaquer", "bataille", "attaque", "riposte", "déploiement", "stratégique", "tactique", "combattre", "combat"]
print(f"{dictionnaire_referent} Type : {type(dictionnaire_referent)}")

# Choix du dictionnaire cible
print("Dictionnaire cible:")
dictionnaire_cible = ["culture", "culturel", "identité", "identitaire", "valeur", "tradition", "héritage", "civilisation", "histoire", "artistique"]
print(f"{dictionnaire_cible} Type : {type(dictionnaire_cible)}")

# -------------------
# Dictionnaire économique : ["économique", "croissance", "développement", "inflation", "budget", "financement", "investissement", "salaire", "capital", "profit"]
# Dictionnaire santé : ["santé", "sanitaire", "soin", "soignant", "médecin", "médical" "hygiène", "maladie", "vaccin", "médicament"]
# Dictionnaire éducation : ["éducation", "université", "éducationnel", "enseignant", "élève", "professeur", "diplôme", "baccalauréat", "master", "enseignement"]
# Dictionnaire environnement : ["environnement", "nature", "biodiversité", "biosphère", "climat", "climatique", "environnemental", "écologie", "écologique", "naturel"]
# Dictionnaire technologie : ["technologie", "technologique", "innovation", "numérique", "spatial", "digital", "robotique", "informatique", "automatisation", "ordinateur"]
# Dictionnaire social : ["social", "société", "solidarité", "assistance", "famille", "familial", "jeunesse", "retraite", "chômage", "âge"]
# Dictionnaire démographie : ["démographie", "démographique", "naissance", "natalité", "enfant", "vieillissement", "génération", "fécondité", "mortalité", "population"]
# Dictionnaire culture : ["culture", "culturel", "identité", "identitaire", "valeur", "tradition", "héritage", "civilisation", "histoire", "artistique"]
# -------------------

# Application du compteur
df["cooccurrence"] = df["corpus-tokens"].apply(lambda discours: compteur_cooccurrence_interlexical(discours, dictionnaire_referent, dictionnaire_cible, fenetre=10))
print("Nombre de cooccurrences:")
print(df["cooccurrence"].sum())

# Attribution à un dataframe annuel
cooccurrences["culture"] = df.groupby("année")["cooccurrence"].mean()
```

    Dictionnaire de référence:
    ['guerre', 'paix', 'sécurité', 'mobilisation', 'réarmement', 'armée', 'conflit', 'menace', 'mobiliser', 'défense', 'front', 'attaquer', 'bataille', 'attaque', 'riposte', 'déploiement', 'stratégique', 'tactique', 'combattre', 'combat'] Type : <class 'list'>
    Dictionnaire cible:
    ['culture', 'culturel', 'identité', 'identitaire', 'valeur', 'tradition', 'héritage', 'civilisation', 'histoire', 'artistique'] Type : <class 'list'>
    Nombre de cooccurrences:
    5165
    


```python
cooccurrences.to_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/cooccurrences.csv")
```


```python
cooccurrences.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cooccurrence-économie</th>
      <th>santé</th>
      <th>éducation</th>
      <th>environnement</th>
      <th>technologie</th>
      <th>social</th>
      <th>démographie</th>
      <th>culture</th>
    </tr>
    <tr>
      <th>année</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001.0</th>
      <td>0.797753</td>
      <td>0.134831</td>
      <td>0.134831</td>
      <td>0.393258</td>
      <td>0.078652</td>
      <td>0.842697</td>
      <td>0.359551</td>
      <td>0.910112</td>
    </tr>
    <tr>
      <th>2002.0</th>
      <td>1.026549</td>
      <td>0.407080</td>
      <td>0.203540</td>
      <td>0.712389</td>
      <td>0.154867</td>
      <td>1.716814</td>
      <td>0.305310</td>
      <td>1.079646</td>
    </tr>
    <tr>
      <th>2003.0</th>
      <td>1.014925</td>
      <td>0.242537</td>
      <td>0.111940</td>
      <td>0.328358</td>
      <td>0.029851</td>
      <td>0.966418</td>
      <td>0.235075</td>
      <td>1.033582</td>
    </tr>
    <tr>
      <th>2004.0</th>
      <td>1.273973</td>
      <td>0.246575</td>
      <td>0.105023</td>
      <td>0.369863</td>
      <td>0.082192</td>
      <td>1.109589</td>
      <td>0.315068</td>
      <td>1.200913</td>
    </tr>
    <tr>
      <th>2005.0</th>
      <td>1.198276</td>
      <td>0.133621</td>
      <td>0.120690</td>
      <td>0.366379</td>
      <td>0.193966</td>
      <td>0.965517</td>
      <td>0.297414</td>
      <td>0.823276</td>
    </tr>
  </tbody>
</table>
</div>



## **4-) Analyse statistique**
### (Statistique descriptive, occurrence de la guerre, saillance de la guerre, cooccurrence de la guerre)

### Statistiques descriptives


```python
# Statistiques descriptives
df = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/analyse-discours-presidentiels-nettoyes.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>web-scraper-order</th>
      <th>web-scraper-start-url</th>
      <th>discours-individuels</th>
      <th>discours-individuels-href</th>
      <th>titre</th>
      <th>thématique</th>
      <th>détails</th>
      <th>intervenant</th>
      <th>corps</th>
      <th>mots-cles</th>
      <th>pages</th>
      <th>corpus-nettoyé</th>
      <th>date</th>
      <th>corpus-tokens</th>
      <th>année</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1768752907-1</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>Discours de M. Jacques Chirac, Président de la...</td>
      <td>https://www.vie-publique.fr/discours/195774-di...</td>
      <td>Discours de M. Jacques Chirac, Président de la...</td>
      <td>NaN</td>
      <td>10 septembre 2001</td>
      <td>Jacques Chirac</td>
      <td>Monsieur le Maire de Saint-Brieuc, mon cher am...</td>
      <td>Situation économique</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>Monsieur maire saint-brieuc cher ami Monsieur ...</td>
      <td>2001-09-10</td>
      <td>['Monsieur', 'maire', 'saint-brieuc', 'cher', ...</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1768752909-2</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>Discours de M. Jacques Chirac, Président de la...</td>
      <td>https://www.vie-publique.fr/discours/194083-di...</td>
      <td>Discours de M. Jacques Chirac, Président de la...</td>
      <td>Économie</td>
      <td>11 septembre 2001</td>
      <td>Jacques Chirac</td>
      <td>Monsieur le président LEMETAYER,\nMonsieur le ...</td>
      <td>Vie économique</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>Monsieur président lemetayer Monsieur ministre...</td>
      <td>2001-09-11</td>
      <td>['Monsieur', 'président', 'lemetayer', 'Monsie...</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1768752912-3</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>Intervention de M. Jacques Chirac, Président d...</td>
      <td>https://www.vie-publique.fr/discours/196432-in...</td>
      <td>Intervention de M. Jacques Chirac, Président d...</td>
      <td>International</td>
      <td>11 septembre 2001</td>
      <td>Jacques Chirac</td>
      <td>Monsieur le Président,\nMesdames, messieurs,\n...</td>
      <td>Relations internationales</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>Monsieur président madame monsieur venir ici g...</td>
      <td>2001-09-11</td>
      <td>['Monsieur', 'président', 'madame', 'monsieur'...</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1768752915-4</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>Intervention télévisée de M. Jacques Chirac, P...</td>
      <td>https://www.vie-publique.fr/discours/194790-in...</td>
      <td>Intervention télévisée de M. Jacques Chirac, P...</td>
      <td>Société</td>
      <td>11 septembre 2001</td>
      <td>Jacques Chirac</td>
      <td>Mes chers compatriotes,\nLes attentats qui ont...</td>
      <td>Sécurité</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>cher compatriote attentat frapper aujourd'hui ...</td>
      <td>2001-09-11</td>
      <td>['cher', 'compatriote', 'attentat', 'frapper',...</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1768752921-5</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>Lettre de M. Jacques Chirac, Président de la R...</td>
      <td>https://www.vie-publique.fr/discours/194906-le...</td>
      <td>Lettre de M. Jacques Chirac, Président de la R...</td>
      <td>International</td>
      <td>11 septembre 2001</td>
      <td>Jacques Chirac</td>
      <td>Monsieur le Président,\nC'est avec une immense...</td>
      <td>Relations internationales</td>
      <td>https://www.vie-publique.fr/discours/recherche...</td>
      <td>Monsieur président immense émotion france appr...</td>
      <td>2001-09-11</td>
      <td>['Monsieur', 'président', 'immense', 'émotion'...</td>
      <td>2001.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["corpus-nettoyé"].count()
```




    np.int64(5521)




```python
df["année"].value_counts()
```




    année
    2013.0    332
    2016.0    330
    2017.0    310
    2015.0    306
    2008.0    299
    2014.0    294
    2007.0    286
    2006.0    278
    2003.0    268
    2012.0    242
    2005.0    232
    2002.0    226
    2009.0    223
    2004.0    219
    2010.0    187
    2011.0    176
    2023.0    170
    2018.0    159
    2025.0    147
    2019.0    125
    2024.0    124
    2021.0    123
    2020.0    114
    2022.0    111
    2001.0     89
    Name: count, dtype: int64




```python
année = pd.DataFrame()
année["nombre"] = df["année"].value_counts()
```


```python
année.to_clipboard()
```


```python
année["nombre"].describe()
```


```python
df["intervenant"].value_counts(normalize=True)
```


```python
intervenant = pd.DataFrame()
intervenant["nom"] = df["intervenant"].value_counts()
```


```python
intervenant["nom"].describe()
```




    count       4.000000
    mean     1381.000000
    std       192.964591
    min      1193.000000
    25%      1250.000000
    50%      1351.500000
    75%      1482.500000
    max      1628.000000
    Name: nom, dtype: float64



### Occurrence à travers le temps


```python
occurrence = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_guerre_v20.csv")
occurrence.set_index("année")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>guerre</th>
      <th>paix</th>
      <th>sécurité</th>
      <th>mobilisation</th>
      <th>réarmement</th>
      <th>armée</th>
      <th>conflit</th>
      <th>mobiliser</th>
      <th>défense</th>
      <th>front</th>
      <th>attaquer</th>
      <th>bataille</th>
      <th>attaque</th>
      <th>riposte</th>
      <th>déploiement</th>
      <th>stratégique</th>
      <th>tactique</th>
      <th>combattre</th>
      <th>combat</th>
    </tr>
    <tr>
      <th>année</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001.0</th>
      <td>11.310295</td>
      <td>25.886105</td>
      <td>25.944289</td>
      <td>3.587053</td>
      <td>0.000000</td>
      <td>5.890807</td>
      <td>8.253164</td>
      <td>6.877781</td>
      <td>7.565359</td>
      <td>1.583265</td>
      <td>2.229691</td>
      <td>1.259448</td>
      <td>5.406173</td>
      <td>3.132152</td>
      <td>0.250105</td>
      <td>1.377579</td>
      <td>0.000000</td>
      <td>4.661113</td>
      <td>16.366117</td>
    </tr>
    <tr>
      <th>2002.0</th>
      <td>7.231691</td>
      <td>16.959144</td>
      <td>30.243960</td>
      <td>4.432326</td>
      <td>0.097677</td>
      <td>9.784633</td>
      <td>4.107639</td>
      <td>6.019574</td>
      <td>11.302255</td>
      <td>0.681147</td>
      <td>0.522536</td>
      <td>1.197492</td>
      <td>1.743114</td>
      <td>0.030307</td>
      <td>0.368573</td>
      <td>2.094152</td>
      <td>0.046675</td>
      <td>3.243085</td>
      <td>9.380159</td>
    </tr>
    <tr>
      <th>2003.0</th>
      <td>10.507901</td>
      <td>22.136598</td>
      <td>22.905702</td>
      <td>5.734779</td>
      <td>0.000000</td>
      <td>3.657114</td>
      <td>6.453131</td>
      <td>5.791511</td>
      <td>7.986109</td>
      <td>1.065969</td>
      <td>1.141838</td>
      <td>0.215260</td>
      <td>0.231212</td>
      <td>0.058761</td>
      <td>0.734919</td>
      <td>2.000383</td>
      <td>0.244531</td>
      <td>1.210583</td>
      <td>5.856501</td>
    </tr>
    <tr>
      <th>2004.0</th>
      <td>5.574556</td>
      <td>27.608531</td>
      <td>20.647349</td>
      <td>5.421994</td>
      <td>0.000000</td>
      <td>6.556760</td>
      <td>5.068190</td>
      <td>8.215026</td>
      <td>5.884646</td>
      <td>0.731813</td>
      <td>0.983879</td>
      <td>1.517508</td>
      <td>0.267492</td>
      <td>0.000000</td>
      <td>0.287138</td>
      <td>3.100584</td>
      <td>0.054882</td>
      <td>2.810315</td>
      <td>15.185504</td>
    </tr>
    <tr>
      <th>2005.0</th>
      <td>3.864744</td>
      <td>27.959190</td>
      <td>20.104472</td>
      <td>6.341646</td>
      <td>0.000000</td>
      <td>4.233997</td>
      <td>6.217995</td>
      <td>8.036844</td>
      <td>10.087769</td>
      <td>0.257866</td>
      <td>0.400539</td>
      <td>1.024138</td>
      <td>0.207150</td>
      <td>0.000000</td>
      <td>0.198416</td>
      <td>3.588168</td>
      <td>0.068809</td>
      <td>4.325326</td>
      <td>6.602530</td>
    </tr>
    <tr>
      <th>2006.0</th>
      <td>4.238021</td>
      <td>28.069290</td>
      <td>24.619445</td>
      <td>3.699032</td>
      <td>0.000000</td>
      <td>4.068930</td>
      <td>4.703405</td>
      <td>9.073337</td>
      <td>7.473316</td>
      <td>0.292411</td>
      <td>1.041906</td>
      <td>0.898378</td>
      <td>0.116033</td>
      <td>0.052881</td>
      <td>0.552062</td>
      <td>6.828262</td>
      <td>0.048856</td>
      <td>3.048550</td>
      <td>10.166374</td>
    </tr>
    <tr>
      <th>2007.0</th>
      <td>5.727603</td>
      <td>17.266384</td>
      <td>21.954740</td>
      <td>4.712469</td>
      <td>0.000000</td>
      <td>4.927978</td>
      <td>2.825342</td>
      <td>5.138174</td>
      <td>12.423071</td>
      <td>0.828723</td>
      <td>0.598138</td>
      <td>1.274533</td>
      <td>0.287932</td>
      <td>0.000000</td>
      <td>1.000211</td>
      <td>3.584379</td>
      <td>0.094544</td>
      <td>2.053042</td>
      <td>7.408692</td>
    </tr>
    <tr>
      <th>2008.0</th>
      <td>9.491202</td>
      <td>23.217964</td>
      <td>19.930629</td>
      <td>2.853314</td>
      <td>0.000000</td>
      <td>8.705325</td>
      <td>2.482939</td>
      <td>3.233313</td>
      <td>10.944349</td>
      <td>0.521591</td>
      <td>0.726360</td>
      <td>1.517193</td>
      <td>0.420016</td>
      <td>0.021168</td>
      <td>0.709085</td>
      <td>5.447568</td>
      <td>0.151426</td>
      <td>2.443908</td>
      <td>7.512764</td>
    </tr>
    <tr>
      <th>2009.0</th>
      <td>6.319372</td>
      <td>23.634384</td>
      <td>21.251497</td>
      <td>1.674962</td>
      <td>0.000000</td>
      <td>4.432279</td>
      <td>6.489815</td>
      <td>4.257392</td>
      <td>9.133141</td>
      <td>0.555794</td>
      <td>0.679130</td>
      <td>1.084363</td>
      <td>0.741289</td>
      <td>0.000000</td>
      <td>0.342360</td>
      <td>8.524976</td>
      <td>0.000000</td>
      <td>2.230501</td>
      <td>6.176574</td>
    </tr>
    <tr>
      <th>2010.0</th>
      <td>7.469426</td>
      <td>12.080997</td>
      <td>20.438948</td>
      <td>2.887187</td>
      <td>0.000000</td>
      <td>6.299230</td>
      <td>1.479987</td>
      <td>4.543450</td>
      <td>6.950751</td>
      <td>0.508018</td>
      <td>1.479942</td>
      <td>0.925987</td>
      <td>1.472866</td>
      <td>0.026765</td>
      <td>0.476007</td>
      <td>7.547969</td>
      <td>0.257594</td>
      <td>2.460049</td>
      <td>5.358070</td>
    </tr>
    <tr>
      <th>2011.0</th>
      <td>8.983144</td>
      <td>11.808438</td>
      <td>21.245150</td>
      <td>1.826715</td>
      <td>0.000000</td>
      <td>5.367651</td>
      <td>2.829903</td>
      <td>4.094741</td>
      <td>4.352630</td>
      <td>0.555194</td>
      <td>1.299660</td>
      <td>1.584417</td>
      <td>1.755616</td>
      <td>0.040411</td>
      <td>0.441928</td>
      <td>2.950286</td>
      <td>0.108535</td>
      <td>1.807845</td>
      <td>5.846258</td>
    </tr>
    <tr>
      <th>2012.0</th>
      <td>5.472345</td>
      <td>10.795978</td>
      <td>19.531136</td>
      <td>4.022735</td>
      <td>0.000000</td>
      <td>10.705217</td>
      <td>4.001834</td>
      <td>6.821539</td>
      <td>6.001628</td>
      <td>0.949636</td>
      <td>1.334703</td>
      <td>2.591452</td>
      <td>2.087438</td>
      <td>0.478334</td>
      <td>0.797983</td>
      <td>1.438157</td>
      <td>0.090099</td>
      <td>1.334061</td>
      <td>7.594004</td>
    </tr>
    <tr>
      <th>2013.0</th>
      <td>7.409289</td>
      <td>13.361643</td>
      <td>20.968705</td>
      <td>3.565122</td>
      <td>0.000000</td>
      <td>7.144638</td>
      <td>4.078637</td>
      <td>7.927655</td>
      <td>8.110186</td>
      <td>0.942122</td>
      <td>0.572566</td>
      <td>2.838418</td>
      <td>0.669383</td>
      <td>0.156428</td>
      <td>0.360501</td>
      <td>3.407286</td>
      <td>0.029941</td>
      <td>2.259077</td>
      <td>7.358668</td>
    </tr>
    <tr>
      <th>2014.0</th>
      <td>15.898269</td>
      <td>10.276298</td>
      <td>18.212133</td>
      <td>4.100492</td>
      <td>0.000000</td>
      <td>5.763825</td>
      <td>6.746153</td>
      <td>9.303960</td>
      <td>5.040957</td>
      <td>0.859923</td>
      <td>1.051879</td>
      <td>4.078545</td>
      <td>1.050601</td>
      <td>0.075336</td>
      <td>0.112317</td>
      <td>2.007352</td>
      <td>0.030274</td>
      <td>3.059118</td>
      <td>8.963780</td>
    </tr>
    <tr>
      <th>2015.0</th>
      <td>9.033037</td>
      <td>9.494244</td>
      <td>21.172468</td>
      <td>4.952110</td>
      <td>0.000000</td>
      <td>6.133664</td>
      <td>5.131328</td>
      <td>9.949957</td>
      <td>9.104006</td>
      <td>0.361001</td>
      <td>1.831478</td>
      <td>0.588932</td>
      <td>2.425713</td>
      <td>0.093860</td>
      <td>0.486874</td>
      <td>1.880777</td>
      <td>0.282448</td>
      <td>1.662999</td>
      <td>6.711360</td>
    </tr>
    <tr>
      <th>2016.0</th>
      <td>13.745836</td>
      <td>11.936886</td>
      <td>26.667679</td>
      <td>3.565636</td>
      <td>0.000000</td>
      <td>7.964081</td>
      <td>5.436848</td>
      <td>9.675230</td>
      <td>14.495609</td>
      <td>0.744859</td>
      <td>3.276831</td>
      <td>2.309934</td>
      <td>4.542317</td>
      <td>0.029709</td>
      <td>0.931749</td>
      <td>1.934870</td>
      <td>0.168828</td>
      <td>1.357689</td>
      <td>5.481081</td>
    </tr>
    <tr>
      <th>2017.0</th>
      <td>11.129723</td>
      <td>17.749869</td>
      <td>22.004785</td>
      <td>4.523391</td>
      <td>0.000000</td>
      <td>6.884497</td>
      <td>5.851272</td>
      <td>7.781246</td>
      <td>16.312784</td>
      <td>0.989249</td>
      <td>1.969854</td>
      <td>4.310440</td>
      <td>2.484809</td>
      <td>0.024738</td>
      <td>0.956903</td>
      <td>5.586515</td>
      <td>0.234558</td>
      <td>1.740566</td>
      <td>12.673385</td>
    </tr>
    <tr>
      <th>2018.0</th>
      <td>10.852312</td>
      <td>15.809709</td>
      <td>19.193431</td>
      <td>4.402559</td>
      <td>0.000000</td>
      <td>6.562399</td>
      <td>4.701308</td>
      <td>4.676549</td>
      <td>14.571332</td>
      <td>0.913046</td>
      <td>1.222690</td>
      <td>4.943190</td>
      <td>1.510474</td>
      <td>0.000000</td>
      <td>0.916551</td>
      <td>9.002724</td>
      <td>0.188188</td>
      <td>2.150812</td>
      <td>12.310457</td>
    </tr>
    <tr>
      <th>2019.0</th>
      <td>6.327459</td>
      <td>8.656026</td>
      <td>19.478615</td>
      <td>5.428145</td>
      <td>0.000000</td>
      <td>12.366930</td>
      <td>5.020257</td>
      <td>7.830262</td>
      <td>12.141667</td>
      <td>1.236543</td>
      <td>1.288441</td>
      <td>4.186043</td>
      <td>1.369167</td>
      <td>0.097324</td>
      <td>0.895410</td>
      <td>11.436287</td>
      <td>0.137693</td>
      <td>2.010029</td>
      <td>19.157134</td>
    </tr>
    <tr>
      <th>2020.0</th>
      <td>6.778076</td>
      <td>4.594657</td>
      <td>17.310974</td>
      <td>9.901627</td>
      <td>0.020876</td>
      <td>9.236431</td>
      <td>3.547079</td>
      <td>14.660178</td>
      <td>7.256000</td>
      <td>1.297793</td>
      <td>2.828310</td>
      <td>2.083001</td>
      <td>2.636148</td>
      <td>0.713161</td>
      <td>0.993423</td>
      <td>8.399400</td>
      <td>0.141378</td>
      <td>2.252643</td>
      <td>12.113782</td>
    </tr>
    <tr>
      <th>2021.0</th>
      <td>4.835925</td>
      <td>7.007805</td>
      <td>15.351814</td>
      <td>4.828188</td>
      <td>0.018776</td>
      <td>10.562304</td>
      <td>2.094687</td>
      <td>11.521699</td>
      <td>10.431902</td>
      <td>0.825601</td>
      <td>0.933486</td>
      <td>2.807606</td>
      <td>2.405104</td>
      <td>0.327700</td>
      <td>1.243998</td>
      <td>7.681625</td>
      <td>0.078099</td>
      <td>1.076354</td>
      <td>13.019462</td>
    </tr>
    <tr>
      <th>2022.0</th>
      <td>34.155028</td>
      <td>14.385454</td>
      <td>19.183707</td>
      <td>7.179642</td>
      <td>0.147548</td>
      <td>9.032407</td>
      <td>5.364811</td>
      <td>10.479636</td>
      <td>10.895240</td>
      <td>1.009119</td>
      <td>1.899908</td>
      <td>1.739032</td>
      <td>2.096513</td>
      <td>0.117765</td>
      <td>1.961240</td>
      <td>7.069298</td>
      <td>0.057382</td>
      <td>1.851054</td>
      <td>10.859681</td>
    </tr>
    <tr>
      <th>2023.0</th>
      <td>20.345692</td>
      <td>14.130889</td>
      <td>19.852503</td>
      <td>4.909359</td>
      <td>0.615525</td>
      <td>7.603077</td>
      <td>4.677102</td>
      <td>8.425415</td>
      <td>13.143922</td>
      <td>1.284972</td>
      <td>1.217614</td>
      <td>4.644648</td>
      <td>3.342636</td>
      <td>0.027385</td>
      <td>1.322205</td>
      <td>11.822125</td>
      <td>0.074744</td>
      <td>1.208776</td>
      <td>8.847462</td>
    </tr>
    <tr>
      <th>2024.0</th>
      <td>26.541036</td>
      <td>19.604670</td>
      <td>29.041067</td>
      <td>4.716438</td>
      <td>0.362888</td>
      <td>12.408360</td>
      <td>6.239865</td>
      <td>9.467949</td>
      <td>18.746890</td>
      <td>2.110808</td>
      <td>1.568441</td>
      <td>2.983011</td>
      <td>5.283925</td>
      <td>0.205786</td>
      <td>0.703909</td>
      <td>11.638378</td>
      <td>0.217645</td>
      <td>1.030319</td>
      <td>10.359977</td>
    </tr>
    <tr>
      <th>2025.0</th>
      <td>16.539529</td>
      <td>37.537907</td>
      <td>35.062234</td>
      <td>7.652518</td>
      <td>0.192381</td>
      <td>16.436660</td>
      <td>4.481910</td>
      <td>11.206270</td>
      <td>15.227267</td>
      <td>1.142875</td>
      <td>1.780222</td>
      <td>2.972390</td>
      <td>3.521431</td>
      <td>0.293426</td>
      <td>1.105871</td>
      <td>10.158130</td>
      <td>0.190496</td>
      <td>0.900761</td>
      <td>14.276353</td>
    </tr>
  </tbody>
</table>
</div>




```python
somme = occurrence.loc[:, "guerre":"combattre"]
occurrence["somme"] = somme.sum(axis=1)
occurrence.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>année</th>
      <th>guerre</th>
      <th>paix</th>
      <th>sécurité</th>
      <th>mobilisation</th>
      <th>réarmement</th>
      <th>armée</th>
      <th>conflit</th>
      <th>mobiliser</th>
      <th>défense</th>
      <th>...</th>
      <th>attaquer</th>
      <th>bataille</th>
      <th>attaque</th>
      <th>riposte</th>
      <th>déploiement</th>
      <th>stratégique</th>
      <th>tactique</th>
      <th>combattre</th>
      <th>combat</th>
      <th>somme</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>2021.0</td>
      <td>4.835925</td>
      <td>7.007805</td>
      <td>15.351814</td>
      <td>4.828188</td>
      <td>0.018776</td>
      <td>10.562304</td>
      <td>2.094687</td>
      <td>11.521699</td>
      <td>10.431902</td>
      <td>...</td>
      <td>0.933486</td>
      <td>2.807606</td>
      <td>2.405104</td>
      <td>0.327700</td>
      <td>1.243998</td>
      <td>7.681625</td>
      <td>0.078099</td>
      <td>1.076354</td>
      <td>13.019462</td>
      <td>84.032673</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2022.0</td>
      <td>34.155028</td>
      <td>14.385454</td>
      <td>19.183707</td>
      <td>7.179642</td>
      <td>0.147548</td>
      <td>9.032407</td>
      <td>5.364811</td>
      <td>10.479636</td>
      <td>10.895240</td>
      <td>...</td>
      <td>1.899908</td>
      <td>1.739032</td>
      <td>2.096513</td>
      <td>0.117765</td>
      <td>1.961240</td>
      <td>7.069298</td>
      <td>0.057382</td>
      <td>1.851054</td>
      <td>10.859681</td>
      <td>128.624784</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2023.0</td>
      <td>20.345692</td>
      <td>14.130889</td>
      <td>19.852503</td>
      <td>4.909359</td>
      <td>0.615525</td>
      <td>7.603077</td>
      <td>4.677102</td>
      <td>8.425415</td>
      <td>13.143922</td>
      <td>...</td>
      <td>1.217614</td>
      <td>4.644648</td>
      <td>3.342636</td>
      <td>0.027385</td>
      <td>1.322205</td>
      <td>11.822125</td>
      <td>0.074744</td>
      <td>1.208776</td>
      <td>8.847462</td>
      <td>118.648588</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2024.0</td>
      <td>26.541036</td>
      <td>19.604670</td>
      <td>29.041067</td>
      <td>4.716438</td>
      <td>0.362888</td>
      <td>12.408360</td>
      <td>6.239865</td>
      <td>9.467949</td>
      <td>18.746890</td>
      <td>...</td>
      <td>1.568441</td>
      <td>2.983011</td>
      <td>5.283925</td>
      <td>0.205786</td>
      <td>0.703909</td>
      <td>11.638378</td>
      <td>0.217645</td>
      <td>1.030319</td>
      <td>10.359977</td>
      <td>152.871384</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2025.0</td>
      <td>16.539529</td>
      <td>37.537907</td>
      <td>35.062234</td>
      <td>7.652518</td>
      <td>0.192381</td>
      <td>16.436660</td>
      <td>4.481910</td>
      <td>11.206270</td>
      <td>15.227267</td>
      <td>...</td>
      <td>1.780222</td>
      <td>2.972390</td>
      <td>3.521431</td>
      <td>0.293426</td>
      <td>1.105871</td>
      <td>10.158130</td>
      <td>0.190496</td>
      <td>0.900761</td>
      <td>14.276353</td>
      <td>166.402279</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
```


```python
pearsonr(occurrence["année"], occurrence["somme"])
```




    PearsonRResult(statistic=np.float64(0.5002764967034047), pvalue=np.float64(0.010871554114235593))




```python
occurrence_économie = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_économie.csv")
occurrence_santé = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_santé.csv")
occurrence_éducation = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_éducation.csv")
occurrence_environnement = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_environnement.csv")
occurrence_technologie = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_technologie.csv")
occurrence_social = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_social.csv")
occurrence_démographie = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_démographie.csv")
occurrence_culture = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/comptage_10000_culture.csv")
```


```python
somme = occurrence_économie.loc[:, "économique":"profit"]
occurrence_économie["somme"] = somme.sum(axis=1)
occurrence_économie = occurrence_économie["somme"]

somme = occurrence_santé.loc[:, "santé":"médicament"]
occurrence_santé["somme"] = somme.sum(axis=1)
occurrence_santé = occurrence_santé["somme"]

somme = occurrence_éducation.loc[:, "éducation":"enseignement"]
occurrence_éducation["somme"] = somme.sum(axis=1)
occurrence_éducation = occurrence_éducation["somme"]

somme = occurrence_environnement.loc[:, "environnement":"naturel"]
occurrence_environnement["somme"] = somme.sum(axis=1)
occurrence_environnement = occurrence_environnement["somme"]

somme = occurrence_technologie.loc[:, "technologie":"ordinateur"]
occurrence_technologie["somme"] = somme.sum(axis=1)
occurrence_technologie = occurrence_technologie["somme"]

somme = occurrence_social.loc[:, "social":"âge"]
occurrence_social["somme"] = somme.sum(axis=1)
occurrence_social = occurrence_social["somme"]

somme = occurrence_démographie.loc[:, "démographie":"population"]
occurrence_démographie["somme"] = somme.sum(axis=1)
occurrence_démographie = occurrence_démographie["somme"]

somme = occurrence_culture.loc[:, "culture":"artistique"]
occurrence_culture["somme"] = somme.sum(axis=1)
occurrence_culture = occurrence_culture["somme"]
```


```python
from scipy.stats import f_oneway
```


```python
F, p = f_oneway(occurrence["somme"], occurrence_économie, occurrence_santé, occurrence_éducation, occurrence_environnement, occurrence_technologie, occurrence_social, occurrence_démographie, occurrence_culture)
print(f"F - value: {F}")
print(f"p - value: {p}")
```

    F - value: 459.39249300058776
    p - value: 4.623703622087847e-131
    

Source du code : https://coderivers.org/blog/analysis-of-variance-python/


```python
import seaborn as sns
sns.set_style("ticks")
sns.set_context("paper")
sns.set_palette("deep")
sns.despine(top=False, right=False, left=False, bottom=False)
```


    <Figure size 640x480 with 0 Axes>



```python
occurrence["somme"].tail()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[18], line 1
    ----> 1 occurrence["somme"].tail()
    

    NameError: name 'occurrence' is not defined



```python
anova = pd.DataFrame()
```


```python
anova = pd.concat([occurrence_économie, occurrence_santé, occurrence_éducation, occurrence_environnement, occurrence_technologie, occurrence_social, occurrence_démographie, occurrence_culture], axis=1)
anova["occurrence_guerre"] = occurrence["somme"]
anova.columns = ["économie", "santé", "éducation", "environnement", "technologie", "social", "démographie", "culture", "guerre"]
```


```python
anova.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>économie</th>
      <th>santé</th>
      <th>éducation</th>
      <th>environnement</th>
      <th>technologie</th>
      <th>social</th>
      <th>démographie</th>
      <th>culture</th>
      <th>guerre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.044124</td>
      <td>2.470004</td>
      <td>1.402047</td>
      <td>7.457597</td>
      <td>0.669298</td>
      <td>7.859759</td>
      <td>3.673536</td>
      <td>6.335241</td>
      <td>9.397732</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.088575</td>
      <td>2.682064</td>
      <td>1.090139</td>
      <td>6.841094</td>
      <td>0.885986</td>
      <td>8.071778</td>
      <td>2.928849</td>
      <td>6.906846</td>
      <td>8.382708</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.810268</td>
      <td>2.936079</td>
      <td>1.114217</td>
      <td>6.172740</td>
      <td>0.730027</td>
      <td>6.884199</td>
      <td>2.972164</td>
      <td>7.425104</td>
      <td>7.978059</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.675582</td>
      <td>2.595627</td>
      <td>1.316815</td>
      <td>5.075336</td>
      <td>0.776974</td>
      <td>7.346281</td>
      <td>3.226267</td>
      <td>8.442457</td>
      <td>8.160736</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.922860</td>
      <td>1.813602</td>
      <td>1.260846</td>
      <td>6.043830</td>
      <td>1.539290</td>
      <td>6.596800</td>
      <td>2.510904</td>
      <td>7.921713</td>
      <td>7.590420</td>
    </tr>
  </tbody>
</table>
</div>




```python
F, p = f_oneway(anova["économie"], anova["santé"], anova["éducation"], anova["environnement"], anova["technologie"], anova["social"], anova["démographie"], anova["culture"], anova["guerre"])
print(f"F - value: {F}")
print(f"p - value: {p}")
```

    F - value: 91.42808743266353
    p - value: 4.640634845584451e-65
    


```python
thèmes = ["économie", "santé", "éducation", "environnement", "technologie", "social", "démographie", "culture", "guerre"]
data_long = pd.DataFrame()
```


```python
for thème in thèmes:
    # Créer un dataframe temporaire pour ce thème
    temp = pd.DataFrame({
        "Thématique": thème,
        "Occurrence pour 10 000 mots": anova[thème]  # vos valeurs
    })
    # Ajouter au dataframe principal
    data_long = pd.concat([data_long, temp], ignore_index=True)
```


```python
data_long.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Thématique</th>
      <th>Occurrence pour 10 000 mots</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>économie</td>
      <td>4.044124</td>
    </tr>
    <tr>
      <th>1</th>
      <td>économie</td>
      <td>6.088575</td>
    </tr>
    <tr>
      <th>2</th>
      <td>économie</td>
      <td>6.810268</td>
    </tr>
    <tr>
      <th>3</th>
      <td>économie</td>
      <td>7.675582</td>
    </tr>
    <tr>
      <th>4</th>
      <td>économie</td>
      <td>7.922860</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = sns.catplot(data= data_long, x="Occurrence pour 10 000 mots", y="Thématique", kind="box", order=data_long.sort_values("Occurrence pour 10 000 mots").Thématique)
fig.figure.savefig("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/anova.png", dpi=450)
```


    
![png](output_136_0.png)
    



```python
# Kruskal-Wallis (équivalent non-paramétrique de l'ANOVA)
from scipy.stats import kruskal

h_stat, p_kw = kruskal(*[data_long[data_long['Thématique']==t]['Occurrence pour 10 000 mots'] 
                          for t in data_long['Thématique'].unique()])
print(f"Kruskal-Wallis H: {h_stat:.2f}, p={p_kw:.4f}")
```

    Kruskal-Wallis H: 178.48, p=0.0000
    

### Score à travers le temps


```python
score = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/score.csv")
```


```python
score.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>année</th>
      <th>guerre</th>
      <th>paix</th>
      <th>sécurité</th>
      <th>mobilisation</th>
      <th>réarmement</th>
      <th>armée</th>
      <th>conflit</th>
      <th>menace</th>
      <th>mobiliser</th>
      <th>...</th>
      <th>attaquer</th>
      <th>bataille</th>
      <th>attaque</th>
      <th>riposte</th>
      <th>déploiement</th>
      <th>stratégique</th>
      <th>tactique</th>
      <th>combattre</th>
      <th>combat</th>
      <th>tfidf_moyen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001.0</td>
      <td>0.009441</td>
      <td>0.015928</td>
      <td>0.012192</td>
      <td>0.003755</td>
      <td>0.000000</td>
      <td>0.008290</td>
      <td>0.009936</td>
      <td>0.004945</td>
      <td>0.004784</td>
      <td>...</td>
      <td>0.004381</td>
      <td>0.002680</td>
      <td>0.008800</td>
      <td>0.009841</td>
      <td>0.000817</td>
      <td>0.001699</td>
      <td>0.000000</td>
      <td>0.006447</td>
      <td>0.014442</td>
      <td>0.006485</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2002.0</td>
      <td>0.005930</td>
      <td>0.011263</td>
      <td>0.016065</td>
      <td>0.005864</td>
      <td>0.000397</td>
      <td>0.012207</td>
      <td>0.005018</td>
      <td>0.004470</td>
      <td>0.005379</td>
      <td>...</td>
      <td>0.001373</td>
      <td>0.001551</td>
      <td>0.002786</td>
      <td>0.000192</td>
      <td>0.001040</td>
      <td>0.002453</td>
      <td>0.000213</td>
      <td>0.004971</td>
      <td>0.007746</td>
      <td>0.005050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003.0</td>
      <td>0.008580</td>
      <td>0.015056</td>
      <td>0.011098</td>
      <td>0.006222</td>
      <td>0.000000</td>
      <td>0.003748</td>
      <td>0.007488</td>
      <td>0.003837</td>
      <td>0.004936</td>
      <td>...</td>
      <td>0.001608</td>
      <td>0.000380</td>
      <td>0.000402</td>
      <td>0.000320</td>
      <td>0.001166</td>
      <td>0.002134</td>
      <td>0.000865</td>
      <td>0.002034</td>
      <td>0.005139</td>
      <td>0.004280</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004.0</td>
      <td>0.004653</td>
      <td>0.017516</td>
      <td>0.009655</td>
      <td>0.005823</td>
      <td>0.000000</td>
      <td>0.008002</td>
      <td>0.005877</td>
      <td>0.004315</td>
      <td>0.006424</td>
      <td>...</td>
      <td>0.001932</td>
      <td>0.001707</td>
      <td>0.000520</td>
      <td>0.000000</td>
      <td>0.000743</td>
      <td>0.003820</td>
      <td>0.000166</td>
      <td>0.003276</td>
      <td>0.011101</td>
      <td>0.004650</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005.0</td>
      <td>0.003201</td>
      <td>0.015836</td>
      <td>0.008086</td>
      <td>0.006194</td>
      <td>0.000000</td>
      <td>0.005342</td>
      <td>0.004968</td>
      <td>0.002607</td>
      <td>0.006016</td>
      <td>...</td>
      <td>0.000916</td>
      <td>0.001572</td>
      <td>0.000536</td>
      <td>0.000000</td>
      <td>0.000316</td>
      <td>0.004237</td>
      <td>0.000333</td>
      <td>0.004742</td>
      <td>0.005427</td>
      <td>0.003987</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
from scipy.stats import pearsonr
pearsonr(score["année"], score["tfidf_moyen"])
```




    PearsonRResult(statistic=np.float64(0.697085537743082), pvalue=np.float64(0.00010789039706677585))



Source : https://medium.com/data-science/getting-started-with-breakpoints-analysis-in-python-124471708d38


```python
score.set_index(score["année"], inplace = True)
ts = score["tfidf_moyen"]
```


```python
import ruptures as rpt
n_breaks = 5
y = np.array(ts.tolist())
model = rpt.Dynp(model="l1")
model.fit(y)
breaks = model.predict(n_bkps=n_breaks-1)
breaks_rpt = []
for i in breaks:
    breaks_rpt.append(ts.index[i-1])
breaks_rpt
```




    [np.float64(2005.0),
     np.float64(2010.0),
     np.float64(2015.0),
     np.float64(2020.0),
     np.float64(2025.0)]




```python
tableau_corrélation_tfidf = pd.DataFrame()
```


```python
tableau_corrélation_tfidf["guerre"] = pearsonr(score["année"], score["guerre"])
tableau_corrélation_tfidf["paix"] = pearsonr(score["année"], score["paix"])
tableau_corrélation_tfidf["sécurité"] = pearsonr(score["année"], score["sécurité"])
tableau_corrélation_tfidf["mobilisation"] = pearsonr(score["année"], score["mobilisation"])
tableau_corrélation_tfidf["réarmement"] = pearsonr(score["année"], score["réarmement"])
tableau_corrélation_tfidf["armée"] = pearsonr(score["année"], score["armée"])
tableau_corrélation_tfidf["conflit"] = pearsonr(score["année"], score["conflit"])
tableau_corrélation_tfidf["menace"] = pearsonr(score["année"], score["menace"])
tableau_corrélation_tfidf["mobiliser"] = pearsonr(score["année"], score["mobiliser"])
tableau_corrélation_tfidf["défense"] = pearsonr(score["année"], score["défense"])
tableau_corrélation_tfidf["front"] = pearsonr(score["année"], score["front"])
tableau_corrélation_tfidf["attaquer"] = pearsonr(score["année"], score["attaquer"])
tableau_corrélation_tfidf["bataille"] = pearsonr(score["année"], score["bataille"])
tableau_corrélation_tfidf["attaque"] = pearsonr(score["année"], score["attaque"])
tableau_corrélation_tfidf["riposte"] = pearsonr(score["année"], score["riposte"])
tableau_corrélation_tfidf["déploiement"] = pearsonr(score["année"], score["déploiement"])
tableau_corrélation_tfidf["stratégique"] = pearsonr(score["année"], score["stratégique"])
tableau_corrélation_tfidf["tactique"] = pearsonr(score["année"], score["tactique"])
tableau_corrélation_tfidf["combattre"] = pearsonr(score["année"], score["combattre"])
tableau_corrélation_tfidf["combat"] = pearsonr(score["année"], score["combat"])
```


```python
tableau_corrélation_tfidf = tableau_corrélation_tfidf.sort_values(axis=1, by=0, ascending=False)
tableau_corrélation_tfidf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>armée</th>
      <th>stratégique</th>
      <th>mobiliser</th>
      <th>bataille</th>
      <th>déploiement</th>
      <th>défense</th>
      <th>guerre</th>
      <th>réarmement</th>
      <th>attaque</th>
      <th>mobilisation</th>
      <th>front</th>
      <th>attaquer</th>
      <th>combat</th>
      <th>tactique</th>
      <th>menace</th>
      <th>sécurité</th>
      <th>conflit</th>
      <th>paix</th>
      <th>riposte</th>
      <th>combattre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.751435</td>
      <td>0.728521</td>
      <td>0.715934</td>
      <td>0.695910</td>
      <td>0.679067</td>
      <td>0.576548</td>
      <td>0.557804</td>
      <td>0.556890</td>
      <td>0.554789</td>
      <td>0.529114</td>
      <td>0.458242</td>
      <td>0.427351</td>
      <td>0.375741</td>
      <td>0.324402</td>
      <td>0.253732</td>
      <td>0.114251</td>
      <td>0.005583</td>
      <td>-0.201641</td>
      <td>-0.214329</td>
      <td>-0.577927</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000015</td>
      <td>0.000036</td>
      <td>0.000057</td>
      <td>0.000112</td>
      <td>0.000190</td>
      <td>0.002555</td>
      <td>0.003764</td>
      <td>0.003834</td>
      <td>0.003998</td>
      <td>0.006535</td>
      <td>0.021240</td>
      <td>0.033107</td>
      <td>0.064168</td>
      <td>0.113627</td>
      <td>0.221005</td>
      <td>0.586583</td>
      <td>0.978870</td>
      <td>0.333762</td>
      <td>0.303573</td>
      <td>0.002481</td>
    </tr>
  </tbody>
</table>
</div>




```python
tableau_corrélation_tfidf.to_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/tableau_corrélation_tfidf.csv")
```


```python
tableau_corrélation_tfidf.to_clipboard()
```

### Cooccurrence à travers le temps


```python
cooccurrence = pd.read_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/cooccurrences.csv")
```


```python
cooccurrence.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>année</th>
      <th>année.1</th>
      <th>économie</th>
      <th>santé</th>
      <th>éducation</th>
      <th>environnement</th>
      <th>technologie</th>
      <th>social</th>
      <th>démographie</th>
      <th>culture</th>
      <th>somme</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>2021</td>
      <td>2021</td>
      <td>1.357724</td>
      <td>1.081301</td>
      <td>0.406504</td>
      <td>0.707317</td>
      <td>0.666667</td>
      <td>1.097561</td>
      <td>0.447154</td>
      <td>1.227642</td>
      <td>6.991870</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2022</td>
      <td>2022</td>
      <td>1.396396</td>
      <td>0.567568</td>
      <td>0.378378</td>
      <td>0.918919</td>
      <td>0.810811</td>
      <td>1.378378</td>
      <td>0.513514</td>
      <td>1.540541</td>
      <td>7.504505</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2023</td>
      <td>2023</td>
      <td>1.564706</td>
      <td>0.370588</td>
      <td>0.300000</td>
      <td>0.858824</td>
      <td>0.782353</td>
      <td>0.811765</td>
      <td>0.576471</td>
      <td>1.005882</td>
      <td>6.270588</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2024</td>
      <td>2024</td>
      <td>2.040323</td>
      <td>0.274194</td>
      <td>0.233871</td>
      <td>0.838710</td>
      <td>1.193548</td>
      <td>0.782258</td>
      <td>0.596774</td>
      <td>0.975806</td>
      <td>6.935484</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2025</td>
      <td>2025</td>
      <td>2.979592</td>
      <td>0.680272</td>
      <td>0.469388</td>
      <td>1.292517</td>
      <td>1.108844</td>
      <td>1.625850</td>
      <td>0.768707</td>
      <td>1.333333</td>
      <td>10.258503</td>
    </tr>
  </tbody>
</table>
</div>




```python
somme = cooccurrence.loc[:, "économie":"culture"]
cooccurrence["somme"] = somme.sum(axis=1)
cooccurrence.to_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/cooccurrences.csv")
cooccurrence.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>année</th>
      <th>économie</th>
      <th>santé</th>
      <th>éducation</th>
      <th>environnement</th>
      <th>technologie</th>
      <th>social</th>
      <th>démographie</th>
      <th>culture</th>
      <th>somme</th>
    </tr>
    <tr>
      <th>année</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021</th>
      <td>2021</td>
      <td>1.357724</td>
      <td>1.081301</td>
      <td>0.406504</td>
      <td>0.707317</td>
      <td>0.666667</td>
      <td>1.097561</td>
      <td>0.447154</td>
      <td>1.227642</td>
      <td>6.991870</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>2022</td>
      <td>1.396396</td>
      <td>0.567568</td>
      <td>0.378378</td>
      <td>0.918919</td>
      <td>0.810811</td>
      <td>1.378378</td>
      <td>0.513514</td>
      <td>1.540541</td>
      <td>7.504505</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>2023</td>
      <td>1.564706</td>
      <td>0.370588</td>
      <td>0.300000</td>
      <td>0.858824</td>
      <td>0.782353</td>
      <td>0.811765</td>
      <td>0.576471</td>
      <td>1.005882</td>
      <td>6.270588</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>2024</td>
      <td>2.040323</td>
      <td>0.274194</td>
      <td>0.233871</td>
      <td>0.838710</td>
      <td>1.193548</td>
      <td>0.782258</td>
      <td>0.596774</td>
      <td>0.975806</td>
      <td>6.935484</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>2025</td>
      <td>2.979592</td>
      <td>0.680272</td>
      <td>0.469388</td>
      <td>1.292517</td>
      <td>1.108844</td>
      <td>1.625850</td>
      <td>0.768707</td>
      <td>1.333333</td>
      <td>10.258503</td>
    </tr>
  </tbody>
</table>
</div>




```python
pearsonr(cooccurrence["année"], cooccurrence["somme"])
```




    PearsonRResult(statistic=np.float64(0.7457006771010537), pvalue=np.float64(1.8833908606278616e-05))




```python
F, p = f_oneway(cooccurrence["économie"], cooccurrence["santé"], cooccurrence["éducation"], cooccurrence["environnement"], cooccurrence["technologie"], cooccurrence["social"], cooccurrence["démographie"], cooccurrence["culture"])
print(f"F - value: {F}")
print(f"p - value: {p}")
```

    F - value: 40.813305970367736
    p - value: 7.974721005359653e-35
    


```python
thèmes = ["économie", "santé", "éducation", "environnement", "technologie", "social", "démographie", "culture"]
data_long = pd.DataFrame()
```


```python
anova = pd.concat([occurrence_économie, occurrence_santé, occurrence_éducation, occurrence_environnement, occurrence_technologie, occurrence_social, occurrence_démographie, occurrence_culture], axis=1)
anova["occurrence_guerre"] = occurrence["somme"]
anova.columns = ["économie", "santé", "éducation", "environnement", "technologie", "social", "démographie", "culture", "guerre"]
```


```python
for thème in thèmes:
    # Créer un dataframe temporaire pour ce thème
    temp = pd.DataFrame({
        "Thématique": thème,
        "Cooccurrence": cooccurrence[thème]  # vos valeurs
    })
    # Ajouter au dataframe principal
    data_long = pd.concat([data_long, temp], ignore_index=True)
```


```python
# Kruskal-Wallis (équivalent non-paramétrique de l'ANOVA)
from scipy.stats import kruskal
h_stat, p_kw = kruskal(*[data_long[data_long["Thématique"]==t]["Cooccurrence"] 
                          for t in data_long['Thématique'].unique()])
print(f"Kruskal-Wallis H: {h_stat:.2f}, p={p_kw:.4f}")
```

    Kruskal-Wallis H: 127.46, p=0.0000
    


```python
fig = sns.catplot(data= data_long, x="Cooccurrence", y="Thématique", kind="box", order=data_long.sort_values("Cooccurrence").Thématique)
fig.figure.savefig("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/anova_cooccurrence.png", dpi=450)
```


    
![png](output_160_0.png)
    



```python
cooccurrence.set_index(cooccurrence["année"], inplace = True)
ts = cooccurrence["somme"]
```


```python
n_breaks = 3
y = np.array(ts.tolist())
```


```python
import ruptures as rpt
model = rpt.Dynp(model="l1")
model.fit(y)
breaks = model.predict(n_bkps=n_breaks-1)
```


```python
breaks_rpt = []
for i in breaks:
    breaks_rpt.append(ts.index[i-1])
breaks_rpt
```




    [np.int64(2005), np.int64(2015), np.int64(2025)]




```python
tableau_corrélation_cooccurrence = pd.DataFrame()
```


```python
tableau_corrélation_cooccurrence["économie"] = pearsonr(cooccurrence["année"], cooccurrence["économie"])
tableau_corrélation_cooccurrence["santé"] = pearsonr(cooccurrence["année"], cooccurrence["santé"])
tableau_corrélation_cooccurrence["éducation"] = pearsonr(cooccurrence["année"], cooccurrence["éducation"])
tableau_corrélation_cooccurrence["environnement"] = pearsonr(cooccurrence["année"], cooccurrence["environnement"])
tableau_corrélation_cooccurrence["technologie"] = pearsonr(cooccurrence["année"], cooccurrence["technologie"])
tableau_corrélation_cooccurrence["social"] = pearsonr(cooccurrence["année"], cooccurrence["social"])
tableau_corrélation_cooccurrence["démographie"] = pearsonr(cooccurrence["année"], cooccurrence["démographie"])
tableau_corrélation_cooccurrence["culture"] = pearsonr(cooccurrence["année"], cooccurrence["culture"])
```


```python
tableau_corrélation_cooccurrence = tableau_corrélation_cooccurrence.sort_values(axis=1, by=0, ascending=False)
tableau_corrélation_cooccurrence.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>technologie</th>
      <th>démographie</th>
      <th>éducation</th>
      <th>économie</th>
      <th>environnement</th>
      <th>santé</th>
      <th>culture</th>
      <th>social</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.650931e-01</td>
      <td>0.794482</td>
      <td>0.702818</td>
      <td>0.668205</td>
      <td>0.663927</td>
      <td>0.509440</td>
      <td>0.440385</td>
      <td>0.272876</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.410298e-08</td>
      <td>0.000002</td>
      <td>0.000089</td>
      <td>0.000262</td>
      <td>0.000296</td>
      <td>0.009291</td>
      <td>0.027581</td>
      <td>0.186922</td>
    </tr>
  </tbody>
</table>
</div>




```python
tableau_corrélation_cooccurrence.to_csv("C:/Users/gsprd/Documents/(2026-2027) - Orientation & Formations/Python/Données/tableau_corrélation_cooccurrence.csv")
```


```python
tableau_corrélation_cooccurrence.to_clipboard()
```
