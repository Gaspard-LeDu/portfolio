##################################################################################
# ENQUETE ISSP 2013 - EXPLIQUER LE VOTE FN/RN CHEZ LES DESCENDANT•E•S D'IMMIGRES #
##################################################################################

# -----------------------
# Préparation des données
# -----------------------


##### Activations des bibliothèques

library(foreign)
library(questionr)
library(tidyverse)
library(survey)
library(clipr)
library(FactoMineR)
library(explor)
library(DescTools)
library(GGally)
library(dplyr)
library(nnet)
library(ggstats)
library(ade4)
library(cluster)
library(RColorBrewer)

##### Importation de la base de donnée & sélection de l'échantillon

# Importation de la BDD, sélection des répondants français
DATA <- read.spss("C:/Users/gsprd/Documents/(2025-2026) - Sciences Po Rennes - M2 RESSP/(2) Cours/(Benoit Giry) Méthodes quantitatives/ISSP (2013)/DATAM2.sav", to.data.frame = TRUE)
DATA_FR <- subset (DATA, V3 == "FR-France")
#Pondération
DATA_FR$WEIGHT <- as.numeric(as.character(DATA_FR$WEIGHT))
DATA_W <- svydesign(ids = ~1, data = DATA_FR, weights = ~ DATA_FR$WEIGHT)

# Sélection des descendant•e•s d'immigrés (DI) parmi l'échantillon français
DATA_I <- DATA_FR %>%
  filter(V64 %in% c("Neither parent was a citizen", 
                    "Only mother was a citizen", 
                    "Only father was a citizen"))
# Pondération
DATA_I$WEIGHT <- as.numeric(as.character(DATA_I$WEIGHT))
DATA_IW <- svydesign(ids = ~1, data = DATA_I, weights = ~ DATA_I$WEIGHT)

# ----------------------
# 1) Analyse descriptive
# ----------------------

# Nombre de DI
summary(DATA_FR$V64)
# n = 229 individus descendant•e•s d'immigrés

# Tableau pondéré du nombre de DI
write_clip(freq(svytable(~V64, DATA_W)))
# DI = 11,8% de la population ISSP 2013 totale

### Origine géographique des DI

# Pour l'origine du père (variable par défaut chez l'INSEE)
irec(DATA_I$F_BORN)
# Recodage de DATA_I$F_BORN en DATA_I$F_BORN_desc
DATA_I$F_BORN_desc <- DATA_I$F_BORN |>
  fct_recode(
    "Asie" = "AF-Afghanistan",
    "Europe" = "AL-Albania",
    "Afrique" = "DZ-Algeria",
    "Afrique" = "AO-Angola",
    "Asie" = "AZ-Azerbaijan",
    "Amérique & Océanie" = "AR-Argentina",
    "Amérique & Océanie" = "AU-Australia",
    "Europe" = "AT-Austria",
    "Asie" = "BD-Bangladesh",
    "Asie" = "AM-Armenia",
    "Europe" = "BE-Belgium",
    "Amérique & Océanie" = "BO-Bolivia",
    "Europe" = "BA-Bosnia and Herzegovina",
    "Afrique" = "BR-Botswana",
    "Amérique & Océanie" = "BR-Brazil",
    "Europe" = "BG-Bulgaria",
    "Asie" = "MM-Myanmar",
    "Afrique" = "BI-Burundi",
    "Europe" = "BY-Belarus",
    "Asie" = "KH-Camboya",
    "Afrique" = "CM-Camerún",
    "Amérique & Océanie" = "CA-Canada",
    "Afrique" = "CV-Cabo Verde",
    "Asie" = "LK-Sri Lanka",
    "Amérique & Océanie" = "CL-Chile",
    "Asie" = "CN-China",
    "Asie" = "TW-Taiwan",
    "Amérique & Océanie" = "CO-Colombia",
    "Afrique" = "KM-Comoras",
    "Afrique" = "CG-Congo (Republic of)",
    "Afrique" = "CD-Congo (Democratic Republic of the)",
    "Europe" = "HR-Croatia",
    "Amérique & Océanie" = "CU-Cuba",
    "Europe" = "CY-Cyprus",
    "Europe" = "CS-Czechoslovakia",
    "Europe" = "CZ-Czech Republic",
    "Europe" = "DK-Denmark",
    "Amérique & Océanie" = "DO-Dominican Republic",
    "Amérique & Océanie" = "EC-Ecuador",
    "Amérique & Océanie" = "SC-El Salvador",
    "Afrique" = "ET-Ethiopia",
    "Europe" = "EE-Estonia",
    "Europe" = "FO-Faroe Islands (the)",
    "Europe" = "FI-Finland",
    "Europe" = "FR-France",
    "Afrique" = "DJ-Djibouty (the Republic of)",
    "Asie" = "GE-Georgia",
    "Asie" = "PS-Palestine, State of",
    "Europe" = "DE-Germany",
    "Afrique" = "GH-Ghana",
    "Europe" = "GR-Greece",
    "Amérique & Océanie" = "GP-Guadalupe",
    "Amérique & Océanie" = "GT-Guatemala",
    "Afrique" = "GN-Ghinea",
    "Amérique & Océanie" = "HT-Haití",
    "Amérique & Océanie" = "HN-Honduras",
    "Asie" = "HK-Hong Kong",
    "Europe" = "HU-Hungary",
    "Europe" = "IS-Iceland",
    "Asie" = "IN-India",
    "Asie" = "ID-Indonesia",
    "Asie" = "IR-Iran",
    "Asie" = "IQ-Iraq",
    "Europe" = "IE-Ireland",
    "Asie" = "IL-Israel",
    "Europe" = "IT-Italy",
    "Afrique" = "CI-Cote d'Ivoire",
    "Asie" = "JP-Japan",
    "Asie" = "KZ-Kazakhstan (the Republic of)",
    "Asie" = "JO-Jordan",
    "Afrique" = "KE-Kenya (the Republic of)",
    "Asie" = "KR-Korea (South)",
    "Asie" = "KG-Kyrgyzstan (the Republic of)",
    "Asie" = "LA-Lao",
    "Asie" = "LB-Lebanon",
    "Europe" = "LV-Latvia",
    "Afrique" = "LY-Libya",
    "Europe" = "LI-Liechtenstein (the Principality of )",
    "Europe" = "LT-Lithuania",
    "Afrique" = "MG-Madagascar",
    "Asie" = "MY-Malaysia",
    "Afrique" = "ML-Mali",
    "Amérique & Océanie" = "MQ-Martinica",
    "Afrique" = "MR-Mauritania",
    "Afrique" = "MU-Mauricio",
    "Amérique & Océanie" = "MX-Mexico",
    "Europe" = "MC-Monaco",
    "Europe" = "MD-Moldova (the Republic of)",
    "Europe" = "ME-Montenegro",
    "Afrique" = "MA-Morocco",
    "Afrique" = "MZ-Mozambique",
    "Asie" = "NP-Nepal",
    "Europe" = "NL-Netherlands",
    "Amérique & Océanie" = "NC-Nueva Caledonia",
    "Amérique & Océanie" = "NZ-New Zealand",
    "Afrique" = "NE-Niger",
    "Afrique" = "NG-Nigeria",
    "Europe" = "NO-Norway",
    "Asie" = "PK-Pakistan",
    "Amérique & Océanie" = "PY-Paraguay",
    "Amérique & Océanie" = "PE-Peru",
    "Asie" = "PH-Philippines",
    "Europe" = "PL-Poland",
    "Europe" = "PT-Portugal",
    "Afrique" = "GW-Guinea-Bissau",
    "Amérique & Océanie" = "PR-Puerto Rico",
    "Afrique" = "RE-Reunión",
    "Europe" = "RO-Romania",
    "Europe" = "RU-Russia",
    "Afrique" = "RW-Rwanda",
    "Afrique" = "ST-Sao Tome and Principe",
    "Afrique" = "SN-Senegal",
    "Europe" = "RS-Serbia",
    "Afrique" = "SC-Seychelles",
    "Afrique" = "SL-Sierra Leone (the Republic of)",
    "Asie" = "SK-Singapore (the Republic of)",
    "Europe" = "SK-Slovak Republic",
    "Asie" = "VN-Vietnam",
    "Europe" = "SI-Slovenia",
    "Afrique" = "SO-Somalia",
    "Afrique" = "ZA-South Africa",
    "Europe" = "ES-Spain",
    "Afrique" = "SS-South Sudan",
    "Afrique" = "SD-Sudan",
    "Amérique & Océanie" = "SR-Suriname",
    "Europe" = "SE-Sweden",
    "Europe" = "CH-Switzerland",
    "Asie" = "SY-Syrian Arab Republic",
    "Asie" = "TJ-Takijistan (the Republic of)",
    "Asie" = "TH-Thailand",
    "Amérique & Océanie" = "TT-Trinidad y Tobago",
    "Afrique" = "TN-Tunisia",
    "Asie" = "TR-Turkey",
    "Asie" = "TR-Turkmenistan",
    "Europe" = "UA-Ukraine",
    "Europe" = "MK-Macedonia (the former Yugoslav Republic of)",
    "Europe" = "SU-USSR",
    "Asie" = "EG-Egypt",
    "Europe" = "GB-Great Britain and/or United Kingdom",
    "Europe" = "JE-Jersey",
    "Afrique" = "TZ-Tanzania (the United Republic of)",
    "Amérique & Océanie" = "US-United States",
    "Afrique" = "BF-Furkina Faso",
    "Amérique & Océanie" = "UY-Uruguay",
    "Asie" = "UZ-Uzbekistan (the Republic of)",
    "Amérique & Océanie" = "VE-Venezuela",
    "Asie" = "YE-Yemen",
    "Europe" = "YU-Yugoslavia",
    "Afrique" = "ZM-Zambia",
    "Europe" = "XK-Kosovo",
    NULL = "Other (CH,EE,JP,RU,TR,US)"
  )

# Pour l'origine de la mère (pour inclure les personnes dont seule la mère est étrangère)
irec(DATA_I$M_BORN)
# Recodage de DATA_I$M_BORN en DATA_I$M_BORN_desc
DATA_I$M_BORN_desc <- DATA_I$M_BORN |>
  fct_recode(
    "Asie" = "AF-Afghanistan",
    "Europe" = "AL-Albania",
    "Afrique" = "DZ-Algeria",
    "Afrique" = "AO-Angola",
    "Asie" = "AZ-Azerbaijan",
    "Amérique & Océanie" = "AR-Argentina",
    "Amérique & Océanie" = "AU-Australia",
    "Europe" = "AT-Austria",
    "Asie" = "BD-Bangladesh",
    "Asie" = "AM-Armenia",
    "Europe" = "BE-Belgium",
    "Amérique & Océanie" = "BO-Bolivia",
    "Europe" = "BA-Bosnia and Herzegovina",
    "Afrique" = "BR-Botswana",
    "Amérique & Océanie" = "BR-Brazil",
    "Europe" = "BG-Bulgaria",
    "Asie" = "MM-Myanmar",
    "Afrique" = "BI-Burundi",
    "Europe" = "BY-Belarus",
    "Asie" = "KH-Camboya",
    "Afrique" = "CM-Camerún",
    "Amérique & Océanie" = "CA-Canada",
    "Afrique" = "CV-Cabo Verde",
    "Asie" = "LK-Sri Lanka",
    "Amérique & Océanie" = "CL-Chile",
    "Asie" = "CN-China",
    "Asie" = "TW-Taiwan",
    "Amérique & Océanie" = "CO-Colombia",
    "Afrique" = "KM-Comoras",
    "Afrique" = "CG-Congo (Republic of)",
    "Afrique" = "CD-Congo (Democratic Republic of the)",
    "Europe" = "HR-Croatia",
    "Amérique & Océanie" = "CU-Cuba",
    "Europe" = "CY-Cyprus",
    "Europe" = "CS-Czechoslovakia",
    "Europe" = "CZ-Czech Republic",
    "Europe" = "DK-Denmark",
    "Amérique & Océanie" = "DO-Dominican Republic",
    "Amérique & Océanie" = "EC-Ecuador",
    "Amérique & Océanie" = "SC-El Salvador",
    "Afrique" = "ET-Ethiopia",
    "Europe" = "EE-Estonia",
    "Europe" = "FO-Faroe Islands (the)",
    "Europe" = "FI-Finland",
    "Europe" = "FR-France",
    "Afrique" = "DJ-Djibouty (the Republic of)",
    "Asie" = "GE-Georgia",
    "Asie" = "PS-Palestine, State of",
    "Europe" = "DE-Germany",
    "Afrique" = "GH-Ghana",
    "Europe" = "GR-Greece",
    "Europe" = "GL-Greenland",
    "Amérique & Océanie" = "GP-Guadalupe",
    "Afrique" = "GN-Ghinea",
    "Amérique & Océanie" = "HT-Haití",
    "Amérique & Océanie" = "HN-Honduras",
    "Asie" = "HK-Hong Kong",
    "Europe" = "HU-Hungary",
    "Europe" = "IS-Iceland",
    "Asie" = "IN-India",
    "Asie" = "ID-Indonesia",
    "Asie" = "IR-Iran",
    "Asie" = "IQ-Iraq",
    "Europe" = "IE-Ireland",
    "Asie" = "IL-Israel",
    "Europe" = "IT-Italy",
    "Afrique" = "CI-Cote d'Ivoire",
    "Asie" = "JP-Japan",
    "Asie" = "KZ-Kazakhstan (the Republic of)",
    "Asie" = "JO-Jordan",
    "Afrique" = "KE-Kenya (the Republic of)",
    "Asie" = "KR-Korea (South)",
    "Asie" = "KG-Kyrgyzstan (the Republic of)",
    "Asie" = "LA-Lao",
    "Asie" = "LB-Lebanon",
    "Europe" = "LV-Latvia",
    "Afrique" = "LY-Libya",
    "Europe" = "LI-Liechtenstein (the Principality of )",
    "Europe" = "LT-Lithuania",
    "Europe" = "LU-Luxembourg",
    "Afrique" = "MG-Madagascar",
    "Asie" = "MY-Malaysia",
    "Afrique" = "ML-Mali",
    "Amérique & Océanie" = "MQ-Martinica",
    "Afrique" = "MR-Mauritania",
    "Afrique" = "MU-Mauricio",
    "Amérique & Océanie" = "MX-Mexico",
    "Europe" = "MC-Monaco",
    "Europe" = "MD-Moldova (the Republic of)",
    "Europe" = "ME-Montenegro",
    "Afrique" = "MA-Morocco",
    "Afrique" = "MZ-Mozambique",
    "Asie" = "NP-Nepal",
    "Europe" = "NL-Netherlands",
    "Amérique & Océanie" = "NC-Nueva Caledonia",
    "Amérique & Océanie" = "NZ-New Zealand",
    "Amérique & Océanie" = "NI-Nicaragua",
    "Afrique" = "NE-Niger",
    "Afrique" = "NG-Nigeria",
    "Europe" = "NO-Norway",
    "Asie" = "PK-Pakistan",
    "Amérique & Océanie" = "PY-Paraguay",
    "Amérique & Océanie" = "PE-Peru",
    "Asie" = "PH-Philippines",
    "Europe" = "PL-Poland",
    "Europe" = "PT-Portugal",
    "Afrique" = "GW-Guinea-Bissau",
    "Amérique & Océanie" = "PR-Puerto Rico",
    "Afrique" = "RE-Reunión",
    "Europe" = "RO-Romania",
    "Europe" = "RU-Russia",
    "Afrique" = "RW-Rwanda",
    "Amérique & Océanie" = "ST-Sao Tome and Principe",
    "Afrique" = "SN-Senegal",
    "Europe" = "RS-Serbia",
    "Afrique" = "SC-Seychelles",
    "Afrique" = "SL-Sierra Leone (the Republic of)",
    "Asie" = "SK-Singapore (the Republic of)",
    "Europe" = "SK-Slovak Republic",
    "Asie" = "VN-Vietnam",
    "Europe" = "SI-Slovenia",
    "Afrique" = "SO-Somalia",
    "Afrique" = "ZA-South Africa",
    "Europe" = "ES-Spain",
    "Afrique" = "SS-South Sudan",
    "Afrique" = "SD-Sudan",
    "Amérique & Océanie" = "SR-Suriname (the Republic of)",
    "Europe" = "SE-Sweden",
    "Europe" = "CH-Switzerland",
    "Asie" = "SY-Syrian Arab Republic",
    "Asie" = "TJ-Takijistan (the Republic of)",
    "Asie" = "TH-Thailand",
    "Afrique" = "TN-Tunisia",
    "Asie" = "TR-Turkey",
    "Asie" = "TR-Turkmenistan",
    "Europe" = "UA-Ukraine",
    "Europe" = "MK-Macedonia (the former Yugoslav Republic of)",
    "Europe" = "SU-USSR",
    "Afrique" = "EG-Egypt",
    "Europe" = "GB-Great Britain and/or United Kingdom",
    "Europe" = "JE-Jersey",
    "Afrique" = "TZ-Tanzania (the United Republic of)",
    "Amérique & Océanie" = "US-United States",
    "Afrique" = "BF-Furkina Faso",
    "Amérique & Océanie" = "UY-Uruguay",
    "Asie" = "UZ-Uzbekistan (the Republic of)",
    "Amérique & Océanie" = "VE-Venezuela",
    "Asie" = "YE-Yemen",
    "Europe" = "YU-Yugoslavia",
    "Afrique" = "ZM-Zambia",
    "Europe" = "XK-Kosovo",
    NULL = "Other (CH,EE,JP,RU,TR,US)"
  )

# Suppression de la ligne "deux parents citoyens" chez les DI
irec(DATA_I$V64)
## Recodage de DATA_I$V64 en DATA_I$V64_desc
DATA_I$V64_desc <- DATA_I$V64 |>
  fct_recode(
    NULL = "Both were citizens"
  )

# Représentation sous la forme de tableau
tab <- svytable(~ V64_desc + F_BORN_desc, DATA_IW)
write_clip(lprop(tab))
tab <- svytable(~ V64_desc + M_BORN_desc, DATA_IW)
write_clip(lprop(tab))

# Chômage
tab <- svytable(~ V64_desc + WORK, DATA_IW)
write_clip(lprop(tab))

# Niveau d'études
irec(DATA_I$FR_DEGR)
## Recodage de DATA_I$FR_DEGR en DATA_I$FR_DEGR_desc
DATA_I$FR_DEGR_desc <- DATA_I$FR_DEGR |>
  fct_recode(
    "Non diplômé•e du supérieur" = "None",
    "Non diplômé•e du supérieur" = "Primary incomplete",
    "Non diplômé•e du supérieur" = "Primary completed",
    "Non diplômé•e du supérieur" = "General secondary level 1",
    "Non diplômé•e du supérieur" = "Vocational secondary level 1 without vocational diploma",
    "Non diplômé•e du supérieur" = "Vocational secondary level 1 with vocational diploma",
    "Non diplômé•e du supérieur" = "Vocational secondary level 2",
    "Non diplômé•e du supérieur" = "Incomplete general secondary level 2",
    "Non diplômé•e du supérieur" = "General secondary level 2",
    "Diplômé•e du supérieur" = "College",
    "Diplômé•e du supérieur" = "University"
  )

tab <- svytable(~ V64_desc + FR_DEGR_desc, DATA_IW)
write_clip(lprop(tab))

# Revenu
irec(DATA_I$FR_RINC)
## Recodage de DATA_I$FR_RINC en DATA_I$FR_RINC_desc
DATA_I$FR_RINC_desc <- DATA_I$FR_RINC |>
  fct_recode(
    "Inférieur au seuil de pauvreté" = "No income",
    "Inférieur au seuil de pauvreté" = "30 EUR per month",
    "Inférieur au seuil de pauvreté" = "47 EUR",
    "Inférieur au seuil de pauvreté" = "60 EUR",
    "Inférieur au seuil de pauvreté" = "99 EUR",
    "Inférieur au seuil de pauvreté" = "100 EUR",
    "Inférieur au seuil de pauvreté" = "120 EUR",
    "Inférieur au seuil de pauvreté" = "127 EUR",
    "Inférieur au seuil de pauvreté" = "135 EUR",
    "Inférieur au seuil de pauvreté" = "148 EUR",
    "Inférieur au seuil de pauvreté" = "150 EUR",
    "Inférieur au seuil de pauvreté" = "160 EUR",
    "Inférieur au seuil de pauvreté" = "170 EUR",
    "Inférieur au seuil de pauvreté" = "200 EUR",
    "Inférieur au seuil de pauvreté" = "218 EUR",
    "Inférieur au seuil de pauvreté" = "250 EUR",
    "Inférieur au seuil de pauvreté" = "260 EUR",
    "Inférieur au seuil de pauvreté" = "300 EUR",
    "Inférieur au seuil de pauvreté" = "305 EUR",
    "Inférieur au seuil de pauvreté" = "380 EUR",
    "Inférieur au seuil de pauvreté" = "388 EUR",
    "Inférieur au seuil de pauvreté" = "390 EUR",
    "Inférieur au seuil de pauvreté" = "400 EUR",
    "Inférieur au seuil de pauvreté" = "410 EUR",
    "Inférieur au seuil de pauvreté" = "425 EUR",
    "Inférieur au seuil de pauvreté" = "430 EUR",
    "Inférieur au seuil de pauvreté" = "440 EUR",
    "Inférieur au seuil de pauvreté" = "450 EUR",
    "Inférieur au seuil de pauvreté" = "456 EUR",
    "Inférieur au seuil de pauvreté" = "460 EUR",
    "Inférieur au seuil de pauvreté" = "468 EUR",
    "Inférieur au seuil de pauvreté" = "475 EUR",
    "Inférieur au seuil de pauvreté" = "476 EUR",
    "Inférieur au seuil de pauvreté" = "480 EUR",
    "Inférieur au seuil de pauvreté" = "487 EUR",
    "Inférieur au seuil de pauvreté" = "490 EUR",
    "Inférieur au seuil de pauvreté" = "500 EUR",
    "Inférieur au seuil de pauvreté" = "518 EUR",
    "Inférieur au seuil de pauvreté" = "550 EUR",
    "Inférieur au seuil de pauvreté" = "560 EUR",
    "Inférieur au seuil de pauvreté" = "580 EUR",
    "Inférieur au seuil de pauvreté" = "600 EUR",
    "Inférieur au seuil de pauvreté" = "604 EUR",
    "Inférieur au seuil de pauvreté" = "620 EUR",
    "Inférieur au seuil de pauvreté" = "645 EUR",
    "Inférieur au seuil de pauvreté" = "650 EUR",
    "Inférieur au seuil de pauvreté" = "657 EUR",
    "Inférieur au seuil de pauvreté" = "680 EUR",
    "Inférieur au seuil de pauvreté" = "690 EUR",
    "Inférieur au seuil de pauvreté" = "700 EUR",
    "Inférieur au seuil de pauvreté" = "712 EUR",
    "Inférieur au seuil de pauvreté" = "715 EUR",
    "Inférieur au seuil de pauvreté" = "720 EUR",
    "Inférieur au seuil de pauvreté" = "740 EUR",
    "Inférieur au seuil de pauvreté" = "750 EUR",
    "Inférieur au seuil de pauvreté" = "770 EUR",
    "Inférieur au seuil de pauvreté" = "774 EUR",
    "Inférieur au seuil de pauvreté" = "777 EUR",
    "Inférieur au seuil de pauvreté" = "780 EUR",
    "Inférieur au seuil de pauvreté" = "800 EUR",
    "Inférieur au seuil de pauvreté" = "820 EUR",
    "Inférieur au seuil de pauvreté" = "822 EUR",
    "Inférieur au seuil de pauvreté" = "830 EUR",
    "Inférieur au seuil de pauvreté" = "840 EUR",
    "Inférieur au seuil de pauvreté" = "850 EUR",
    "Inférieur au seuil de pauvreté" = "860 EUR",
    "Inférieur au seuil de pauvreté" = "863 EUR",
    "Inférieur au seuil de pauvreté" = "870 EUR",
    "Inférieur au seuil de pauvreté" = "874 EUR",
    "Inférieur au seuil de pauvreté" = "890 EUR",
    "Inférieur au seuil de pauvreté" = "900 EUR",
    "Inférieur au seuil de pauvreté" = "910 EUR",
    "Inférieur au seuil de pauvreté" = "930 EUR",
    "Inférieur au seuil de pauvreté" = "932 EUR",
    "Inférieur au seuil de pauvreté" = "934 EUR",
    "Inférieur au seuil de pauvreté" = "939 EUR",
    "Inférieur au seuil de pauvreté" = "940 EUR",
    "Inférieur au seuil de pauvreté" = "950 EUR",
    "Inférieur au seuil de pauvreté" = "960 EUR",
    "Inférieur au seuil de pauvreté" = "970 EUR",
    "Inférieur au seuil de pauvreté" = "980 EUR",
    "Inférieur au seuil de pauvreté" = "985 EUR",
    "Inférieur au seuil de pauvreté" = "990 EUR",
    "Inférieur au seuil de pauvreté" = "993 EUR",
    "Inférieur ou égal au revenu médian" = "1.000 EUR",
    "Inférieur ou égal au revenu médian" = "1.013 EUR",
    "Inférieur ou égal au revenu médian" = "1.015 EUR",
    "Inférieur ou égal au revenu médian" = "1.034 EUR",
    "Inférieur ou égal au revenu médian" = "1.039 EUR",
    "Inférieur ou égal au revenu médian" = "1.042 EUR",
    "Inférieur ou égal au revenu médian" = "1.045 EUR",
    "Inférieur ou égal au revenu médian" = "1.050 EUR",
    "Inférieur ou égal au revenu médian" = "1.055 EUR",
    "Inférieur ou égal au revenu médian" = "1.072 EUR",
    "Inférieur ou égal au revenu médian" = "1.080 EUR",
    "Inférieur ou égal au revenu médian" = "1.090 EUR",
    "Inférieur ou égal au revenu médian" = "1.100 EUR",
    "Inférieur ou égal au revenu médian" = "1.104 EUR",
    "Inférieur ou égal au revenu médian" = "1.123 EUR",
    "Inférieur ou égal au revenu médian" = "1.127 EUR",
    "Inférieur ou égal au revenu médian" = "1.130 EUR",
    "Inférieur ou égal au revenu médian" = "1.150 EUR",
    "Inférieur ou égal au revenu médian" = "1.170 EUR",
    "Inférieur ou égal au revenu médian" = "1.171 EUR",
    "Inférieur ou égal au revenu médian" = "1.175 EUR",
    "Inférieur ou égal au revenu médian" = "1.180 EUR",
    "Inférieur ou égal au revenu médian" = "1.200 EUR",
    "Inférieur ou égal au revenu médian" = "1.210 EUR",
    "Inférieur ou égal au revenu médian" = "1.230 EUR",
    "Inférieur ou égal au revenu médian" = "1.240 EUR",
    "Inférieur ou égal au revenu médian" = "1.250 EUR",
    "Inférieur ou égal au revenu médian" = "1.260 EUR",
    "Inférieur ou égal au revenu médian" = "1.270 EUR",
    "Inférieur ou égal au revenu médian" = "1.280 EUR",
    "Inférieur ou égal au revenu médian" = "1.284 EUR",
    "Inférieur ou égal au revenu médian" = "1.290 EUR",
    "Inférieur ou égal au revenu médian" = "1.291 EUR",
    "Inférieur ou égal au revenu médian" = "1.300 EUR",
    "Inférieur ou égal au revenu médian" = "1.324 EUR",
    "Inférieur ou égal au revenu médian" = "1.340 EUR",
    "Inférieur ou égal au revenu médian" = "1.350 EUR",
    "Inférieur ou égal au revenu médian" = "1.360 EUR",
    "Inférieur ou égal au revenu médian" = "1.375 EUR",
    "Inférieur ou égal au revenu médian" = "1.376 EUR",
    "Inférieur ou égal au revenu médian" = "1.382 EUR",
    "Inférieur ou égal au revenu médian" = "1.400 EUR",
    "Inférieur ou égal au revenu médian" = "1.405 EUR",
    "Inférieur ou égal au revenu médian" = "1.409 EUR",
    "Inférieur ou égal au revenu médian" = "1.420 EUR",
    "Inférieur ou égal au revenu médian" = "1.430 EUR",
    "Inférieur ou égal au revenu médian" = "1.450 EUR",
    "Inférieur ou égal au revenu médian" = "1.460 EUR",
    "Inférieur ou égal au revenu médian" = "1.480 EUR",
    "Inférieur ou égal au revenu médian" = "1.500 EUR",
    "Inférieur ou égal au revenu médian" = "1.520 EUR",
    "Inférieur ou égal au revenu médian" = "1.527 EUR",
    "Inférieur ou égal au revenu médian" = "1.530 EUR",
    "Inférieur ou égal au revenu médian" = "1.540 EUR",
    "Inférieur ou égal au revenu médian" = "1.543 EUR",
    "Inférieur ou égal au revenu médian" = "1.550 EUR",
    "Inférieur ou égal au revenu médian" = "1.590 EUR",
    "Inférieur ou égal au revenu médian" = "1.598 EUR",
    "Inférieur ou égal au revenu médian" = "1.600 EUR",
    "Inférieur ou égal au revenu médian" = "1.615 EUR",
    "Inférieur ou égal au revenu médian" = "1.623 EUR",
    "Inférieur ou égal au revenu médian" = "1.625 EUR",
    "Inférieur ou égal au revenu médian" = "1.645 EUR",
    "Inférieur ou égal au revenu médian" = "1.650 EUR",
    "Supérieur au revenu médian" = "1.680 EUR",
    "Supérieur au revenu médian" = "1.690 EUR",
    "Supérieur au revenu médian" = "1.700 EUR",
    "Supérieur au revenu médian" = "1.720 EUR",
    "Supérieur au revenu médian" = "1.750 EUR",
    "Supérieur au revenu médian" = "1.760 EUR",
    "Supérieur au revenu médian" = "1.800 EUR",
    "Supérieur au revenu médian" = "1.820 EUR",
    "Supérieur au revenu médian" = "1.830 EUR",
    "Supérieur au revenu médian" = "1.850 EUR",
    "Supérieur au revenu médian" = "1.860 EUR",
    "Supérieur au revenu médian" = "1.890 EUR",
    "Supérieur au revenu médian" = "1.900 EUR",
    "Supérieur au revenu médian" = "1.920 EUR",
    "Supérieur au revenu médian" = "1.950 EUR",
    "Supérieur au revenu médian" = "2.000 EUR",
    "Supérieur au revenu médian" = "2.010 EUR",
    "Supérieur au revenu médian" = "2.015 EUR",
    "Supérieur au revenu médian" = "2.018 EUR",
    "Supérieur au revenu médian" = "2.050 EUR",
    "Supérieur au revenu médian" = "2.060 EUR",
    "Supérieur au revenu médian" = "2.100 EUR",
    "Supérieur au revenu médian" = "2.108 EUR",
    "Supérieur au revenu médian" = "2.130 EUR",
    "Supérieur au revenu médian" = "2.135 EUR",
    "Supérieur au revenu médian" = "2.140 EUR",
    "Supérieur au revenu médian" = "2.150 EUR",
    "Supérieur au revenu médian" = "2.170 EUR",
    "Supérieur au revenu médian" = "2.200 EUR",
    "Supérieur au revenu médian" = "2.250 EUR",
    "Supérieur au revenu médian" = "2.293 EUR",
    "Supérieur au revenu médian" = "2.300 EUR",
    "Supérieur au revenu médian" = "2.350 EUR",
    "Supérieur au revenu médian" = "2.367 EUR",
    "Supérieur au revenu médian" = "2.400 EUR",
    "Supérieur au revenu médian" = "2.440 EUR",
    "Supérieur au revenu médian" = "2.450 EUR",
    "Supérieur au revenu médian" = "2.458 EUR",
    "Supérieur au revenu médian" = "2.496 EUR",
    "Supérieur au revenu médian" = "2.500 EUR",
    "Supérieur au revenu médian" = "2.590 EUR",
    "Supérieur au revenu médian" = "2.600 EUR",
    "Supérieur au revenu médian" = "2.650 EUR",
    "Supérieur au revenu médian" = "2.700 EUR",
    "Supérieur au revenu médian" = "2.776 EUR",
    "Supérieur au revenu médian" = "2.800 EUR",
    "Supérieur au revenu médian" = "2.870 EUR",
    "Supérieur au revenu médian" = "2.900 EUR",
    "Supérieur au revenu médian" = "3.000 EUR",
    "Supérieur au revenu médian" = "3.099 EUR",
    "Supérieur au revenu médian" = "3.100 EUR",
    "Supérieur au revenu médian" = "3.130 EUR",
    "Supérieur au revenu médian" = "3.200 EUR",
    "Supérieur au revenu médian" = "3.300 EUR",
    "Supérieur au revenu médian" = "3.400 EUR",
    "Supérieur au revenu médian" = "3.430 EUR",
    "Supérieur au revenu médian" = "3.450 EUR",
    "Supérieur au revenu médian" = "3.500 EUR",
    "Supérieur au revenu médian" = "3.600 EUR",
    "Supérieur au revenu médian" = "3.700 EUR",
    "Supérieur au revenu médian" = "3.740 EUR",
    "Supérieur au revenu médian" = "3.756 EUR",
    "Supérieur au revenu médian" = "3.800 EUR",
    "Supérieur au revenu médian" = "4.000 EUR",
    "Supérieur au revenu médian" = "4.100 EUR",
    "Supérieur au revenu médian" = "4.180 EUR",
    "Supérieur au revenu médian" = "4.200 EUR",
    "Supérieur au revenu médian" = "4.300 EUR",
    "Supérieur au revenu médian" = "4.400 EUR",
    "Supérieur au revenu médian" = "4.500 EUR",
    "Supérieur au revenu médian" = "4.600 EUR",
    "Supérieur au revenu médian" = "4.700 EUR",
    "Supérieur au revenu médian" = "4.800 EUR",
    "Supérieur au revenu médian" = "4.900 EUR",
    "Supérieur au revenu médian" = "5.000 EUR",
    "Supérieur au revenu médian" = "5.400 EUR",
    "Supérieur au revenu médian" = "5.500 EUR",
    "Supérieur au revenu médian" = "5.600 EUR",
    "Supérieur au revenu médian" = "6.000 EUR",
    "Supérieur au revenu médian" = "6.500 EUR",
    "Supérieur au revenu médian" = "7.000 EUR",
    "Supérieur au revenu médian" = "7.020 EUR",
    "Supérieur au revenu médian" = "7.500 EUR",
    "Supérieur au revenu médian" = "7.600 EUR",
    "Supérieur au revenu médian" = "7.800 EUR",
    "Supérieur au revenu médian" = "8.000 EUR",
    "Supérieur au revenu médian" = "8.400 EUR",
    "Supérieur au revenu médian" = "8.500 EUR",
    "Supérieur au revenu médian" = "9.000 EUR",
    "Supérieur au revenu médian" = "9.500 EUR",
    "Supérieur au revenu médian" = "10.000 EUR",
    "Supérieur au revenu médian" = "10.515 EUR",
    "Supérieur au revenu médian" = "11.000 EUR",
    "Supérieur au revenu médian" = "11.111 EUR",
    "Supérieur au revenu médian" = "12.000 EUR",
    "Supérieur au revenu médian" = "12.500 EUR",
    "Supérieur au revenu médian" = "13.000 EUR",
    "Supérieur au revenu médian" = "13.600 EUR",
    "Supérieur au revenu médian" = "14.352 EUR",
    "Supérieur au revenu médian" = "15.000 EUR",
    "Supérieur au revenu médian" = "15.600 EUR",
    "Supérieur au revenu médian" = "16.000 EUR",
    "Supérieur au revenu médian" = "16.200 EUR",
    "Supérieur au revenu médian" = "17.000 EUR",
    "Supérieur au revenu médian" = "17.500 EUR",
    "Supérieur au revenu médian" = "18.000 EUR",
    "Supérieur au revenu médian" = "18.600 EUR",
    "Supérieur au revenu médian" = "18.900 EUR",
    "Supérieur au revenu médian" = "19.000 EUR",
    "Supérieur au revenu médian" = "20.000 EUR",
    "Supérieur au revenu médian" = "20.400 EUR",
    "Supérieur au revenu médian" = "21.000 EUR",
    "Supérieur au revenu médian" = "21.085 EUR",
    "Supérieur au revenu médian" = "22.000 EUR",
    "Supérieur au revenu médian" = "22.100 EUR",
    "Supérieur au revenu médian" = "23.000 EUR",
    "Supérieur au revenu médian" = "23.560 EUR",
    "Supérieur au revenu médian" = "24.000 EUR",
    "Supérieur au revenu médian" = "25.000 EUR",
    "Supérieur au revenu médian" = "27.000 EUR",
    "Supérieur au revenu médian" = "28.020 EUR",
    "Supérieur au revenu médian" = "30.000 EUR",
    "Supérieur au revenu médian" = "35.000 EUR",
    "Supérieur au revenu médian" = "36.000 EUR",
    "Supérieur au revenu médian" = "38.000 EUR",
    "Supérieur au revenu médian" = "38.900 EUR",
    "Supérieur au revenu médian" = "40.000 EUR",
    "Supérieur au revenu médian" = "45.000 EUR",
    "Supérieur au revenu médian" = "46.000 EUR",
    "Supérieur au revenu médian" = "50.000 EUR",
    "Supérieur au revenu médian" = "51.000 EUR",
    "Supérieur au revenu médian" = "60.000 EUR",
    "Supérieur au revenu médian" = "70.020 EUR",
    "Supérieur au revenu médian" = "75.200 EUR",
    "Supérieur au revenu médian" = "130.000 EUR per month"
  )

tab <- svytable(~ V64_desc + FR_RINC_desc, DATA_IW)
write_clip(lprop(tab))

# Age
DATA_I$AGE<-as.numeric(as.character(DATA_I$AGE))
DATA_FR$AGE<-as.numeric(as.character(DATA_FR$AGE))
write_clip(summary(DATA_I$AGE))
write_clip(summary(DATA_FR$AGE))

# Genre
tab <- svytable(~ V64 + SEX, DATA_W)
write_clip(lprop(tab))

# Diplôme
irec(DATA_FR$FR_DEGR)
## Recodage de DATA_FR$FR_DEGR en DATA_FR$FR_DEGR_rec
DATA_FR$FR_DEGR_rec <- DATA_FR$FR_DEGR |>
  fct_recode(
    "Pas d'études" = "None",
    "Primaire" = "Primary incomplete",
    "Primaire" = "Primary completed",
    "Secondaire général" = "General secondary level 1",
    "Secondaire professionnel" = "Vocational secondary level 1 without vocational diploma",
    "Secondaire professionnel" = "Vocational secondary level 1 with vocational diploma",
    "Secondaire professionnel" = "Vocational secondary level 2",
    "Secondaire général" = "Incomplete general secondary level 2",
    "Secondaire général" = "General secondary level 2",
    "Licence" = "College",
    "Master" = "University"
  )

tab <- svytable(~ V64 + FR_DEGR_rec, DATA_W)
write_clip(lprop(tab))
chisq.test(tab)
cramer.v(tab)
fisher.test(tab)

irec(DATA_I$FR_DEGR)
# Recodage de DATA_I$FR_DEGR en DATA_I$FR_DEGR_rec
DATA_I$FR_DEGR_rec <- DATA_I$FR_DEGR |>
  fct_recode(
    NULL = "None",
    NULL = "Primary incomplete",
    NULL = "Primary completed",
    "Secondaire général" = "General secondary level 1",
    "Secondaire professionnel" = "Vocational secondary level 1 without vocational diploma",
    "Secondaire professionnel" = "Vocational secondary level 1 with vocational diploma",
    "Secondaire professionnel" = "Vocational secondary level 2",
    "Secondaire général" = "Incomplete general secondary level 2",
    "Secondaire général" = "General secondary level 2",
    "Licence" = "College",
    "Master" = "University"
  )

tab <- svytable(~ F_BORN_desc + FR_DEGR_rec, DATA_IW)
write_clip(tab)
chisq.test(tab)
cramer.v(tab)
fisher.test(tab)

# Situation géographique

irec(DATA_FR$URBRURAL)
## Recodage de DATA_FR$URBRURAL en DATA_FR$URBRURAL_desc
DATA_FR$URBRURAL_desc <- DATA_FR$URBRURAL |>
  fct_recode(
    "Urbain" = "A big city",
    "Urbain" = "The suburbs or outskirts of a big city",
    "Urbain" = "A town or a small city",
    "Rural" = "A country village",
    "Rural" = "A farm or home in the country",
    NULL = "Other answer (GB)"
  )

tab <- svytable(~ V64 + URBRURAL_desc, DATA_W)
write_clip(lprop(tab))
chisq.test(tab)
cramer.v(tab)
fisher.test(tab)

irec(DATA_I$URBRURAL)
## Recodage de DATA_I$URBRURAL en DATA_I$URBRURAL_desc
DATA_I$URBRURAL_desc <- DATA_I$URBRURAL |>
  fct_recode(
    "Urbain" = "A big city",
    "Urbain" = "The suburbs or outskirts of a big city",
    "Urbain" = "A town or a small city",
    "Rural" = "A country village",
    "Rural" = "A farm or home in the country",
    NULL = "Other answer (GB)"
  )

tab <- svytable(~ F_BORN_desc + URBRURAL_desc, DATA_IW)
write_clip(lprop(tab))
chisq.test(tab)
cramer.v(tab)
fisher.test(tab)

# Revenu
# Revenu
irec(DATA_FR$FR_RINC)
## Recodage de DATA_FR$FR_RINC en DATA_FR$FR_RINC_desc
DATA_FR$FR_RINC_desc <- DATA_FR$FR_RINC |>
  fct_recode(
    "Inférieur au seuil de pauvreté" = "No income",
    "Inférieur au seuil de pauvreté" = "30 EUR per month",
    "Inférieur au seuil de pauvreté" = "47 EUR",
    "Inférieur au seuil de pauvreté" = "60 EUR",
    "Inférieur au seuil de pauvreté" = "99 EUR",
    "Inférieur au seuil de pauvreté" = "100 EUR",
    "Inférieur au seuil de pauvreté" = "120 EUR",
    "Inférieur au seuil de pauvreté" = "127 EUR",
    "Inférieur au seuil de pauvreté" = "135 EUR",
    "Inférieur au seuil de pauvreté" = "148 EUR",
    "Inférieur au seuil de pauvreté" = "150 EUR",
    "Inférieur au seuil de pauvreté" = "160 EUR",
    "Inférieur au seuil de pauvreté" = "170 EUR",
    "Inférieur au seuil de pauvreté" = "200 EUR",
    "Inférieur au seuil de pauvreté" = "218 EUR",
    "Inférieur au seuil de pauvreté" = "250 EUR",
    "Inférieur au seuil de pauvreté" = "260 EUR",
    "Inférieur au seuil de pauvreté" = "300 EUR",
    "Inférieur au seuil de pauvreté" = "305 EUR",
    "Inférieur au seuil de pauvreté" = "380 EUR",
    "Inférieur au seuil de pauvreté" = "388 EUR",
    "Inférieur au seuil de pauvreté" = "390 EUR",
    "Inférieur au seuil de pauvreté" = "400 EUR",
    "Inférieur au seuil de pauvreté" = "410 EUR",
    "Inférieur au seuil de pauvreté" = "425 EUR",
    "Inférieur au seuil de pauvreté" = "430 EUR",
    "Inférieur au seuil de pauvreté" = "440 EUR",
    "Inférieur au seuil de pauvreté" = "450 EUR",
    "Inférieur au seuil de pauvreté" = "456 EUR",
    "Inférieur au seuil de pauvreté" = "460 EUR",
    "Inférieur au seuil de pauvreté" = "468 EUR",
    "Inférieur au seuil de pauvreté" = "475 EUR",
    "Inférieur au seuil de pauvreté" = "476 EUR",
    "Inférieur au seuil de pauvreté" = "480 EUR",
    "Inférieur au seuil de pauvreté" = "487 EUR",
    "Inférieur au seuil de pauvreté" = "490 EUR",
    "Inférieur au seuil de pauvreté" = "500 EUR",
    "Inférieur au seuil de pauvreté" = "518 EUR",
    "Inférieur au seuil de pauvreté" = "550 EUR",
    "Inférieur au seuil de pauvreté" = "560 EUR",
    "Inférieur au seuil de pauvreté" = "580 EUR",
    "Inférieur au seuil de pauvreté" = "600 EUR",
    "Inférieur au seuil de pauvreté" = "604 EUR",
    "Inférieur au seuil de pauvreté" = "620 EUR",
    "Inférieur au seuil de pauvreté" = "645 EUR",
    "Inférieur au seuil de pauvreté" = "650 EUR",
    "Inférieur au seuil de pauvreté" = "657 EUR",
    "Inférieur au seuil de pauvreté" = "680 EUR",
    "Inférieur au seuil de pauvreté" = "690 EUR",
    "Inférieur au seuil de pauvreté" = "700 EUR",
    "Inférieur au seuil de pauvreté" = "712 EUR",
    "Inférieur au seuil de pauvreté" = "715 EUR",
    "Inférieur au seuil de pauvreté" = "720 EUR",
    "Inférieur au seuil de pauvreté" = "740 EUR",
    "Inférieur au seuil de pauvreté" = "750 EUR",
    "Inférieur au seuil de pauvreté" = "770 EUR",
    "Inférieur au seuil de pauvreté" = "774 EUR",
    "Inférieur au seuil de pauvreté" = "777 EUR",
    "Inférieur au seuil de pauvreté" = "780 EUR",
    "Inférieur au seuil de pauvreté" = "800 EUR",
    "Inférieur au seuil de pauvreté" = "820 EUR",
    "Inférieur au seuil de pauvreté" = "822 EUR",
    "Inférieur au seuil de pauvreté" = "830 EUR",
    "Inférieur au seuil de pauvreté" = "840 EUR",
    "Inférieur au seuil de pauvreté" = "850 EUR",
    "Inférieur au seuil de pauvreté" = "860 EUR",
    "Inférieur au seuil de pauvreté" = "863 EUR",
    "Inférieur au seuil de pauvreté" = "870 EUR",
    "Inférieur au seuil de pauvreté" = "874 EUR",
    "Inférieur au seuil de pauvreté" = "890 EUR",
    "Inférieur au seuil de pauvreté" = "900 EUR",
    "Inférieur au seuil de pauvreté" = "910 EUR",
    "Inférieur au seuil de pauvreté" = "930 EUR",
    "Inférieur au seuil de pauvreté" = "932 EUR",
    "Inférieur au seuil de pauvreté" = "934 EUR",
    "Inférieur au seuil de pauvreté" = "939 EUR",
    "Inférieur au seuil de pauvreté" = "940 EUR",
    "Inférieur au seuil de pauvreté" = "950 EUR",
    "Inférieur au seuil de pauvreté" = "960 EUR",
    "Inférieur au seuil de pauvreté" = "970 EUR",
    "Inférieur au seuil de pauvreté" = "980 EUR",
    "Inférieur au seuil de pauvreté" = "985 EUR",
    "Inférieur au seuil de pauvreté" = "990 EUR",
    "Inférieur au seuil de pauvreté" = "993 EUR",
    "Inférieur ou égal au revenu médian" = "1.000 EUR",
    "Inférieur ou égal au revenu médian" = "1.013 EUR",
    "Inférieur ou égal au revenu médian" = "1.015 EUR",
    "Inférieur ou égal au revenu médian" = "1.034 EUR",
    "Inférieur ou égal au revenu médian" = "1.039 EUR",
    "Inférieur ou égal au revenu médian" = "1.042 EUR",
    "Inférieur ou égal au revenu médian" = "1.045 EUR",
    "Inférieur ou égal au revenu médian" = "1.050 EUR",
    "Inférieur ou égal au revenu médian" = "1.055 EUR",
    "Inférieur ou égal au revenu médian" = "1.072 EUR",
    "Inférieur ou égal au revenu médian" = "1.080 EUR",
    "Inférieur ou égal au revenu médian" = "1.090 EUR",
    "Inférieur ou égal au revenu médian" = "1.100 EUR",
    "Inférieur ou égal au revenu médian" = "1.104 EUR",
    "Inférieur ou égal au revenu médian" = "1.123 EUR",
    "Inférieur ou égal au revenu médian" = "1.127 EUR",
    "Inférieur ou égal au revenu médian" = "1.130 EUR",
    "Inférieur ou égal au revenu médian" = "1.150 EUR",
    "Inférieur ou égal au revenu médian" = "1.170 EUR",
    "Inférieur ou égal au revenu médian" = "1.171 EUR",
    "Inférieur ou égal au revenu médian" = "1.175 EUR",
    "Inférieur ou égal au revenu médian" = "1.180 EUR",
    "Inférieur ou égal au revenu médian" = "1.200 EUR",
    "Inférieur ou égal au revenu médian" = "1.210 EUR",
    "Inférieur ou égal au revenu médian" = "1.230 EUR",
    "Inférieur ou égal au revenu médian" = "1.240 EUR",
    "Inférieur ou égal au revenu médian" = "1.250 EUR",
    "Inférieur ou égal au revenu médian" = "1.260 EUR",
    "Inférieur ou égal au revenu médian" = "1.270 EUR",
    "Inférieur ou égal au revenu médian" = "1.280 EUR",
    "Inférieur ou égal au revenu médian" = "1.284 EUR",
    "Inférieur ou égal au revenu médian" = "1.290 EUR",
    "Inférieur ou égal au revenu médian" = "1.291 EUR",
    "Inférieur ou égal au revenu médian" = "1.300 EUR",
    "Inférieur ou égal au revenu médian" = "1.324 EUR",
    "Inférieur ou égal au revenu médian" = "1.340 EUR",
    "Inférieur ou égal au revenu médian" = "1.350 EUR",
    "Inférieur ou égal au revenu médian" = "1.360 EUR",
    "Inférieur ou égal au revenu médian" = "1.375 EUR",
    "Inférieur ou égal au revenu médian" = "1.376 EUR",
    "Inférieur ou égal au revenu médian" = "1.382 EUR",
    "Inférieur ou égal au revenu médian" = "1.400 EUR",
    "Inférieur ou égal au revenu médian" = "1.405 EUR",
    "Inférieur ou égal au revenu médian" = "1.409 EUR",
    "Inférieur ou égal au revenu médian" = "1.420 EUR",
    "Inférieur ou égal au revenu médian" = "1.430 EUR",
    "Inférieur ou égal au revenu médian" = "1.450 EUR",
    "Inférieur ou égal au revenu médian" = "1.460 EUR",
    "Inférieur ou égal au revenu médian" = "1.480 EUR",
    "Inférieur ou égal au revenu médian" = "1.500 EUR",
    "Inférieur ou égal au revenu médian" = "1.520 EUR",
    "Inférieur ou égal au revenu médian" = "1.527 EUR",
    "Inférieur ou égal au revenu médian" = "1.530 EUR",
    "Inférieur ou égal au revenu médian" = "1.540 EUR",
    "Inférieur ou égal au revenu médian" = "1.543 EUR",
    "Inférieur ou égal au revenu médian" = "1.550 EUR",
    "Inférieur ou égal au revenu médian" = "1.590 EUR",
    "Inférieur ou égal au revenu médian" = "1.598 EUR",
    "Inférieur ou égal au revenu médian" = "1.600 EUR",
    "Inférieur ou égal au revenu médian" = "1.615 EUR",
    "Inférieur ou égal au revenu médian" = "1.623 EUR",
    "Inférieur ou égal au revenu médian" = "1.625 EUR",
    "Inférieur ou égal au revenu médian" = "1.645 EUR",
    "Inférieur ou égal au revenu médian" = "1.650 EUR",
    "Supérieur au revenu médian" = "1.680 EUR",
    "Supérieur au revenu médian" = "1.690 EUR",
    "Supérieur au revenu médian" = "1.700 EUR",
    "Supérieur au revenu médian" = "1.720 EUR",
    "Supérieur au revenu médian" = "1.750 EUR",
    "Supérieur au revenu médian" = "1.760 EUR",
    "Supérieur au revenu médian" = "1.800 EUR",
    "Supérieur au revenu médian" = "1.820 EUR",
    "Supérieur au revenu médian" = "1.830 EUR",
    "Supérieur au revenu médian" = "1.850 EUR",
    "Supérieur au revenu médian" = "1.860 EUR",
    "Supérieur au revenu médian" = "1.890 EUR",
    "Supérieur au revenu médian" = "1.900 EUR",
    "Supérieur au revenu médian" = "1.920 EUR",
    "Supérieur au revenu médian" = "1.950 EUR",
    "Supérieur au revenu médian" = "2.000 EUR",
    "Supérieur au revenu médian" = "2.010 EUR",
    "Supérieur au revenu médian" = "2.015 EUR",
    "Supérieur au revenu médian" = "2.018 EUR",
    "Supérieur au revenu médian" = "2.050 EUR",
    "Supérieur au revenu médian" = "2.060 EUR",
    "Supérieur au revenu médian" = "2.100 EUR",
    "Supérieur au revenu médian" = "2.108 EUR",
    "Supérieur au revenu médian" = "2.130 EUR",
    "Supérieur au revenu médian" = "2.135 EUR",
    "Supérieur au revenu médian" = "2.140 EUR",
    "Supérieur au revenu médian" = "2.150 EUR",
    "Supérieur au revenu médian" = "2.170 EUR",
    "Supérieur au revenu médian" = "2.200 EUR",
    "Supérieur au revenu médian" = "2.250 EUR",
    "Supérieur au revenu médian" = "2.293 EUR",
    "Supérieur au revenu médian" = "2.300 EUR",
    "Supérieur au revenu médian" = "2.350 EUR",
    "Supérieur au revenu médian" = "2.367 EUR",
    "Supérieur au revenu médian" = "2.400 EUR",
    "Supérieur au revenu médian" = "2.440 EUR",
    "Supérieur au revenu médian" = "2.450 EUR",
    "Supérieur au revenu médian" = "2.458 EUR",
    "Supérieur au revenu médian" = "2.496 EUR",
    "Supérieur au revenu médian" = "2.500 EUR",
    "Supérieur au revenu médian" = "2.590 EUR",
    "Supérieur au revenu médian" = "2.600 EUR",
    "Supérieur au revenu médian" = "2.650 EUR",
    "Supérieur au revenu médian" = "2.700 EUR",
    "Supérieur au revenu médian" = "2.776 EUR",
    "Supérieur au revenu médian" = "2.800 EUR",
    "Supérieur au revenu médian" = "2.870 EUR",
    "Supérieur au revenu médian" = "2.900 EUR",
    "Supérieur au revenu médian" = "3.000 EUR",
    "Supérieur au revenu médian" = "3.099 EUR",
    "Supérieur au revenu médian" = "3.100 EUR",
    "Supérieur au revenu médian" = "3.130 EUR",
    "Supérieur au revenu médian" = "3.200 EUR",
    "Supérieur au revenu médian" = "3.300 EUR",
    "Supérieur au revenu médian" = "3.400 EUR",
    "Supérieur au revenu médian" = "3.430 EUR",
    "Supérieur au revenu médian" = "3.450 EUR",
    "Supérieur au revenu médian" = "3.500 EUR",
    "Supérieur au revenu médian" = "3.600 EUR",
    "Supérieur au revenu médian" = "3.700 EUR",
    "Supérieur au revenu médian" = "3.740 EUR",
    "Supérieur au revenu médian" = "3.756 EUR",
    "Supérieur au revenu médian" = "3.800 EUR",
    "Supérieur au revenu médian" = "4.000 EUR",
    "Supérieur au revenu médian" = "4.100 EUR",
    "Supérieur au revenu médian" = "4.180 EUR",
    "Supérieur au revenu médian" = "4.200 EUR",
    "Supérieur au revenu médian" = "4.300 EUR",
    "Supérieur au revenu médian" = "4.400 EUR",
    "Supérieur au revenu médian" = "4.500 EUR",
    "Supérieur au revenu médian" = "4.600 EUR",
    "Supérieur au revenu médian" = "4.700 EUR",
    "Supérieur au revenu médian" = "4.800 EUR",
    "Supérieur au revenu médian" = "4.900 EUR",
    "Supérieur au revenu médian" = "5.000 EUR",
    "Supérieur au revenu médian" = "5.400 EUR",
    "Supérieur au revenu médian" = "5.500 EUR",
    "Supérieur au revenu médian" = "5.600 EUR",
    "Supérieur au revenu médian" = "6.000 EUR",
    "Supérieur au revenu médian" = "6.500 EUR",
    "Supérieur au revenu médian" = "7.000 EUR",
    "Supérieur au revenu médian" = "7.020 EUR",
    "Supérieur au revenu médian" = "7.500 EUR",
    "Supérieur au revenu médian" = "7.600 EUR",
    "Supérieur au revenu médian" = "7.800 EUR",
    "Supérieur au revenu médian" = "8.000 EUR",
    "Supérieur au revenu médian" = "8.400 EUR",
    "Supérieur au revenu médian" = "8.500 EUR",
    "Supérieur au revenu médian" = "9.000 EUR",
    "Supérieur au revenu médian" = "9.500 EUR",
    "Supérieur au revenu médian" = "10.000 EUR",
    "Supérieur au revenu médian" = "10.515 EUR",
    "Supérieur au revenu médian" = "11.000 EUR",
    "Supérieur au revenu médian" = "11.111 EUR",
    "Supérieur au revenu médian" = "12.000 EUR",
    "Supérieur au revenu médian" = "12.500 EUR",
    "Supérieur au revenu médian" = "13.000 EUR",
    "Supérieur au revenu médian" = "13.600 EUR",
    "Supérieur au revenu médian" = "14.352 EUR",
    "Supérieur au revenu médian" = "15.000 EUR",
    "Supérieur au revenu médian" = "15.600 EUR",
    "Supérieur au revenu médian" = "16.000 EUR",
    "Supérieur au revenu médian" = "16.200 EUR",
    "Supérieur au revenu médian" = "17.000 EUR",
    "Supérieur au revenu médian" = "17.500 EUR",
    "Supérieur au revenu médian" = "18.000 EUR",
    "Supérieur au revenu médian" = "18.600 EUR",
    "Supérieur au revenu médian" = "18.900 EUR",
    "Supérieur au revenu médian" = "19.000 EUR",
    "Supérieur au revenu médian" = "20.000 EUR",
    "Supérieur au revenu médian" = "20.400 EUR",
    "Supérieur au revenu médian" = "21.000 EUR",
    "Supérieur au revenu médian" = "21.085 EUR",
    "Supérieur au revenu médian" = "22.000 EUR",
    "Supérieur au revenu médian" = "22.100 EUR",
    "Supérieur au revenu médian" = "23.000 EUR",
    "Supérieur au revenu médian" = "23.560 EUR",
    "Supérieur au revenu médian" = "24.000 EUR",
    "Supérieur au revenu médian" = "25.000 EUR",
    "Supérieur au revenu médian" = "27.000 EUR",
    "Supérieur au revenu médian" = "28.020 EUR",
    "Supérieur au revenu médian" = "30.000 EUR",
    "Supérieur au revenu médian" = "35.000 EUR",
    "Supérieur au revenu médian" = "36.000 EUR",
    "Supérieur au revenu médian" = "38.000 EUR",
    "Supérieur au revenu médian" = "38.900 EUR",
    "Supérieur au revenu médian" = "40.000 EUR",
    "Supérieur au revenu médian" = "45.000 EUR",
    "Supérieur au revenu médian" = "46.000 EUR",
    "Supérieur au revenu médian" = "50.000 EUR",
    "Supérieur au revenu médian" = "51.000 EUR",
    "Supérieur au revenu médian" = "60.000 EUR",
    "Supérieur au revenu médian" = "70.020 EUR",
    "Supérieur au revenu médian" = "75.200 EUR",
    "Supérieur au revenu médian" = "130.000 EUR per month"
  )

tab <- svytable(~ V64 + FR_RINC_desc, DATA_W)
write_clip(lprop(tab))
chisq.test(tab)
cramer.v(tab)
fisher.test(tab)

tab <- svytable(~ F_BORN_desc + FR_RINC_desc, DATA_IW)
write_clip(tab)

# Religion
irec(DATA_FR$FR_RELIG)
## Recodage de DATA_FR$FR_RELIG en DATA_FR$FR_RELIG_desc
DATA_FR$FR_RELIG_desc <- DATA_FR$FR_RELIG |>
  fct_recode(
    "Non religieux" = "No religion",
    "Chrétiens et apparentés" = "Catholic",
    "Chrétiens et apparentés" = "Protestant",
    "Chrétiens et apparentés" = "Orthodox",
    "Chrétiens et apparentés" = "Other Christian",
    "Autres religions" = "Jewish",
    "Musulmans" = "Islamic",
    "Autres religions" = "Buddhist",
    "Autres religions" = "Hindu",
    "Autres religions" = "Other Asian Religions",
    "Autres religions" = "Other Religions",
    NULL = "Don't know"
  )

tab <- svytable(~ V64 + FR_RELIG_desc, DATA_W)
write_clip(lprop(tab))
chisq.test(tab)
cramer.v(tab)
fisher.test(tab)

irec(DATA_I$FR_RELIG)
## Recodage de DATA_I$FR_RELIG en DATA_I$FR_RELIG_desc
DATA_I$FR_RELIG_desc <- DATA_I$FR_RELIG |>
  fct_recode(
    "Non religieux" = "No religion",
    "Chrétiens et apparentés" = "Catholic",
    "Chrétiens et apparentés" = "Protestant",
    "Chrétiens et apparentés" = "Orthodox",
    "Chrétiens et apparentés" = "Other Christian",
    "Autres religions" = "Jewish",
    "Musulmans" = "Islamic",
    "Autres religions" = "Buddhist",
    "Autres religions" = "Hindu",
    "Autres religions" = "Other Asian Religions",
    "Autres religions" = "Other Religions",
    NULL = "Don't know"
  )

tab <- svytable(~ F_BORN_desc + FR_RELIG_desc, DATA_IW)
write_clip(tab)
chisq.test(tab)
cramer.v(tab)
fisher.test(tab)

# Chômage
irec(DATA_FR$WORK)
## Recodage de DATA_FR$WORK en DATA_FR$WORK_desc
DATA_FR$WORK_desc <- DATA_FR$WORK |>
  fct_recode(
    NULL = "Never had paid work"
  )

tab <- svytable(~ V64 + WORK_desc, DATA_W)
write_clip(lprop(tab))
chisq.test(tab)
cramer.v(tab)
fisher.test(tab)

irec(DATA_I$WORK)
## Recodage de DATA_I$WORK en DATA_I$WORK_desc
DATA_I$WORK_desc <- DATA_I$WORK |>
  fct_recode(
    NULL = "Never had paid work"
  )

tab <- svytable(~ F_BORN_desc + WORK_desc, DATA_IW)
write_clip(lprop(tab))

# Relation d'emploi
# Recodage de DATA_FR$EMPREL en DATA_FR$EMPREL_desc
DATA_FR$EMPREL_desc <- DATA_FR$EMPREL |>
  fct_recode(
    "Travaille pour quelqu'un d'autre" = "Employee",
    "Propre employé" = "Self-employed without employees",
    "Propre employé" = "Self-employed with employees",
    "Propre employé" = "Working for own family's business"
  )

tab <- svytable(~ V64 + EMPREL_desc, DATA_W)
write_clip(lprop(tab))
chisq.test(tab)
cramer.v(tab)
fisher.test(tab)

# Recodage de DATA_I$EMPREL en DATA_I$EMPREL_desc
DATA_I$EMPREL_desc <- DATA_I$EMPREL |>
  fct_recode(
    "Travaille pour quelqu'un d'autre" = "Employee",
    "Propre employé" = "Self-employed without employees",
    "Propre employé" = "Self-employed with employees",
    "Propre employé" = "Working for own family's business"
  )

tab <- svytable(~ F_BORN_desc + EMPREL_desc, DATA_IW)
write_clip(lprop(tab))

# Vote
irec(DATA_FR$FR_PRTY)
## Recodage de DATA_FR$FR_PRTY en DATA_FR$FR_PRTY_desc
DATA_FR$FR_PRTY_desc <- DATA_FR$FR_PRTY |>
  fct_recode(
    NULL = "Worker's Struggle - LO - Nathalie Arthaud, far left",
    NULL = "New Anticapitalist Party - NPA - Philippe Poutou",
    "Jean-Luc Mélenchon (PG)" = "Left Front - FG - Jean-Luc Mélenchon",
    "François Hollande (PS)" = "Socialist Party - PS - Francois Hollande",
    NULL = "Green Party - EELV - Eva Joly",
    NULL = "Democratic Movement - MoDem - François Bayrou",
    "Nicolas Sarkozy (UMP)" = "Union for a Popular Movement - UMP - Nicolas Sarkozy",
    NULL = "Arise the Republic - DLR - Nicolas Dupont-Aignan",
    "Marine Le Pen (FN)" = "National front - FN - Marine Le Pen",
    NULL = "Solidarity and Progress - Jacques Cheminade",
    NULL = "Invalid ballot, Vote blank"
  )

tab <- svytable(~ V64 + FR_PRTY_desc, DATA_W)
write_clip(lprop(tab))
chisq.test(tab)
cramer.v(tab)

residu <- chisq.test(svytable(~ V64 + FR_PRTY_desc, DATA_W))
write_clip(residu$stdres)

# En fonction de l'origine du père
irec(DATA_I$FR_PRTY)
## Recodage de DATA_I$FR_PRTY en DATA_I$FR_PRTY_desc
DATA_I$FR_PRTY_desc <- DATA_I$FR_PRTY |>
  fct_recode(
    NULL = "Worker's Struggle - LO - Nathalie Arthaud, far left",
    NULL = "New Anticapitalist Party - NPA - Philippe Poutou",
    "Jean-Luc Mélenchon (PG)" = "Left Front - FG - Jean-Luc Mélenchon",
    "François Hollande (PS)" = "Socialist Party - PS - Francois Hollande",
    NULL = "Green Party - EELV - Eva Joly",
    NULL = "Democratic Movement - MoDem - François Bayrou",
    "Nicolas Sarkozy (UMP)" = "Union for a Popular Movement - UMP - Nicolas Sarkozy",
    NULL = "Arise the Republic - DLR - Nicolas Dupont-Aignan",
    "Marine Le Pen (FN)" = "National front - FN - Marine Le Pen",
    NULL = "Solidarity and Progress - Jacques Cheminade",
    NULL = "Invalid ballot, Vote blank"
  )

tab <- svytable(~ F_BORN_desc + FR_PRTY_desc, DATA_IW)
write_clip(lprop(tab))
fisher.test(tab)

residu <- chisq.test(svytable(~ F_BORN_desc + FR_PRTY_desc, DATA_IW))
write_clip(residu$stdres)


# -----------------
# 3 - Identification d'un proxy du vote FN parmi les descendant•e•s d'immigrés
# -----------------

irec(DATA_FR$FR_PRTY)
## Recodage de DATA_FR$FR_PRTY en DATA_FR$FR_PRTY_ACM
DATA_I$FR_PRTY_ACM <- DATA_I$FR_PRTY |>
  fct_recode(
    NULL = "Worker's Struggle - LO - Nathalie Arthaud, far left",
    NULL = "New Anticapitalist Party - NPA - Philippe Poutou",
    "Jean-Luc Mélenchon (PG)" = "Left Front - FG - Jean-Luc Mélenchon",
    "François Hollande (PS)" = "Socialist Party - PS - Francois Hollande",
    NULL = "Green Party - EELV - Eva Joly",
    NULL = "Democratic Movement - MoDem - François Bayrou",
    "Nicolas Sarkozy (UMP)" = "Union for a Popular Movement - UMP - Nicolas Sarkozy",
    NULL = "Arise the Republic - DLR - Nicolas Dupont-Aignan",
    "Marine Le Pen (FN)" = "National front - FN - Marine Le Pen",
    NULL = "Solidarity and Progress - Jacques Cheminade",
    NULL = "Invalid ballot, Vote blank"
  )

TAB_ACM <- DATA_I[, c("FR_PRTY_ACM", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37", "V38", "V39", "V40", "V41", "V42", "V43", "V44", "V45", "V46", "V47", "V48", "V49", "V50", "V51", "V52", "V53", "V54", "V55", "V56", "V57", "V58", "V59", "V60", "V61", "V62")]
TAB_ACM <- na.omit(TAB_ACM)
ACM <- MCA(TAB_ACM, quali.sup = 6, level.ventil = 0.05)
explor(ACM)

summary(DATA_I$V37)

irec(DATA_I$V37)
## Recodage de DATA_I$V37 en DATA_I$V37_rec
DATA_I$V37_rec <- DATA_I$V37 |>
  fct_recode(
    "4" = "Agree strongly",
    "3" = "Agree",
    "2" = "Neither agree nor disagree",
    "1" = "Disagree",
    "0" = "Disagree strongly"
  ) |>
  as.character() |>
  as.numeric()

irec(DATA_I$V19)
## Recodage de DATA_I$V19 en DATA_I$V19_rec
DATA_I$V19_rec <- DATA_I$V19 |>
  fct_recode(
    "4" = "Agree strongly",
    "3" = "Agree",
    "2" = "Neither agree nor disagree",
    "1" = "Disagree",
    "0" = "Disagree strongly"
  ) |>
  as.character() |>
  as.numeric()

irec(DATA_I$V13)
## Recodage de DATA_I$V13 en DATA_I$V13_rec
DATA_I$V13_rec <- DATA_I$V13 |>
  fct_recode(
    "4" = "Very important",
    "3" = "Fairly important",
    "2" = "Not very important",
    "1" = "Not important at all"
  ) |>
  as.character() |>
  as.numeric()

irec(DATA_I$V50)
## Recodage de DATA_I$V50 en DATA_I$V50_rec
DATA_I$V50_rec <- DATA_I$V50 |>
  fct_recode(
    "4" = "Agree strongly",
    "3" = "Agree",
    "2" = "Neither agree nor disagree",
    "1" = "Disagree",
    "0" = "Disagree strongly"
  ) |>
  as.character() |>
  as.numeric()

irec(DATA_I$PRTY_LR)

## Recodage de DATA_I$PARTY_LR en DATA_I$PARTY_LR_rec
DATA_I$PARTY_LR_linear <- DATA_I$PARTY_LR |>
  fct_recode(
    "0" = "Far left (communist etc.)",
    "1" = "Left, center left",
    "2" = "Center, liberal",
    "3" = "Right, conservative",
    "4" = "Far right (fascist etc.)",
    NULL = "Other",
    NULL = "Invalid ballot, Vote blank, No party affiliation (PH)"
  ) |>
  as.character() |>
  as.numeric()

irec(DATA_I$V56)

## Recodage de DATA_I$V56 en DATA_I$V56_rec
DATA_I$V56_rec <- DATA_I$V56 |>
  fct_recode(
    "0" = "Increased a lot",
    "1" = "Increased a little",
    "2" = "Remain the same as it is",
    "3" = "Reduced a little",
    "4" = "Reduced a lot"
  ) |>
  as.character() |>
  as.numeric()

irec(DATA_I$V38)

## Recodage de DATA_I$V38 en DATA_I$V38_rec
DATA_I$V38_rec <- DATA_I$V38 |>
  fct_recode(
    "4" = "Agree strongly",
    "3" = "Agree",
    "2" = "Neither agree nor disagree",
    "1" = "Disagree",
    "0" = "Disagree strongly"
  ) |>
  as.character() |>
  as.numeric()

ggplot(DATA_I) +
  aes(x = V56_rec, y = PARTY_LR_rec) +
  geom_point(colour = "blue", alpha = .25) +
  geom_smooth(method = "lm") +
  labs(x = "Volonté de réduire l'immigration", y = "Positionnement à droite") +
  theme_light()

mod <- lm(PARTY_LR_rec ~ V50_rec, data = DATA_I)
mod

summary(DATA_I$V35)
cor.test(DATA_I$V50_rec, DATA_I$PARTY_LR_linear)
chisq.test(DATA_I$V50_rec, DATA_I$PARTY_LR_linear)

cor.test(DATA_I$V38_rec, DATA_I$PARTY_LR_linear)
chisq.test(DATA_I$V38_rec, DATA_I$PARTY_LR_linear)

irec(DATA_I$V19)

## Recodage de DATA_I$V19 en DATA_I$V19_rec
DATA_I$V19_rec <- DATA_I$V19 |>
  fct_recode(
    "4" = "Agree strongly",
    "3" = "Agree",
    "2" = "Neither agree nor disagree",
    "1" = "Disagree",
    "0" = "Disagree strongly"
  ) |>
  as.character() |>
  as.numeric()

cor.test(DATA_I$V19_rec, DATA_I$PARTY_LR_linear)
chisq.test(DATA_I$V19_rec, DATA_I$PARTY_LR_linear)

irec(DATA_I$V13)
## Recodage de DATA_I$V13 en DATA_I$V13_rec
DATA_I$V13_rec <- DATA_I$V13 |>
  fct_recode(
    "4" = "Very important",
    "3" = "Fairly important",
    "2" = "Not very important",
    "1" = "Not important at all"
  ) |>
  as.character() |>
  as.numeric()

cor.test(DATA_I$V13_rec, DATA_I$PARTY_LR_linear)
chisq.test(DATA_I$V13_rec, DATA_I$PARTY_LR_linear)

irec(DATA_I$V16)
## Recodage de DATA_I$V16 en DATA_I$V16_rec
DATA_I$V16_rec <- DATA_I$V16 |>
  fct_recode(
    "4" = "Very important",
    "3" = "Fairly important",
    "2" = "Not very important",
    "1" = "Not important at all"
  ) |>
  as.character() |>
  as.numeric()

cor.test(DATA_I$V16_rec, DATA_I$PARTY_LR_linear)
chisq.test(DATA_I$V16_rec, DATA_I$PARTY_LR_linear)

irec(DATA_I$V60)
## Recodage de DATA_I$V60 en DATA_I$V60_rec
DATA_I$V60_rec <- DATA_I$V60 |>
  fct_recode(
    "4" = "Agree strongly",
    "3" = "Agree",
    "2" = "Neither agree nor disagree",
    "1" = "Disagree",
    "0" = "Disagree strongly"
  ) |>
  as.character() |>
  as.numeric()

cor.test(DATA_I$V60_rec, DATA_I$PARTY_LR_linear)
chisq.test(DATA_I$V60_rec, DATA_I$PARTY_LR_linear)

irec(DATA_I$V35)
## Recodage de DATA_I$V35 en DATA_I$V35_rec
DATA_I$V35_rec <- DATA_I$V35 |>
  fct_recode(
    "4" = "Agree strongly",
    "3" = "Agree",
    "2" = "Neither agree nor disagree",
    "1" = "Disagree",
    "0" = "Disagree strongly"
  ) |>
  as.character() |>
  as.numeric()

cor.test(DATA_I$V35_rec, DATA_I$PARTY_LR_linear)
chisq.test(DATA_I$V35_rec, DATA_I$PARTY_LR_linear)

irec(DATA_I$V56)
## Recodage de DATA_I$V56 en DATA_I$V56_rec
DATA_I$V56_rec <- DATA_I$V56 |>
  fct_recode(
    "0" = "Increased a lot",
    "1" = "Increased a little",
    "2" = "Remain the same as it is",
    "3" = "Reduced a little",
    "4" = "Reduced a lot"
  ) |>
  as.character() |>
  as.numeric()

cor.test(DATA_I$V56_rec, DATA_I$PARTY_LR_linear)
chisq.test(DATA_I$V56_rec, DATA_I$PARTY_LR_linear)

cor.test(DATA_I$V38_rec, DATA_I$V56_rec)
cor.test(DATA_I$V50_rec, DATA_I$V56_rec)

irec(DATA_I$F_BORN)

## Recodage de DATA_I$F_BORN en DATA_I$F_BORN_reg
DATA_I$F_BORN_reg <- DATA_I$F_BORN |>
  fct_recode(
    "Hors UE" = "AF-Afghanistan",
    "Hors UE" = "AL-Albania",
    "Hors UE" = "DZ-Algeria",
    "Hors UE" = "AO-Angola",
    "Hors UE" = "AZ-Azerbaijan",
    "Hors UE" = "AR-Argentina",
    "Hors UE" = "AU-Australia",
    "Originaire UE" = "AT-Austria",
    "Hors UE" = "BD-Bangladesh",
    "Hors UE" = "AM-Armenia",
    "Originaire UE" = "BE-Belgium",
    "Hors UE" = "BO-Bolivia",
    "Hors UE" = "BA-Bosnia and Herzegovina",
    "Hors UE" = "BR-Botswana",
    "Hors UE" = "BR-Brazil",
    "Originaire UE" = "BG-Bulgaria",
    "Hors UE" = "MM-Myanmar",
    "Hors UE" = "BI-Burundi",
    "Hors UE" = "BY-Belarus",
    "Hors UE" = "KH-Camboya",
    "Hors UE" = "CM-Camerún",
    "Hors UE" = "CA-Canada",
    "Hors UE" = "CV-Cabo Verde",
    "Hors UE" = "LK-Sri Lanka",
    "Hors UE" = "CL-Chile",
    "Hors UE" = "CN-China",
    "Hors UE" = "TW-Taiwan",
    "Hors UE" = "CO-Colombia",
    "Hors UE" = "KM-Comoras",
    "Hors UE" = "CG-Congo (Republic of)",
    "Hors UE" = "CD-Congo (Democratic Republic of the)",
    "Originaire UE" = "HR-Croatia",
    "Hors UE" = "CU-Cuba",
    "Originaire UE" = "CS-Czechoslovakia",
    "Originaire UE" = "CZ-Czech Republic",
    "Originaire UE" = "DK-Denmark",
    "Hors UE" = "DO-Dominican Republic",
    "Hors UE" = "EC-Ecuador",
    "Hors UE" = "SC-El Salvador",
    "Hors UE" = "ET-Ethiopia",
    "Originaire UE" = "EE-Estonia",
    "Hors UE" = "FO-Faroe Islands (the)",
    "Originaire UE" = "FI-Finland",
    "Originaire UE" = "FR-France",
    "Hors UE" = "DJ-Djibouty (the Republic of)",
    "Hors UE" = "GE-Georgia",
    "Hors UE" = "PS-Palestine, State of",
    "Originaire UE" = "DE-Germany",
    "Hors UE" = "GH-Ghana",
    "Originaire UE" = "GR-Greece",
    "Originaire UE" = "GP-Guadalupe",
    "Hors UE" = "GT-Guatemala",
    "Hors UE" = "GN-Ghinea",
    "Hors UE" = "HT-Haití",
    "Hors UE" = "HN-Honduras",
    "Hors UE" = "HK-Hong Kong",
    "Originaire UE" = "HU-Hungary",
    "Hors UE" = "IS-Iceland",
    "Hors UE" = "IN-India",
    "Hors UE" = "ID-Indonesia",
    "Hors UE" = "IR-Iran",
    "Hors UE" = "IQ-Iraq",
    "Originaire UE" = "IE-Ireland",
    "Hors UE" = "IL-Israel",
    "Originaire UE" = "IT-Italy",
    "Hors UE" = "CI-Cote d'Ivoire",
    "Hors UE" = "JP-Japan",
    "Hors UE" = "KZ-Kazakhstan (the Republic of)",
    "Hors UE" = "JO-Jordan",
    "Hors UE" = "KE-Kenya (the Republic of)",
    "Hors UE" = "KR-Korea (South)",
    "Hors UE" = "KG-Kyrgyzstan (the Republic of)",
    "Hors UE" = "LA-Lao",
    "Hors UE" = "LB-Lebanon",
    "Originaire UE" = "LV-Latvia",
    "Hors UE" = "LY-Libya",
    "Originaire UE" = "LI-Liechtenstein (the Principality of )",
    "Originaire UE" = "LT-Lithuania",
    "Hors UE" = "MG-Madagascar",
    "Hors UE" = "MY-Malaysia",
    "Hors UE" = "ML-Mali",
    "Hors UE" = "MQ-Martinica",
    "Hors UE" = "MR-Mauritania",
    "Hors UE" = "MU-Mauricio",
    "Hors UE" = "MX-Mexico",
    "Hors UE" = "MC-Monaco",
    "Hors UE" = "MD-Moldova (the Republic of)",
    "Hors UE" = "ME-Montenegro",
    "Hors UE" = "MA-Morocco",
    "Hors UE" = "MZ-Mozambique",
    "Hors UE" = "NP-Nepal",
    "Originaire UE" = "NL-Netherlands",
    "Originaire UE" = "NC-Nueva Caledonia",
    "Hors UE" = "NZ-New Zealand",
    "Hors UE" = "NE-Niger",
    "Hors UE" = "NG-Nigeria",
    "Hors UE" = "NO-Norway",
    "Hors UE" = "PK-Pakistan",
    "Hors UE" = "PY-Paraguay",
    "Hors UE" = "PE-Peru",
    "Hors UE" = "PH-Philippines",
    "Originaire UE" = "PL-Poland",
    "Originaire UE" = "PT-Portugal",
    "Hors UE" = "GW-Guinea-Bissau",
    "Hors UE" = "PR-Puerto Rico",
    "Originaire UE" = "RE-Reunión",
    "Originaire UE" = "RO-Romania",
    "Hors UE" = "RU-Russia",
    "Hors UE" = "RW-Rwanda",
    "Hors UE" = "ST-Sao Tome and Principe",
    "Hors UE" = "SN-Senegal",
    "Hors UE" = "RS-Serbia",
    "Hors UE" = "SC-Seychelles",
    "Hors UE" = "SL-Sierra Leone (the Republic of)",
    "Hors UE" = "SK-Singapore (the Republic of)",
    "Originaire UE" = "SK-Slovak Republic",
    "Hors UE" = "VN-Vietnam",
    "Originaire UE" = "SI-Slovenia",
    "Hors UE" = "SO-Somalia",
    "Hors UE" = "ZA-South Africa",
    "Originaire UE" = "ES-Spain",
    "Hors UE" = "SS-South Sudan",
    "Hors UE" = "SD-Sudan",
    "Hors UE" = "SR-Suriname",
    "Originaire UE" = "SE-Sweden",
    "Hors UE" = "CH-Switzerland",
    "Hors UE" = "SY-Syrian Arab Republic",
    "Hors UE" = "TJ-Takijistan (the Republic of)",
    "Hors UE" = "TH-Thailand",
    "Hors UE" = "TT-Trinidad y Tobago",
    "Hors UE" = "TN-Tunisia",
    "Hors UE" = "TR-Turkey",
    "Hors UE" = "TR-Turkmenistan",
    "Hors UE" = "UA-Ukraine",
    "Hors UE" = "MK-Macedonia (the former Yugoslav Republic of)",
    "Hors UE" = "SU-USSR",
    "Hors UE" = "EG-Egypt",
    "Originaire UE" = "GB-Great Britain and/or United Kingdom",
    "Hors UE" = "JE-Jersey",
    "Hors UE" = "TZ-Tanzania (the United Republic of)",
    "Hors UE" = "US-United States",
    "Hors UE" = "BF-Furkina Faso",
    "Hors UE" = "UY-Uruguay",
    "Hors UE" = "UZ-Uzbekistan (the Republic of)",
    "Hors UE" = "VE-Venezuela",
    "Hors UE" = "YE-Yemen",
    "Hors UE" = "YU-Yugoslavia",
    "Hors UE" = "ZM-Zambia",
    "Hors UE" = "XK-Kosovo",
    "Hors UE" = "Other (CH,EE,JP,RU,TR,US)",
    "Originaire UE" = "CY-Cyprus"
  )

levels(DATA_I$F_BORN_reg)

# TEST PROXY V50

irec(DATA_I$V50)
## Recodage de DATA_I$V50 en DATA_I$V50_reg
DATA_I$V50_reg <- DATA_I$V50 |>
  fct_recode(
    "Agree" = "Agree strongly",
    NULL = "Neither agree nor disagree",
    "Disagree" = "Disagree strongly"
  )

levels(DATA_I$V50_reg)
DATA_I$V50_reg <- fct_relevel(DATA_I$V50_reg, "Disagree")
DATA_I$SEX_reg <- fct_relevel(DATA_I$SEX_reg, "Femme")
DATA_I$URBRURAL_reg <- fct_relevel(DATA_I$URBRURAL_reg, "Rural")
DATA_I$FR_RINC_reg <- fct_relevel(DATA_I$FR_RINC_reg, "Supérieur salaire médian")


logit <- glm(V50_reg ~ V64_papa + F_BORN_reg + V64_maman + SEX_reg + AGE_reg + FR_DEGR_reg + URBRURAL_reg + FR_RINC_reg + EMPREL_reg + FR_RELIG_reg, DATA_I, family = quasibinomial())
odds.ratio(logit)
write_clip(odds.ratio(logit))
ggcoef_table(logit, exponentiate = TRUE)

PseudoR2(logit, which = "McFadden")
summary(logit)

summary(DATA_I$V38)



irec(DATA_I$V38)

## Recodage de DATA_I$V38 en DATA_I$V38_reg
DATA_I$V38_reg <- DATA_I$V38 |>
  fct_recode(
    "Agree" = "Agree strongly",
    "Agree" = "Neither agree nor disagree",
    "Disagree" = "Disagree strongly"
  )

DATA_I$V38_reg <- fct_relevel(DATA_I$V38_reg, "Disagree")


logit <- glm(V38_reg ~ V64_papa + V64_maman + F_BORN_reg + SEX_reg + AGE_reg + FR_DEGR_reg + URBRURAL_reg + FR_RINC_reg + EMPREL_reg + FR_RELIG_reg, DATA_I, family = quasibinomial())
odds.ratio(logit)
write_clip(odds.ratio(logit))
ggcoef_table(logit, exponentiate = TRUE)

# MODELE 1

logit <- glm(V56_reg ~ V64_papa + V64_maman + SEX_reg + AGE_reg + FR_DEGR_reg + URBRURAL_reg + FR_RINC_reg + EMPREL_reg + FR_RELIG_reg, DATA_I, family = quasibinomial())
odds.ratio(logit)
write_clip(odds.ratio(logit))
ggcoef_table(logit, exponentiate = TRUE)

# MODELE 2

logit <- glm(V38_reg ~ V64_papa + V64_maman + F_BORN_reg + SEX_reg + AGE_reg + FR_DEGR_reg + URBRURAL_reg + FR_RINC_reg + EMPREL_reg + FR_RELIG_reg, DATA_I, family = quasibinomial())
odds.ratio(logit)
write_clip(odds.ratio(logit))
ggcoef_table(logit, exponentiate = TRUE)

summary(DATA_FR$FR_ETHN1)
summary(DATA_FR$F_BORN)

# MODELES DE FIRTH

library(logistf)
modele_firth <- logistf(V50_reg ~ V64_papa + V64_maman + F_BORN_reg + SEX_reg + AGE_reg + FR_DEGR_reg + URBRURAL_reg + FR_RINC_reg + EMPREL_reg + FR_RELIG_reg, data = DATA_I)
ggcoef_table(modele_firth, exponentiate = TRUE)

modele_firth <- logistf(V56_reg ~ V64_papa + V64_maman + F_BORN_reg + SEX_reg + AGE_reg + FR_DEGR_reg + URBRURAL_reg + FR_RINC_reg + EMPREL_reg + FR_RELIG_reg, data = DATA_I)
ggcoef_table(modele_firth, exponentiate = TRUE)


           
