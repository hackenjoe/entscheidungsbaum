# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:46:05 2021

@author: Pascal
"""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from dtreeviz.trees import dtreeviz
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
#import SessionState
#plt.rcParams.update({'font.size': 14})

## init
st.set_page_config(
    page_title="Entscheidungsbaum Generator",
    page_icon=":deciduous_tree:",
    layout="wide",
    )
st.title("Entscheidungsbaum Generator :deciduous_tree:")
    
## sidebar
st.sidebar.title("Konfiguration")
rand_depth = 2
rand_split = 35

st.sidebar.header("1. Trainingsdaten")
input_split = st.sidebar.empty()
input_split_value = st.sidebar.empty()

st.sidebar.header("2. Baumtiefe")
input_num = st.sidebar.empty()

btn_random = st.sidebar.button(label="Zufällige Parameter")
if (btn_random):
    rand_depth = np.random.randint(100, size=1)[0]
    rand_split = np.random.choice(np.arange(5,100,5))
    
    
option_split = input_split.slider(
     label="Aufteilung in Trainings- und Testdaten",
     min_value=5,
     value=int(rand_split),
     max_value=100,
     step=5,
     format="%i%%",
     help="Hier können Sie den prozentualen Anteil der Trainings- und Testdaten bestimmen. Wir haben __1000__ Datensätze.",
     )

input_split_value.text(f"Trainingsmenge: {option_split}% \nTestmenge: {100-option_split}%")

option_depth = input_num.number_input(
     label="Die gewünschte Baumtiefe",
     min_value=0,
     max_value=100,
     value=rand_depth,
     help="Hier können Sie die maximale Baumtiefe angeben. **Hinweis:** Eine $0$ bedeutet keine Beschränkung der Baumtiefe.",
    )
    

## main window
"1. Sie können die Parameter für Ihren Entscheidungsbaum links im Navigationsmenü einstellen."
"2. Testen Sie verschiedene Parameterkonstellationen. Können Sie einen Entscheidungsbaum erzeugen, welcher eine höhere Testgenauigkeit als $80\%$ aufweist?"
"3. Probieren Sie auch eigene Testdatensätze. Erhält jemand der erst kürzlich einen Kredit beantragte wieder einen Kredit?"

st.subheader("Kreditdaten")
df = pd.read_csv(r"german_credits.csv", sep=",")

mask_cols = ["Kontosaldo",
             "Alter_in_Jahren",
             "Kredithöhe",
             "Kreditdauer_monatlich",
             "Bezahlstatus_vorheriger_Kredit",
             "Wertaktien",
             "Beschäftigungsdauer",
             "Geschlecht_Familienstand",
             "Anzahl_Bankkredite",
             "Gastarbeiter",
             "Weitere_aktive_Kredite",
             "Bürgen",
             "Zweck"]

df[mask_cols]

X = df[mask_cols].values
y = df.iloc[:,0].values

# some default parameters
test_size = (100-option_split)/100
if (test_size == 0):
    test_size = 0.01
if (option_depth == 0):
    option_depth = None
if (option_depth <= 5) & (option_depth > 0):
    dtree_height = 750
    dtree_width = 1400
else:
    dtree_height = 1080
    dtree_width = 1920

# code
st.subheader("Code")
# everything inside this block will be also printed
with st.echo():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    dtree = DecisionTreeClassifier(criterion='gini', max_depth=option_depth, random_state=42)
    dtree = dtree.fit(X_train, y_train)
    dot_data = tree.export_graphviz(dtree, feature_names=mask_cols, class_names=["abgelehnt", "angenommen"], filled=True, precision=2)
    plt.tight_layout()
    st.subheader("Visualisierung")
    st.graphviz_chart(dot_data, use_container_width=False)

print(f"Trainingsdaten\t: {X_train.shape[0]} Datensätze\nTestdaten\t: {X_test.shape[0]} Datensätze")

# And now we're back to _not_ printing to the screen
st.markdown("***")

label_acc = st.sidebar.text("Vorhersagegenauigkeit: {0:.1%}".format(dtree.score(X_test, y_test)))
print("Die durchschnittliche Vorhersagegenauigkeit des Modells beträgt ca. {0:.1%}".format(dtree.score(X_test, y_test)))

check_detail = st.checkbox("Detailansicht", help="Zeigt die Verteilungen und Histogramme an.")

def st_dtree(plot, height=None, width=None):
    dtree_html = f"<body>{viz.svg()}</body>"
    components.html(dtree_html, height=height, width=width, scrolling=True)

#@st.cache
def view_tab(viz):
    viz.view()
    
if (check_detail):
    viz = dtreeviz(dtree,
               X_train,
               y_train,
               target_name="Kreditwürdigkeit",
               feature_names=mask_cols,
               class_names=["abgelehnt", "angenommen"],
               title="Entscheidungsbaum Verbraucherkredit",
               fontname="Arial",
               title_fontsize=16,
             #  fancy=False,
               scale=1.5,
               colors={"title":"darkblue",
                       "highlight":"darkgreen"})
    #st.image(viz._repr_svg_(), use_column_width=True)
    st_dtree(viz, dtree_height, dtree_width)
    btn_view = st.button(label="Im neuen Tab öffnen")
    if (btn_view):
        view_tab(viz)
    st.markdown("***")
    


# testing
st.subheader("Testdatensatz")
username = st.text_input(label="Name des Kreditnehmers?", value="Jan Gustafsson")
st.write("Kreditnehmer: ", "Jan Gustafsson" if username == "" else username)
df_kontosaldo = st.selectbox('Kontosaldo', ('kein Konto vorhanden', 'kein Guthaben', 'weniger als 200 Euro', 'über 200 Euro'))
df_alter = st.slider('Alter in Jahren', min_value=18, max_value=90, step=1)
df_kredithoehe = st.number_input('Kredithöhe', min_value=1, max_value=50000,)
df_kreditdauer = st.slider('Kreditlaufzeit in Monaten', min_value=1, max_value=84, step=1, value=48)
df_bezahlstatus = st.selectbox('Bezahlstatus vorheriger Kredit', ('ausstehend', 'andere Kredite', 'eingezahlt', 'kein Problem mit gegenwärtigen Krediten', 'alte Kredite beglichen'))
df_wertaktien = st.selectbox('Vorhandene Wertaktien', ('keine', 'unter 100 Euro', 'zwischen 100 und 500 Euro', 'zwischen 500 und 1000 Euro', 'über 1000 Euro'))
df_beschaeftigungsdauer = st.selectbox('Beschäftigungsdauer', ('nicht beschäftigt', 'unter einem Jahr', 'zwischen einem Jahr und 4 Jahren', 'zwischen 4 und 7 Jahren', 'über 7 Jahren'))
df_familienstand = st.selectbox('Aktueller Familienstand', ('männlich und geschieden', 'männlich single', 'männlich verheiratet oder verwitwet', 'weiblich'))
df_anzahl_kredite = st.selectbox('Anzahl Bankkredite', ('einen einzigen', 'zwei oder drei Kredite', '4 oder 5 Kredite', 'über 6 Kredite'))
df_gastarbeiter = st.selectbox('Gastarbeiter', ('ja', 'nein'))
df_weitere_aktive_kredite = st.selectbox('Weitere aktive Kredite', ('bei einer anderen Bank', 'Kaufhaus', 'keine'))
df_buergen = st.selectbox('Bürgen', ('keinen', 'auch Kreditnehmer', 'Bürgschaft erfüllt'))
df_zweck = st.selectbox('Grund für den Kredit', ('neues Auto', 'Gebrauchtwagen', 'Möbel', 'Radio/TV', 'Haushaltsgeräte', "Reparaturen", "Urlaub", "Fortbildung", "Business", "Sonstiges"))


st.write('Ihr ausgewählter Datensatz:')

new_cols = mask_cols.copy()
new_cols.insert(0, "Name")

df_test = pd.DataFrame([[username, df_kontosaldo, df_alter, df_kredithoehe, df_kreditdauer, df_bezahlstatus, 
                        df_wertaktien, df_beschaeftigungsdauer, df_familienstand, df_anzahl_kredite,
                        df_gastarbeiter, df_weitere_aktive_kredite, df_buergen, df_zweck]], columns=new_cols)

mapping_dict = {
    'Kontosaldo' : {'kein Konto vorhanden': 1, 'kein Guthaben': 2, 'weniger als 200 Euro': 3, 'über 200 Euro': 4},
    'Bezahlstatus_vorheriger_Kredit' : {'ausstehend': 0, 'andere Kredite': 1, 'eingezahlt': 2, 'kein Problem mit gegenwärtigen Krediten': 3, 'alte Kredite beglichen': 4},
    'Wertaktien': {'keine': 1, 'unter 100 Euro': 2, 'zwischen 100 und 500 Euro': 3, 'zwischen 500 und 1000 Euro': 4, 'über 1000 Euro': 5},
    'Beschäftigungsdauer': {'nicht beschäftigt': 1, 'unter einem Jahr': 2, 'zwischen einem Jahr und 4 Jahren': 3, 'zwischen 4 und 7 Jahren': 4, 'über 7 Jahren': 5},
    'Geschlecht_Familienstand': {'männlich und geschieden': 1, 'männlich single': 2, 'männlich verheiratet oder verwitwet': 3, 'weiblich': 4},
    'Anzahl_Bankkredite': {'einen einzigen': 1, 'zwei oder drei Kredite': 2, '4 oder 5 Kredite': 3, 'über 6 Kredite': 4},
    'Gastarbeiter': {'ja': 1, 'nein': 2},
    'Weitere_aktive_Kredite': {'bei einer anderen Bank': 1, 'Kaufhaus': 2, 'keine': 3},
    'Bürgen': {'keinen': 1, 'auch Kreditnehmer': 2, 'Bürgschaft erfüllt': 3},
    'Zweck': {'neues Auto': 0, 'Gebrauchtwagen': 1, 'Möbel': 2, 'Radio/TV': 3, 'Haushaltsgeräte': 4, 'Reparaturen': 5, 'Urlaub': 6, 'Fortbildung': 8, 'Business': 9, 'Sonstiges': 10},
    }


df_test = df_test.replace(to_replace=mapping_dict, value=None)

result = np.argmax(dtree.predict_proba(df_test.replace(to_replace=mapping_dict, value=None).values[:, 1:]))
risk = dtree.predict_proba(df_test.replace(to_replace=mapping_dict, value=None).values[:, 1:])[:, result]
if (result == 0):
    result_str = "abgelehnt"
    risk_str = "sehr hohes"
else:
    result_str = "akzeptiert"
    if (risk >= 0.5) & (risk <= 0.65):
        risk_str = "hohes"
    elif (risk > 0.65) & (risk <= 0.75):
        risk_str = "mäßiges"
    else:
        risk_str = "geringes"
    
st.text(f'Der Kredit für {username} wird wahrscheinlich {result_str}!\nEs existiert ein {risk_str} Kreditrisiko.')
#st.info("ausgabe")

