import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import lifelines 
from lifelines import KaplanMeierFitter 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import seaborn as sns

# Configurer la page de l'application
st.set_page_config(layout="wide")
global data, val
val =""
#Initialise session state of une clé dataFrame pour passer travers menus
if "dataFrame" not in st.session_state :
    st.session_state.dataFrame = pd.DataFrame()

def load_data(file,delimiter,encode) :
    data = pd.read_csv(file, sep=delimiter, encoding=encode)
    return data

codecs = ['ascii','big5','big5hkscs','cp037','cp273','cp424','cp437','cp500','cp720','cp737','cp775','cp850','cp852','cp855',
          'cp856','cp857','cp858','cp860','cp861','cp862','cp863','cp864','cp865','cp866','cp869','cp874','cp875','cp932','cp949',
          'cp950','cp1006','cp1026','cp1125','cp1140','cp1250','cp1251','cp1252','cp1253','cp1254','cp1255','cp1256','cp1257','cp1258',
          'euc_jp','euc_jis_2004','euc_jisx0213','euc_kr','gb2312','gbk','gb18030','hz','iso2022_jp','iso2022_jp_1','iso2022_jp_2',
          'iso2022_jp_2004','iso2022_jp_3','iso2022_jp_ext','iso2022_kr','latin_1','iso8859_2','iso8859_3','iso8859_4','iso8859_5','iso8859_6',
          'iso8859_7','iso8859_8','iso8859_9','iso8859_10','iso8859_11','iso8859_13','iso8859_14','iso8859_15','iso8859_16','johab','koi8_r','koi8_t',
          'koi8_u','kz1048','mac_cyrillic','mac_greek','mac_iceland','mac_latin2','mac_roman','mac_turkish','ptcp154','shift_jis','shift_jis_2004',
          'shift_jisx0213','utf_32','utf_32_be','utf_32_le','utf_16','utf_16_be','utf_16_le','utf_7','utf_8','utf_8_sig']

codecsArray = np.array(codecs) 

def calculVal(method,serie) :
    if method == "mean" :
        val = serie.mean()
    elif method == "median" :
        val = serie.median()
    else :
        val = serie.mode()
    return val

# Sidebar paramétrage
with st.sidebar :
    st.header("Paramétrage ")

# Main affichage
selected = option_menu("Analyse de suivre", ["Lecture des données", "Traitement des données manquantes", "Statistiques descriptives",
        "Représentations graphiques des variables","Probabilités de suivre et courbres de suivre","Prédiction de suivre d'un individu",
        "Modèle de régression de Cox","Analyse coût-efficacité"], 
        icons=['book-fill', 'file-spreadsheet-fill', "clipboard-data", 'bar-chart-fill',"graph-down","person-bounding-box",
        "graph-up","currency-euro"], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={   
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "14px"}, 
        "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        })
selected

# Rubrique Lecture des données
if selected == "Lecture des données" :
    col1, col2 = st.columns(2)
    with col1 :
        delimiter = st.text_input("Entrer un délimiter pour un fichier CSV",placeholder="Par exemple ; ou , ou | ")
    with col2 :
        encode = st.selectbox("Sélectionner un codec :",codecsArray, index=95)
    uploaded_file = st.file_uploader("Télécharger le fichier ici")
    if uploaded_file is not None :
        if delimiter == "" :
            # delimiter par défault = ";"
            delimiter = ";"
        
        data = load_data(uploaded_file, delimiter,encode) 
        # Set session avec un key dataFrame pour contenir une varaible data
        st.session_state.dataFrame = data
        data    

# Rubrique Traitement des données manquantes
if selected =="Traitement des données manquantes" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        data
        data.dtypes
        columns = data.columns
        col1, col2 = st.columns(2)
        with col1 :
             selected_col = st.sidebar.selectbox("Sélectionner une colonne à traiter", columns, key="column")
             st.markdown("#### :orange[Transformer un type de donnée dans la conlonne sélectée]")
        #    selected_col = st.selectbox("Sélectionner une colonne à traiter", columns, key="column")
        with col2 :
            selected_type = st.selectbox("Sélectionner un type de donnée :",('object','numérique','category','datetime','bool'))
            if st.button("Transformer") and selected_col is not None :
                if selected_type == "object" or selected_type == "category" or selected_type == "bool" :
                    data[selected_col] = data[selected_col].astype(selected_type)
                elif selected_type == "numérique" :
                    data[selected_col] = pd.to_numeric(data[selected_col])
                else :
                    data[selected_col] = pd.to_datetime(data[selected_col], format='%d/%m/%Y')
                #st.session_state.dataFrame = data
        
        col3, col4 =st.columns(2)
        with col3 :
            st.markdown("#### Remplacer les données manquantes dans la colonne sélectionée par :")
        
        with col4 :
            if is_numeric_dtype(data[selected_col]) :
                radio_disabled = False
                text_disabled = True
            else :
                radio_disabled = True
                text_disabled = False

            text = st.text_input("Entrer un mot :", disabled=text_disabled)
            method = st.radio("Choisir une façon à remplacer :",("mean","median","mode"), key = "methodNumeric",disabled=radio_disabled)
            serie = st.session_state.dataFrame[st.session_state.column]
            if is_numeric_dtype(serie) :
                val = calculVal(method, serie)
                st.markdown(f"Valeur = {val}")
            if st.button("Remplacer") :
                if text != "" :
                    data[selected_col].fillna(text, inplace=True)
                elif method != "mode" :     
                    data[selected_col].fillna(val, inplace=True)
                else :
                    # Choisir le première valeur de mode trouvée
                    data[selected_col].fillna(val[0], inplace=True)
                st.experimental_rerun()
               
         #   st.session_state.dataFrame = data
         
        col5, col6 =st.columns(2)
        with col5 :
            st.markdown("#### ou supprimer les lignes de données manquantes en critère :")

        with col6 :
            delete_line = st.radio("""any : Si des valeurs NA sont présentes, supprimez cette ligne. 
                                    all : Si toutes les valeurs sont NA, supprimez cette ligne.""",
                         ('any','all'), index=1)
            st.error("Voulez-vous supprimer les lignes de données manquantes?", icon="🚨")
            if st.button("Oui, supprimer ces lignes") :
                if delete_line == 'any' :
                    data.dropna(axis=0, how='any', inplace=True)
                elif delete_line =='all' :
                    data.dropna(axis=0, how="all", inplace=True)
                st.experimental_rerun()

        col7, col8 =st.columns(2) 
        with col7 :
            st.markdown("#### ou supprimer les colonnes de données manquantes en critère :") 
        with col8 :
            delete_column = st.radio("""any : Si des valeurs NA sont présentes, supprimez cette colonne. 
                                    all : Si toutes les valeurs sont NA, supprimez cette colonne.""",
                         ('any','all'), index=1)
            st.error("Voulez-vous supprimer les colonnes de données manquantes?", icon="🚨")
            if st.button("Oui, supprimer ces colonnes") :
                if delete_column == 'any' :
                    data.dropna(axis=1, how='any', inplace=True)
                elif delete_column =='all' :
                    data.dropna(axis=1, how="all", inplace=True)
                st.experimental_rerun()
    # Après le traitement , on peut enregistrer les données en fichiers csv
    col9, col10 = st.columns(2)
    with col9 :
        st.markdown("### Enregistre le fichier :")
        delimiter = st.selectbox("Sélectionner un délimiteur :", (",",";","|"))
        st.download_button("Enregistre le fichier csv",
                        data.to_csv(sep=delimiter,encoding="utf_8"),
                        mime="text/csv") 

#Rubrique Statistiques descriptives
if selected == "Statistiques descriptives" :
    data = st.session_state.dataFrame
 #   st.sidebar.multiselect("Multi colonnes sélectables", data.columns)
    st.write(data.describe())

#Function à montrer des correlations entre variables  
@st.cache_data
def plotCorr(df) :
    matrix_corr = df.corr().round(2)
    fig = plt.figure(figsize=(6,6))
    sns.heatmap(data=matrix_corr,linewidths=0.3, annot=True)
    st.pyplot(fig)

#Function à dessiner des plots de correlations entre variables  
@st.cache_data 
def plotPairGraph(df,event) :
    plt.figure(figsize=(12,12)) 
    sns.pairplot(df,hue=event)
    st.pyplot(plt)

#Function à dessiner l'histogramme par type de l'ensemble    
@st.cache_data
def histogramme(data, col,limit='') :
    plt.figure(figsize=(5,5))
    if limit in data.columns :
        data.hist(column=col,by=limit)   
    else :
        data.hist(column=col) 
    st.pyplot(plt)

#Function à trouver les colonnes avec  un type numéric
def columsNumeric(data) :
    colNum = []
    if ~data.empty :
        for col in data.columns :
            if is_numeric_dtype(data[col]) :
                colNum.append(col)
    return colNum

#Rubrique Représentations graphiques des variables
if selected == "Représentations graphiques des variables" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        #Correlation heatmap
        st.subheader(":orange[Correlations entre variables quantitatives]") 
        df = data.iloc[:,1:]
        plotCorr(df)
        #Pairplot
        st.sidebar.subheader(":orange[Pairplot : Couleurs]")
        event = st.sidebar.selectbox("Sélectionner une colonne pour des couleurs", data.columns[1:])
        st.subheader(":orange[Pairplot]")
        st.warning("Veuillez sélectionner une colonne pour distinguer les graphiques en couleurs",icon="⚠️")
        if st.button("Afficher le pairplot") :
            plotPairGraph(df, event)
        #Histogramme
        #Créer une liste des colonnes numériques sur le sidebar
        colNum = columsNumeric(df)
   #     print("colNum = ",colNum)
        st.sidebar.subheader(":orange[Histogramme :]")
        histCol = st.sidebar.selectbox("Sélectionner une colonne", colNum)
        colNoNum = list(set(df.columns) - set(colNum))
        colNoNum.sort(key=str.lower)
  #      print("\ndata_cols = ",df.columns)
  #      print("colNoNum = ",colNoNum)
        checkCrit = st.sidebar.checkbox("avec un critère") 
        if checkCrit:
            limit = st.sidebar.selectbox("Sélectionner un critère par une colonne en dessous", colNoNum )
        st.subheader(":orange[Histogramme]")
        st.warning("Veuillez sélectionner une colonne pour dessiner l'histrograme et/ou avec un critère",icon="⚠️")
        if st.button("Afficher l'histogramme") :
            if checkCrit :
                histogramme(data, histCol, limit)
            else :
                histogramme(data, histCol)