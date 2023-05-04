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
import math
from lifelines import BreslowFlemingHarringtonFitter
from lifelines import NelsonAalenFitter
from lifelines import WeibullFitter
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from io import StringIO
import sys
import pickle

# Configurer la page de l'application
st.set_page_config(layout = "wide")
global data, val
val = ""
#Initialise session state of une clé dataFrame pour passer travers menus
if "dataFrame" not in st.session_state :
    st.session_state.dataFrame = pd.DataFrame()

def load_data(file, delimiter, encode) :
    data = pd.read_csv(file, sep = delimiter, encoding = encode)
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

def calculVal(method, serie) :
    if method == "mean" :
        val = serie.mean()
    elif method == "median" :
        val = serie.median()
    else :
        val = serie.mode()
    return val

def makeTextMode(serie) :
    text=""
    for val in serie :
        text += str(val)+" "
    text = text.strip()
    return text

# Sidebar paramétrage
with st.sidebar :
    st.header("Paramétrage ")

# Main affichage
selected = option_menu("Analyse de survie", ["Lecture des données", "Traitement des données manquantes", "Statistiques descriptives",
        "Représentations graphiques des variables","Probabilités de survie et courbes de survie","Prédiction de survie d'un individu",
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
        delimiter = st.text_input("Entrer un délimiter pour un fichier CSV", placeholder = "Par exemple ; ou , ou | ")
    with col2 :
        encode = st.selectbox("Sélectionner un codec :", codecsArray, index = 95)
    uploaded_file = st.file_uploader("Télécharger le fichier ici")
    if uploaded_file is not None :
        if delimiter == "" :
            # delimiter par défault = ";"
            delimiter = ";"
        
        data = load_data(uploaded_file, delimiter, encode) 
        # Set session avec un key dataFrame pour contenir une varaible data
        st.session_state.dataFrame = data
        st.dataframe(data.head()) 

# sauvegarder le dataframe d'origine pour undo le traitement
st.session_state.oldDataframe = pickle.loads(pickle.dumps(st.session_state.dataFrame))  
#print("avant de traitement, oldDf",st.session_state.oldDataframe)
# Rubrique Traitement des données manquantes
data = st.session_state.dataFrame
#data = data.convert_dtypes()
if selected =="Traitement des données manquantes" :
    if len(data.columns) != 0 :
        data
        df_miss_data = pd.DataFrame({'Cols' : data.columns, 'Types' : np.array(data.dtypes), 'NANs' : np.array(data.isna().sum())})
        df_miss_data
        #df_miss_data.dtypes
   #     print("data",data.loc[6,"AdherenciaTtoMM1Mto1"])
        #Undo button pour revenir la version dernière de data
   #     if st.button("Undo le traitement") :
   #        print("oldDf",st.session_state.oldDataframe.loc[6,"AdherenciaTtoMM1Mto1"])
   #        data = pickle.loads(pickle.dumps(st.session_state.oldDataframe)) 
   #        print("data après undo",data.loc[6,"AdherenciaTtoMM1Mto1"])
   #        st.experimental_rerun()
        columns = data.columns
        col1, col2 = st.columns(2)
        with col1 :
             selected_col = st.sidebar.selectbox("Sélectionner une colonne à traiter", columns, key = "column")
             st.markdown("#### :orange[Transformer un type de donnée dans la colonne sélectionnée]")
        with col2 :
            selected_type = st.selectbox("Sélectionner un type de donnée :",('object', 'numérique', 'category', 'datetime', 'bool'))
            if st.button("Transformer") and selected_col is not None :
     #           st.session_state.oldDataframe = pickle.loads(pickle.dumps(data))
                if selected_type == "object" or selected_type == "category" or selected_type == "bool" :
                    data[selected_col] = data[selected_col].astype(selected_type)
                elif selected_type == "numérique" :
                    data[selected_col] = pd.to_numeric(data[selected_col])
                else :
                    data[selected_col] = pd.to_datetime(data[selected_col], format = '%d/%m/%Y')
        
        col3, col4 =st.columns(2)
        with col3 :
            st.markdown("#### :orange[Remplacer les données manquantes dans la colonne sélectionnée par :]")
        
        with col4 :
            if is_numeric_dtype(data[selected_col]) :
                radio_disabled = False
                selected_text_disabled = True     
            else :
                radio_disabled = True
                selected_text_disabled = False
               
            text_disabled = True
            texts = data[selected_col].dropna().unique()
            list_selected_text = list(texts)
            list_selected_text.append("Autre")
            if not(selected_text_disabled) :
                modeSerie = data[selected_col].dropna().mode()
                textMode = makeTextMode(modeSerie)
                st.markdown(f'Le mode = :orange[{textMode}]') 
            selected_text = st.selectbox("Sélectionner une value existante dans la colonne :", list_selected_text, disabled = selected_text_disabled)
            print(selected_text)
            if selected_text == "Autre" :
                text_disabled = False
            text = st.text_input("Entrer un mot :", disabled=text_disabled)
            method = st.radio("Choisir une façon à remplacer :",("mean","median","mode"), key = "methodNumeric", disabled = radio_disabled)
            serie = st.session_state.dataFrame[st.session_state.column]
            if is_numeric_dtype(serie) :
                val = calculVal(method, serie)
                if isinstance(val, pd.Series) :
                    textMode = makeTextMode(val)
                    st.markdown(f'Valeur = {textMode}')
                else :
                    st.markdown(f"Valeur = {val}")
            if st.button("Remplacer") :
    #            st.session_state.oldDataframe = pickle.loads(pickle.dumps(data))
            #    print("oldDf",st.session_state.oldDataframe.loc[6,"AdherenciaTtoMM1Mto1"])
                if text != "" :
                    data[selected_col].fillna(text, inplace = True)
                elif not(selected_text_disabled) and selected_text != "Autre" :
                    data[selected_col].fillna(selected_text, inplace = True)
                elif method != "mode" :     
                    data[selected_col].fillna(val, inplace = True)
                else :
                    # Choisir le première valeur de mode trouvée
                    data[selected_col].fillna(val[0], inplace = True)
                st.experimental_rerun()
         
        col5, col6 = st.columns(2)
        with col5 :
            st.markdown("#### :orange[ou supprimer les lignes de données manquantes en critère :]")

        with col6 :
            delete_line = st.radio("""any : Si des valeurs NA sont présentes, supprimez cette ligne. 
                                    all : Si toutes les valeurs sont NA, supprimez cette ligne.""",
                         ('any','all'), index=1)
            st.error("Voulez-vous supprimer les lignes de données manquantes?", icon="🚨")
            if st.button("Oui, supprimer ces lignes") :
    #            st.session_state.oldDataframe = pickle.loads(pickle.dumps(data))
                if delete_line == 'any' :
                    data.dropna(axis = 0, how = 'any', inplace = True)
                elif delete_line =='all' :
                    data.dropna(axis = 0, how = "all", inplace = True)
                st.experimental_rerun()

        col7, col8 =st.columns(2) 
        with col7 :
            st.markdown("#### :orange[ou supprimer les colonnes de données manquantes en critère :]") 
        with col8 :
            delete_column = st.radio("""any : Si des valeurs NA sont présentes, supprimez cette colonne. 
                                    all : Si toutes les valeurs sont NA, supprimez cette colonne.""",
                         ('any','all'), index=1)
            st.error("Voulez-vous supprimer les colonnes de données manquantes?", icon="🚨")
            if st.button("Oui, supprimer ces colonnes") :
    #            st.session_state.oldDataframe = pickle.loads(pickle.dumps(data))
                if delete_column == 'any' :
                    data.dropna(axis = 1, how = 'any', inplace = True)
                elif delete_column =='all' :
                    data.dropna(axis=1, how = "all", inplace = True)
                st.experimental_rerun()
    # Après le traitement , on peut enregistrer les données en fichiers csv
    col9, col10 = st.columns(2)
    with col9 :
        st.markdown("### :orange[Enregistre le fichier :]")
        delimiter = st.selectbox("Sélectionner un délimiteur :", (",",";","|"))
        st.download_button("Enregistre le fichier csv",
                        data.to_csv(sep = delimiter, encoding = "utf_8"),
                        mime = "text/csv") 

#Rubrique Statistiques descriptives
st.session_state.dataFrame = data
if selected == "Statistiques descriptives" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        st.write(data.describe())

#Fonction à montrer des correlations entre variables  
@st.cache_data
def plotCorr(df) :
    matrix_corr = df[columsNumeric(df)].corr().round(2)
    fig = plt.figure(figsize = (6, 6))
    sns.heatmap(data = matrix_corr, linewidths = 0.3, annot = True)
    st.pyplot(fig)

#Fonction à dessiner des plots de correlations entre variables  
@st.cache_data 
def plotPairGraph(df, event) :
    plt.figure(figsize = (12, 12)) 
    sns.pairplot(df,hue = event)
    st.pyplot(plt)

#Fonction à dessiner l'histogramme par type de l'ensemble    
@st.cache_data
def histogramme(data, col, limit = '') :
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

#Fonction à trouver les colonnes non numérique
def columnsNoNum(df,colNum) :
    colNoNum = list(set(df.columns) - set(colNum))
    colNoNum.sort(key=str.lower)
    return colNoNum

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
        st.sidebar.subheader(":orange[Histogramme :]")
        histCol = st.sidebar.selectbox("Sélectionner une colonne", colNum)
        colNoNum = columnsNoNum(df, colNum)
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

#Fonction à trouver et à afficher la survival function par Kaplan-Meier
@st.cache_data
def kaplanMeyer(data, col_duration, col_event, montre_ci, crit, isGrid, col_ent_tard="") :
    title_table = "Proportion de survivants et confidence interval à l'instant t"
    title_graph = "Survival function"
    kmf = KaplanMeierFitter()
    #Si one ne conside pas les entrées tardives
    if col_ent_tard == "" :
        if crit == "":
            kmf.fit(data[col_duration], data[col_event])
            #Afficher des tables de suivival function et confidence interval
            st.subheader(title_table)
            survival = kmf.survival_function_
            ci = kmf.confidence_interval_
            table = survival.join(ci)
            table
            #Réprrésenter graphiquement la courbre de survie avec l'interval de confiance
            ax = plt.subplot(2, 2, 1)
            kmf.plot_survival_function(ax = ax, title = title_graph, ci_show = montre_ci)
            ax.set_ylabel("Probabilités de survie, S(t)", fontsize = "x-small")
            st.pyplot(fig=plt)
        else :
            if isGrid :
                plotKmfGrid_survival_function(data, col_duration, col_event, montre_ci, crit)
            else :
                plotKmf_survival_function(data,col_duration,col_event,montre_ci,crit)
    else :
        if crit == "":
            kmf.fit(data[col_duration], data[col_event], entry = data[col_ent_tard], label = "Modéle d'entrée tardive")
            #Afficher des tables de suivival function et confidence interval
            st.subheader(title_table)
            survival = kmf.survival_function_
            ci = kmf.confidence_interval_
            table = survival.join(ci)
            table
            #Réprrésenter graphiquement la courbre de survie avec l'interval de confiance
            ax =plt.subplot(1,1,1)
            kmf.plot_survival_function(ax = ax, title = title_graph, ci_show = montre_ci)

            kmf.fit(data[col_duration], data[col_event], label = "Ignore les entrées tardives")
            kmf.plot_survival_function(ax = ax, ci_show = montre_ci)
            ax.set_ylabel("Probabilités de survie, S(t)", fontsize = "small")
            st.pyplot(fig = plt)
        else :
            if isGrid :
                plotKmfGridLateEntry_survival_function(data, col_duration, col_event, montre_ci, crit, col_ent_tard)                
            else :
                plotKmfLateEntry_survival_function(data, col_duration, col_event, montre_ci, crit, col_ent_tard)
            
def plotKmfGrid_survival_function(data,col_duration,col_event,montre_ci,crit) :
    kmf = KaplanMeierFitter()
    critArray = data[crit].unique()
    nbRow = math.ceil(len(critArray)/3)
    fig ,axes = plt.subplots(nrows = nbRow, ncols = 3, figsize = (12, nbRow*4))
    axes = axes.ravel()
    i = 0
    for val, ax in zip(critArray,axes) :
        ix = data[crit] == val
        kmf.fit(data.loc[ix, col_duration], data.loc[ix,col_event], label = val)
        kmf.plot_survival_function(ax = ax, ci_show = montre_ci, legend = False)
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("Probabilités de survie")
        i+=1
    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig = plt)   

def plotKmf_survival_function(data,col_duration,col_event,montre_ci,crit) :
    kmf = KaplanMeierFitter()
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    for val in critArray :
        ix = data[crit] == val
        kmf.fit(data.loc[ix, col_duration], data.loc[ix, col_event], label = val)
        kmf.plot_survival_function(ax = ax, ci_show = montre_ci, legend = True)
    ax.set_title("Fonction de survie", fontsize = "x-small")
    ax.set_ylabel("Probabilités de survie", fontsize = "x-small")
    st.pyplot(fig = plt)

def plotKmfGridLateEntry_survival_function(data, col_duration, col_event, montre_ci, crit, col_ent_tard) :
    critArray = data[crit].unique()
    nbRow = math.ceil(len(critArray)/3)
    st.header("Survival function avec le modéle d'entrée tardive")
    fig ,axes = plt.subplots(nrows = nbRow, ncols = 3, figsize = (12, nbRow*4))
    axes = axes.ravel()
    i = 0
    #Initialise BrewslowFlemingHarringtonFitter
    bfh = BreslowFlemingHarringtonFitter()
    for val, ax in zip(critArray,axes) :
        ix = data[crit] == val
        try :
            bfh.fit(data.loc[ix, col_duration], data.loc[ix, col_event], entry = data.loc[ix, col_ent_tard], label = "Modéle d'entrée tardive")
            bfh.plot_survival_function(ax = ax, ci_show = montre_ci, marker = ".", legend = True)
            bfh.fit(data.loc[ix, col_duration], data.loc[ix, col_event], label = "Ignore les entrées tardives")
            bfh.plot_survival_function(ax = ax, ci_show = montre_ci, marker = "^", legend = True)
        except :
            pass            
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("Probabilités de survie")
        i+=1

    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig = plt)

def plotKmfLateEntry_survival_function(data, col_duration, col_event, montre_ci, crit, col_ent_tard) :
    bfh = BreslowFlemingHarringtonFitter()
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1, 1, figsize = (8,5))
    for val in critArray :
        ix = data[crit] == val
        try :
            bfh.fit(data.loc[ix, col_duration], data.loc[ix, col_event], entry = data.loc[ix, col_ent_tard], label = val)
            bfh.plot_survival_function(ax = ax, ci_show = montre_ci, legend = True)
        except :
            pass
    ax.set_title("Fonction de surviec en modéle d'entrée tardive", fontsize="x-small")
    ax.set_ylabel("Probabilités de survie", fontsize="x-small")
    st.pyplot(fig=plt)

def nelsonAalen(data, col_duration, col_event, montre_ci, crit, bandwidth, isGrid, col_ent_tard = "") :
    naf = NelsonAalenFitter()
    #Si one ne conside pas les entrées tardives
    if col_ent_tard == "" :      
        if crit =="" :
            naf.fit(data[col_duration], data[col_event])
            #Afficher la function risque cumulatif
            naf.plot_cumulative_hazard(title = "La function risque cumulatif", ci_show = montre_ci, figsize = (8,4))
            st.pyplot(fig = plt)
            #Afficher la function risque
            plt.close()
            naf.plot_hazard(bandwidth = bandwidth, title = "La function risque", ci_show = montre_ci, figsize = (8,4))
            st.pyplot(fig = plt)
        else :
            if isGrid :
                critArray = data[crit].unique()
                plotNelsonGrid_cumulative_hazard(data,col_duration, col_event, critArray, crit, montre_ci)
                st.divider()
                plotNelsonGrid_hazard(data,col_duration, col_event, critArray, crit, montre_ci, bandwidth)
            else :
                plotNelson_cumulative_hazard(data, col_duration, col_event, crit, montre_ci)
                st.divider()
                plotNelson_hazard(data, col_duration, col_event, crit, montre_ci, bandwidth)
           
    else : 
        if crit =="" :
            naf.fit(data[col_duration], data[col_event], entry=data[col_ent_tard], label = "Modéle d'entrée tardive")
            #Afficher la function risque cumulatif
            ax = plt.subplot(1,1,1)
            naf.plot_cumulative_hazard(ax = ax,title = "La function risque cumulatif", ci_show = montre_ci,figsize = (8,4))
            naf.fit(data[col_duration], data[col_event], label = "Ignore les entrées tardives")
            naf.plot_cumulative_hazard(ax = ax, ci_show = montre_ci)
            st.pyplot(fig = plt)
            #Afficher la function risque
            plt.close()
            ax = plt.subplot(1,1,1)
            naf.fit(data[col_duration], data[col_event], entry=data[col_ent_tard], label="Modéle d'entrée tardive")
            naf.plot_hazard(ax = ax, bandwidth = bandwidth, title = "La function risque", ci_show = montre_ci,figsize = (8,4))
            naf.fit(data[col_duration], data[col_event], label = "Ignore les entrées tardives")
            naf.plot_hazard(ax = ax,bandwidth = bandwidth, ci_show = montre_ci)
            st.pyplot(fig=plt)
        else :
            if isGrid :
                critArray = data[crit].unique()
                plotNelsonGridLateEntry_cumulative_hazard(data, col_duration, col_event, col_ent_tard, critArray, crit, montre_ci)
                st.divider()
                plotNelsonGridLateEntry_hazard(data, col_duration, col_event, col_ent_tard, critArray, crit, montre_ci, bandwidth)
            else :
                plotNelsonLateEntry_cumulative_hazard(data, col_duration, col_event, col_ent_tard, crit, montre_ci)
                st.divider()
                plotNelsonLateEntry_hazard(data, col_duration, col_event, col_ent_tard, crit,montre_ci, bandwidth)


def plotNelsonGrid_cumulative_hazard(data, col_duration, col_event, critArray, crit, montre_ci) :
    naf = NelsonAalenFitter()
    nbRow = math.ceil(len(critArray)/3)
    fig ,axes = plt.subplots(nrows = nbRow, ncols = 3, figsize = (12,nbRow*4))
    axes = axes.ravel()
    i = 0
    for val, ax in zip(critArray,axes) :
        ix = data[crit] == val
        naf.fit(data.loc[ix, col_duration], data.loc[ix,col_event], label = val)
        naf.plot_cumulative_hazard(ax = ax, ci_show = montre_ci, legend = False)
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("La function risque cumulatif")
        i+=1
    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig = plt)

def plotNelsonGrid_hazard(data, col_duration, col_event, critArray, crit, montre_ci, bandwidth) :
    naf = NelsonAalenFitter()
    nbRow = math.ceil(len(critArray)/3)
    fig ,axes = plt.subplots(nrows = nbRow, ncols = 3, figsize = (12,nbRow*4))
    axes = axes.ravel()
    i = 0
    for val, ax in zip(critArray,axes) :
        ix = data[crit] == val
        naf.fit(data.loc[ix, col_duration], data.loc[ix,col_event], label=val)
        naf.plot_hazard(ax=ax, bandwidth = bandwidth, ci_show = montre_ci, legend = False)
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("La function risque")
        i+=1
    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig=plt)

def plotNelson_cumulative_hazard(data, col_duration, col_event, crit, montre_ci) :
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    naf = NelsonAalenFitter()
    for val in critArray :
        ix = data[crit] == val
        naf.fit(data.loc[ix, col_duration], data.loc[ix, col_event], label = val)
        naf.plot_cumulative_hazard(ax = ax, ci_show = montre_ci, legend = True)
    ax.set_title("Nelson-Aalen", fontsize = "small")
    ax.set_ylabel("La function risque cumulatif", fontsize = "small")
    st.pyplot(fig=plt)

def plotNelson_hazard(data, col_duration, col_event, crit, montre_ci, bandwidth) :
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    naf = NelsonAalenFitter()
    for val in critArray :
        ix = data[crit] == val
        naf.fit(data.loc[ix, col_duration], data.loc[ix, col_event], label=val)
        naf.plot_hazard(ax = ax, bandwidth = bandwidth, ci_show = montre_ci, legend = True)
    ax.set_title("Nelson-Aalen", fontsize = "small")
    ax.set_ylabel("La function risque", fontsize = "small")
    st.pyplot(fig=plt)

def plotNelsonGridLateEntry_cumulative_hazard(data, col_duration, col_event, col_ent_tard, critArray, crit, montre_ci) :
    naf = NelsonAalenFitter()
    nbRow = math.ceil(len(critArray)/3)
    fig ,axes = plt.subplots(nrows=nbRow,ncols=3,figsize=(12,nbRow*4))
    axes = axes.ravel()
    i = 0
    for val, ax in zip(critArray,axes) :
        ix = data[crit] == val
        try :
            naf.fit(data.loc[ix, col_duration], data.loc[ix,col_event], entry=data.loc[ix,col_ent_tard], label="Modèle d'entrée tardive")
            naf.plot_cumulative_hazard(ax=ax, ci_show=montre_ci, legend =True)
        except Exception as e:
            print(e)
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("La function risque cumulatif")
        i+=1
    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig=plt)

def plotNelsonGridLateEntry_hazard(data,col_duration,col_event,col_ent_tard,critArray,crit,montre_ci,bandwidth) :
    naf = NelsonAalenFitter()
    nbRow = math.ceil(len(critArray)/3)
    fig ,axes = plt.subplots(nrows=nbRow,ncols=3,figsize=(12,nbRow*4))
    axes = axes.ravel()
    i = 0
    for val, ax in zip(critArray,axes) :
        ix = data[crit] == val
        try :
            naf.fit(data.loc[ix, col_duration], data.loc[ix,col_event], entry=data.loc[ix,col_ent_tard], label="Modèle d'entrée tardive")
            naf.plot_hazard(ax=ax, bandwidth=bandwidth, ci_show=montre_ci, legend =True)
        except Exception as e :
            print(e)
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("La function risque")
        i+=1
    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig=plt)

def plotNelsonLateEntry_cumulative_hazard(data, col_duration, col_event, col_ent_tard, crit, montre_ci) :
    naf = NelsonAalenFitter()
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    for val in critArray :
        ix = data[crit] == val
        try :
            naf.fit(data.loc[ix, col_duration], data.loc[ix, col_event], entry = data.loc[ix, col_ent_tard], label = val)
            naf.plot_cumulative_hazard(ax = ax, ci_show = montre_ci, legend = True)
        except :
            pass
    ax.set_title("Nelson-Aalen", fontsize = "small")
    ax.set_ylabel("La function risque cumulatif", fontsize = "small")
    st.pyplot(fig=plt)

def plotNelsonLateEntry_hazard(data, col_duration, col_event, col_ent_tard, crit,montre_ci, bandwidth) :
    naf = NelsonAalenFitter()
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    for val in critArray :
        ix = data[crit] == val
        try :
            naf.fit(data.loc[ix, col_duration], data.loc[ix, col_event], entry = data.loc[ix, col_ent_tard], label = val)
            naf.plot_hazard(ax = ax, bandwidth = bandwidth, ci_show = montre_ci, legend = True)
        except :
            pass
    ax.set_title("Nelson-Aalen", fontsize = "small")
    ax.set_ylabel("La function risque", fontsize = "small")
    st.pyplot(fig=plt)

def weibullPrintSummary(wbf) :
    #Redirect où stdout va, écrire à mystdout
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    wbf.print_summary()
    sys.stdout = old_stdout
    with st.expander("Récapitulatif") :
        st.text(mystdout.getvalue())

def weibull(data, col_duration, col_event, montre_ci, crit, isGrid, col_ent_tard="") :
    wbf = WeibullFitter()
    #Si one ne conside pas les entrées tardives
    if col_ent_tard == "" :      
        if crit =="" :
            wbf.fit(data[col_duration], data[col_event])
            #Afficher summary de paramètres du modèle Weibull
            weibullPrintSummary(wbf)
            #Afficher la function risque cumulatif
            wbf.plot_cumulative_hazard(title = "La function risque cumulatif", ci_show = montre_ci, figsize = (8,4))
            st.pyplot(fig=plt)
            #Afficher la function risque
            plt.close()
            wbf.plot_hazard(title = "La function risque", ci_show = montre_ci, figsize = (8,4))
            st.pyplot(fig = plt)
        else :
            if isGrid :
                critArray = data[crit].unique()
                plotWeibullGrid_cumulative_hazard(data, col_duration, col_event, critArray, crit, montre_ci)
                st.divider()
                plotWeibullGrid_hazard(data, col_duration, col_event, critArray, crit, montre_ci)
            else :
                plotWeibull_cumulative_hazard(data, col_duration, col_event, crit, montre_ci) 
                st.divider()
                plotWeibull_hazard(data, col_duration, col_event, crit, montre_ci)
    else :
        if crit =="" :
            wbf.fit(data[col_duration], data[col_event], entry = data[col_ent_tard], label = "Modéle d'entrée tardive")
            #Afficher summary de paramètres du modèle Weibull
            weibullPrintSummary(wbf)
            ax = plt.subplot(1,1,1)
            wbf.plot_cumulative_hazard(ax = ax, title = "La function risque cumulatif", ci_show = montre_ci, figsize = (8,4))
            wbf.fit(data[col_duration], data[col_event], label = "Ignore les entrées tardives")
            wbf.plot_cumulative_hazard(ax = ax, ci_show = montre_ci)
            st.pyplot(fig = plt)
            #Afficher la function risque
            plt.close()
            ax=plt.subplot(1, 1, 1)
            wbf.fit(data[col_duration], data[col_event], entry = data[col_ent_tard], label = "Modéle d'entrée tardive")
            wbf.plot_hazard(ax = ax, title = "La function risque", ci_show = montre_ci, figsize = (8,4))
            wbf.fit(data[col_duration], data[col_event], label = "Ignore les entrées tardives")
            wbf.plot_hazard(ax = ax, ci_show = montre_ci)
            st.pyplot(fig = plt)
        else :
            if isGrid :
                critArray = data[crit].unique()
                plotWeibullGridLateEntry_cumulative_hazard(data, col_duration, col_event, col_ent_tard, critArray, crit, montre_ci)
                st.divider()
                plotWeibullGridLateEntry_hazard(data, col_duration, col_event, col_ent_tard, critArray, crit, montre_ci)
            else :
                plotWeibullLateEntry_cumulative_hazard(data, col_duration, col_event, col_ent_tard, crit, montre_ci) 
                st.divider()
                plotWeibullLateEntry_hazard(data, col_duration, col_event, col_ent_tard, crit,montre_ci)

def plotWeibullGrid_cumulative_hazard(data, col_duration, col_event, critArray, crit, montre_ci) :
    wbf = WeibullFitter()
    nbRow = math.ceil(len(critArray) / 3)
    fig ,axes = plt.subplots(nrows = nbRow, ncols = 3, figsize = (12, nbRow*4))
    axes = axes.ravel()
    i = 0
    for val, ax in zip(critArray, axes) :
        ix = data[crit] == val
        wbf.fit(data.loc[ix, col_duration], data.loc[ix,col_event], label = val)
        wbf.plot_cumulative_hazard(ax = ax, ci_show = montre_ci, legend = False)
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("La function risque cumulatif")
        i+=1
    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig = plt)

def plotWeibullGrid_hazard(data, col_duration, col_event, critArray, crit, montre_ci) :
    wbf = WeibullFitter()
    nbRow = math.ceil(len(critArray) / 3)
    fig ,axes = plt.subplots(nrows = nbRow, ncols = 3, figsize = (12, nbRow*4))
    axes = axes.ravel()
    i = 0
    for val, ax in zip(critArray, axes) :
        ix = data[crit] == val
        wbf.fit(data.loc[ix, col_duration], data.loc[ix,col_event], label = val)
        wbf.plot_hazard(ax = ax, ci_show = montre_ci, legend = False)
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("La function risque")
        i+=1
    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig = plt)

def plotWeibull_cumulative_hazard(data, col_duration, col_event, crit, montre_ci) :
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    wbf = WeibullFitter()
    for val in critArray :
        ix = data[crit] == val
        wbf.fit(data.loc[ix, col_duration], data.loc[ix, col_event], label = val)
        wbf.plot_cumulative_hazard(ax = ax, ci_show = montre_ci, legend = True)
    ax.set_title("Weibull", fontsize = "small")
    ax.set_ylabel("La function risque cumulatif", fontsize = "small")
    st.pyplot(fig = plt)

def plotWeibull_hazard(data, col_duration, col_event, crit, montre_ci) :
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1, 1, figsize = (8, 5))
    wbf = WeibullFitter()
    for val in critArray :
        ix = data[crit] == val
        wbf.fit(data.loc[ix, col_duration], data.loc[ix, col_event], label = val)
        wbf.plot_hazard(ax = ax, ci_show = montre_ci, legend = True)
    ax.set_title("Weibull", fontsize = "small")
    ax.set_ylabel("La function risque", fontsize = "small")
    st.pyplot(fig = plt)

def plotWeibullGridLateEntry_cumulative_hazard(data, col_duration, col_event, col_ent_tard, critArray, crit, montre_ci) :
    wbf = WeibullFitter()
    nbRow = math.ceil(len(critArray)/3)
    fig ,axes = plt.subplots(nrows = nbRow, ncols = 3, figsize = (12, nbRow*4))
    axes = axes.ravel()
    i = 0
    for val, ax in zip(critArray,axes) :
        ix = data[crit] == val
        try :
            wbf.fit(data.loc[ix, col_duration], data.loc[ix,col_event], entry = data.loc[ix,col_ent_tard], label = "Modèle d'entrée tardive")
            wbf.plot_cumulative_hazard(ax = ax, ci_show = montre_ci, legend = True)
        except Exception as e:
            print(e)
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("La function risque cumulatif")
        i+=1
    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig = plt)

def plotWeibullGridLateEntry_hazard(data, col_duration, col_event, col_ent_tard, critArray, crit, montre_ci) :
    wbf = WeibullFitter()
    nbRow = math.ceil(len(critArray)/3)
    fig ,axes = plt.subplots(nrows=nbRow,ncols=3,figsize=(12,nbRow*4))
    axes = axes.ravel()
    i = 0
    for val, ax in zip(critArray,axes) :
        ix = data[crit] == val
        try :
            wbf.fit(data.loc[ix, col_duration], data.loc[ix,col_event], entry=data.loc[ix,col_ent_tard], label="Modèle d'entrée tardive")
            wbf.plot_hazard(ax=ax, ci_show=montre_ci, legend =True)
        except Exception as e :
            print(e)
        ax.set_title(val)
        plt.xlim(0, data[col_duration].max())
        if i == 0 :
            ax.set_ylabel("La function risque")
        i+=1
    for i, ax in enumerate(axes) :
        if i >= len(critArray) :
            fig.delaxes(ax)
    plt.tight_layout()
    st.pyplot(fig = plt)

def plotWeibullLateEntry_cumulative_hazard(data, col_duration, col_event, col_ent_tard, crit, montre_ci) :
    wbf = WeibullFitter()
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    for val in critArray :
        ix = data[crit] == val
        try :
            wbf.fit(data.loc[ix, col_duration], data.loc[ix, col_event], entry = data.loc[ix, col_ent_tard], label = val)
            wbf.plot_cumulative_hazard(ax = ax, ci_show = montre_ci, legend = True)
        except :
            pass
    ax.set_title("Weibull", fontsize = "small")
    ax.set_ylabel("La function risque cumulatif", fontsize = "small")
    st.pyplot(fig = plt)

def plotWeibullLateEntry_hazard(data, col_duration, col_event, col_ent_tard, crit,montre_ci) :
    wbf = WeibullFitter()
    critArray = data[crit].unique()
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    for val in critArray :
        ix = data[crit] == val
        try :
            wbf.fit(data.loc[ix, col_duration], data.loc[ix, col_event], entry = data.loc[ix, col_ent_tard], label = val)
            wbf.plot_hazard(ax = ax, ci_show = montre_ci, legend = True)
        except :
            pass
    ax.set_title("Weibull", fontsize = "small")
    ax.set_ylabel("La function risque", fontsize = "small")
    st.pyplot(fig = plt)

#Rubrique Probabilités de survie et courbres de survie
if selected == "Probabilités de survie et courbes de survie" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        df = data.iloc[:,1:]
        colNum = columsNumeric(df)
        colNoNum = columnsNoNum(df, colNum)
        st.sidebar.markdown(":orange[Fonction de survie : Kaplan-Meier]")
        st.sidebar.markdown(":orange[Fonction risque : Nelson-Aalen]")
        st.sidebar.markdown(":orange[Fonction risque : Weibull]")
        col_duration = st.sidebar.selectbox("Sélectionner une colonne pour une durée ", data.columns)
        col_event = st.sidebar.selectbox("Sélectionner une colonne pour un événement", data.columns)
        chkCrit = st.sidebar.checkbox("Estimer en critère.")
        crit =""
        if chkCrit :
            crit =  st.sidebar.selectbox("Sélectionner un critère de données :", colNoNum)
        ent_tard = st.sidebar.checkbox("Consider les entrées tardive")

        if ent_tard :
            col_ent_tard = st.sidebar.selectbox("Sélectionner une colonne pour les entrées tardives", data.columns)
        montre_ci = st.sidebar.checkbox("Montre l'intervalle de confiance dans le graphique de survival function", value=True) 
        bandwidth = st.sidebar.slider("Choisir un nombre de bandwidth pour la function risque par le modèle Nelson-Aalen",1,10,1)
        isGrid = st.sidebar.checkbox("Montre sur grille")
        st.warning("Veuillez sélectionner bien des colonnes de la durée et de l'événement et un critère si besoin",icon="⚠️" )
        if st.button("Afficher la fonction de survie") :
            if ent_tard :
                kaplanMeyer(data, col_duration, col_event, montre_ci, crit, isGrid, col_ent_tard)
            else :
                kaplanMeyer(data, col_duration, col_event, montre_ci, crit,isGrid)
        
        st.subheader("La function risque ou Hazard rates")
        if st.button("Afficher par le modéle Nelson-Aalen") :
            if ent_tard :
                nelsonAalen(data, col_duration, col_event, montre_ci, crit, bandwidth, isGrid, col_ent_tard)
            else :
                nelsonAalen(data, col_duration, col_event, montre_ci, crit, bandwidth, isGrid)
        if st.button("Afficher par le modèle Weibull") :
            if ent_tard :
                weibull(data, col_duration, col_event, montre_ci, crit, isGrid, col_ent_tard)
            else :
                weibull(data, col_duration, col_event, montre_ci, crit, isGrid)

style_predict_box = 'style = "border : 2px solid orange; max-width : 10%;  text-align: center; margin-bottom: 15px"'

def kaplanMeierPredict(data, col_duration, col_event, crit, time_predict, col_ent_tard="") :
    kmf = KaplanMeierFitter()
    #BFH pour calculer une fonction de survie avec l'entrée tard car lifelines.exceptions.StatError:
    # There are too few early truncation times and too many events. S(t)==0 for all t>10. Recommend BreslowFlemingHarringtonFitter.
    bfh = BreslowFlemingHarringtonFitter()
    if crit == "":
        newData = data
        if col_ent_tard == "" :
            kmf.fit(newData[col_duration], newData[col_event])
        else :
            try :
                bfh.fit(newData[col_duration], newData[col_event], entry = newData[col_ent_tard])
            except :
                pass
    else :
        ix = data[crit] == st.session_state.filter_value
        newData = data.loc[ix, :]
        if col_ent_tard == "" :
            kmf.fit(newData[col_duration], newData[col_event])
        else :
            try :
                bfh.fit(newData[col_duration], newData[col_event], entry = newData[col_ent_tard])
            except :
                pass

    st.write("Le nombre de lignes de données impliquées = ", newData[col_duration].count())
    if col_ent_tard == "" :
        predict = "{:.2f}".format(kmf.predict(time_predict, interpolate = True))
    else :
        predict = "{:.2f}".format(bfh.predict(time_predict, interpolate = True))
      
    st.markdown(f'<div {style_predict_box}>{predict}</div>', unsafe_allow_html=True)   

def weibullPredict(data, col_duration, col_event, crit, time_predict, col_ent_tard="") :
    wbf = WeibullFitter()
    if crit == "":
        newData = data
        if col_ent_tard == "" :
            wbf.fit(newData[col_duration], newData[col_event])   
        else :
            wbf.fit(newData[col_duration], newData[col_event], entry = newData[col_ent_tard])
    else :
        ix = data[crit] == st.session_state.filter_value
        newData = data.loc[ix, :]
        if col_ent_tard == "" :    
            wbf.fit(newData[col_duration], newData[col_event])
        else :
            wbf.fit(newData[col_duration], newData[col_event], entry = newData[col_ent_tard])
    st.write("Le nombre de lignes de données impliquées = ", newData[col_duration].count())
    predict = "{:.2f}".format(wbf.predict(time_predict, interpolate = True))
    st.markdown(f'<div {style_predict_box}>{predict}</div>', unsafe_allow_html=True)   
            
#Rubrique Prédiction de survie d'un individu
if selected == "Prédiction de survie d'un individu" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        st.sidebar.markdown(":orange[Prediction de la fonction de survie]")
        st.sidebar.markdown(":orange[: Kaplan-Meier et Weibull]")
        col_duration = st.sidebar.selectbox("Sélectionner une colonne pour une durée ", data.columns)
        col_event = st.sidebar.selectbox("Sélectionner une colonne pour un événement", data.columns)
        chkCrit = st.sidebar.checkbox("Estimer en critère.")
        crit =""
        colNum = columsNumeric(data.iloc[:,1:])
        colNoNum = columnsNoNum(data.iloc[:,1:], colNum)
        if chkCrit :
            crit =  st.sidebar.selectbox("Sélectionner un critère de données :", colNoNum)
            filterValue = st.sidebar.selectbox("Choisir une value pour le filtrage", sorted(list(data[crit].unique())), key = "filter_value")
        ent_tard = st.sidebar.checkbox("Consider les entrées tardive")
        col_ent_tard = ""
        if ent_tard :
            col_ent_tard = st.sidebar.selectbox("Sélectionner une colonne pour les entrées tardives", data.columns)
        maxTime = data[col_duration].max()
        #print(maxTime)
        #print(2*maxTime)
        st.warning("Veuillez sélectionner bien des colonnes de la durée et de l'événement et un critère si besoin", icon = "⚠️" )
        time_predict = st.number_input(label = "**:orange[Veuillez saisir un nombre de temps où vous voulez pour la prediction]**"
                    , min_value = 1, max_value = 2*maxTime, value = 1, step= 1)
        if st.button("Afficher une prediction de la fonction de survie: Kaplan-Meier") :
            kaplanMeierPredict(data, col_duration, col_event, crit, time_predict, col_ent_tard)
        if st.button("Afficher une prediction de la fonction de survie: modèle de Weibull ") :
            weibullPredict(data, col_duration, col_event, crit, time_predict, col_ent_tard) 

def coxPHFitterCustom(data, cols, col_duration, col_event, col_ent_tard) :
    cph = CoxPHFitter(penalizer=0.1)
    newData = data.loc[:, cols]
    newData = pd.get_dummies(newData, drop_first = True, dtype = int)
    if col_ent_tard == "" :
        cph.fit(newData, duration_col = col_duration, event_col = col_event)
    else :
        cph.fit(newData, duration_col = col_duration, event_col = col_event, entry_col = col_ent_tard)
    
    return cph, newData


# Rubrique Modèle de régression de Cox
if selected == "Modèle de régression de Cox" :
    data = st.session_state.dataFrame
    if len(data.columns) != 0 :
        cols = st.sidebar.multiselect("Sélectionner des colonnes à traiter dans le modèle  de régression de COX", data.columns)
        col_duration = st.sidebar.selectbox("Sélectionner une colonne pour une durée ", data.columns)
        col_event = st.sidebar.selectbox("Sélectionner une colonne pour un événement", data.columns)
        ent_tard = st.sidebar.checkbox("Consider les entrées tardive")
        col_ent_tard = ""
        if ent_tard :
            col_ent_tard = st.sidebar.selectbox("Sélectionner une colonne pour les entrées tardives", data.columns)
        
        st.warning("Veuillez sélectionner bien des colonnes à traiter",icon="⚠️" )
        if st.button("Afficher un récapitulatif") :
            try :
                cph, newData = coxPHFitterCustom(data, cols, col_duration, col_event, col_ent_tard)
                with st.expander("Dataframe tranformé par One-Hot Encoding") :
                    newData
                with st.expander("Description statistique") :
                    st.dataframe(newData.describe())
                cph.summary   
                cph.plot()
                st.pyplot(fig = plt)
            except :
                st.error("Veuillez sélectionner bien des colonnes à traiter", icon="🚨")

        st.subheader(":orange[Tracer l'effet de la variation d'une covariable]")
        if len(cols) != 0 :
            newData = data.loc[:, cols]
            newData = pd.get_dummies(newData, drop_first = True, dtype = int)
            covariable = st.sidebar.selectbox("Sélectionner une colonne covariable", newData.columns) 
            #Si une colonne est en type numérique, on cherche min at max dans cette colonne et prend 5 values entre min et max (inclusif)
            #de même écart entre eux
            minVal = newData[covariable].min()
            maxVal = newData[covariable].max()
            if minVal != 0 and maxVal != 1 :
                sample = np.linspace(start = minVal, stop = maxVal, num = 5) 
            else :
            #Si c'est une colonne de category ou string, on prend 0 et 1 comme une variable binaire
                sample = [0,1]

        st.warning("Veuillez sélectionner bien une colonne covariable",icon="⚠️" )
        if st.button("Afficher la fonction de survie") :
            try :
                cph, newData = coxPHFitterCustom(data, cols, col_duration, col_event, col_ent_tard)
            #    print(covariable)
            #    print(sample)
                cph.plot_partial_effects_on_outcome(covariates = covariable, values = sample, cmap='coolwarm', figsize=(8, 5))
                st.pyplot(fig = plt)
                #Hypothèse de test selon laquelle il n’y a pas de différence de survie entre groupes 
                #tout en contrôlant une autre variable
                results = multivariate_logrank_test(newData[col_duration], newData[covariable], newData[col_event])
                results.summary
            except :
                st.error("Veuillez sélectionner bien des colonnes à traiter et une colonne covariable", icon="🚨")

            