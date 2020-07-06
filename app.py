import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import requests
from io import StringIO
plt.rcParams["figure.figsize"] = [16, 5]
import numpy as np
import plotly.express as px
import io
from sklearn.cluster import KMeans



VARIABLES = ["race", "gender", "age group"]

PARAMS = {
    "race": [
        "BLACK",
        "ASIAN / PACIFIC ISLANDER",
        "WHITE",
        "WHITE HISPANIC",
        "BLACK HISPANIC",
        "UNKNOWN",
        "AMERICAN INDIAN/ALASKAN NATIVE",
        "OTHER",
    ],
    "gender": ["M", "F"],
    "age group": ["<18", "18-24", "25-44", "45-64", "65+",],
}

DATA_URL = "https://github.com/linynjosh/nypd-crime-vis/blob/master/NYPD_Arrests_Data__Historic.csv?raw=true"

st.title("New York City🗽Police Department Crime Visualization")

@st.cache(persist=True)
def load_data(data, year):
    data = data[(data["ARREST_DATE"].str.contains(str(year)))]
    lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    data.rename(
        columns={
            "ARREST_DATE": "arrest date",
            "OFNS_DESC": "offense description",
            "AGE_GROUP": "age group",
            "PERP_SEX": "gender",
            "PERP_RACE": "race",
            "LAW_CODE": "law code",
            "Longitude": "longitude",
            "Latitude": "latitude",
        },
        inplace=True,
    )
    return data


def kmeans(data, cluster_slider):
    kmeans = KMeans(n_clusters=cluster_slider).fit(data)
    centroids = kmeans.cluster_centers_
    df = pd.DataFrame(centroids, columns=["latitude", "longitude"])
    return df


def get_hist(data, param):
    hist = {
        group: data[data[param].str.contains(group, na=False)].shape[0]
        for group in PARAMS[param]
    }
    hist_reshape = {param: list(hist.keys()), "count": list(hist.values())}
    df = pd.DataFrame.from_dict(hist_reshape)
    df = df.set_index(param)
    return df


def offense_race(data, race):
    df = data[data["race"].str.contains(race)]
    k = list(df["offense description"].value_counts().index[0:20])
    v = list(df["offense description"].value_counts().values[0:20])
    reshape = {"offense": k, "count": v}
    df_new = pd.DataFrame.from_dict(reshape)
    df_new = df_new.set_index("offense")
    return df_new


st.sidebar.subheader("About")
st.sidebar.markdown(
    "This interactive program investigates a variety of crime aspects within New York City. "
    "While this report shows the most frequent locations of crimes in NYC it further shows the "
    "leading numbers of arrest in terms of race, age, offense, and gender. "
)

# -------------- App Layout
st.header("Objective")
st.markdown("To analyze the relationship between demographic features of suspects and the number of arrests over the years.")
st.header("Methods")
st.subheader("Pt I Procedure")
st.markdown("Firstly, for our primary visual analysis, the data will be categorized by the suspects' demographic features such as age group, race, and gender. "
            "Following that, we can visualize the outcome through graphing three pie charts that show the percentage weigh of each categorized demographics. "
            "The frequency of arrests differs for various regions in NYC, so a map will be graphed showing a random sample of arrests from 2006-2019. "
            "Lastly, we will graph a bar chart showing the top 20 offense descriptions from 2006-2019 in the arrest dataset. ")
st.subheader("Pt II Procedure")
st.markdown("Firstly, the Kmeans clustering algorithm will be applied to cluster the data into the top locations of arrest on a NYC map. "
            "In this secondary visual analysis, the year of the arrests becomes an independant variable along with the demographic features such as age group, race, and gender. "
            "On the left sidebar, there will be a year slidebar to choose which year to view for the graphs in the secondary analysis. Thus, the data will be categorized in a similar "
            "manner as Pt I, with the only difference in the addition of a year parameter and the Kmeans clustering map. ")
st.header("Overview of Dataset")
st.markdown("This NYPD dataset consists of a breakdown on every arrest in NYC from 2006 to 2019. "
            "Each record equates to an arrest in NYC by the NYPD and includes "
            "data on the location, suspect demographics, time of enforcement, type of crime, and etc. "
            "Please refer to the attached link for additional question about this dataset. ")


summary = {
    "Total number of arrests from 2006-2019": 5005855,
    "Average number of arrests per year from 2006-2019": 357561,
    "Total number of unique offenses from 2006-2019": 87,
}
x = list(summary.keys())
y = list(summary.values())
df_summary = {"Data Characteristics": x, "Count": y}
df_summary = pd.DataFrame.from_dict(df_summary)
df_summary = df_summary.set_index("Data Characteristics")
st.table(df_summary)

year_slider = st.sidebar.slider(
    min_value=2006,
    max_value=2019,
    value=2016,
    label="Year to view for secondary visual analysis: ",
)
# -------------- Load data
df = pd.read_csv(
    DATA_URL,
    usecols=[
        "Latitude",
        "Longitude",
        "ARREST_DATE",
        "OFNS_DESC",
        "AGE_GROUP",
        "PERP_SEX",
        "PERP_RACE",
    ],
)
df.dropna(
    subset=[
        "Latitude",
        "ARREST_DATE",
        "OFNS_DESC",
        "AGE_GROUP",
        "PERP_SEX",
        "PERP_RACE",
    ],
    inplace=True,
)
df = df[(df["Latitude"] < 42)]
df = df.sample(frac=0.1)
data = load_data(df, year=year_slider)

if st.checkbox("Show Sample Raw Data", False):
    st.subheader("Raw Data")
    st.write(data.sample(frac=0.01))

st.header("Primary Visual Analysis")
st.subheader("How does the number of arrests depend on race, gender, or age group from 2006-2019?")

""" First level dropdown"""
first_level_variable = st.selectbox("Choose Parameter:", VARIABLES, 0)
if first_level_variable == 'gender':
    st.markdown(
        """
        Total arrests based on gender from 2006-2019: 
        """
    )
    st.plotly_chart(
        go.Figure(data=[go.Pie(labels=["Male", "Female"], values=[4165257, 840598])])
    )
elif first_level_variable == 'race':
    st.markdown(
        """
        Total arrests based on race from 2006-2019: 
        """
    )
    st.plotly_chart(
        go.Figure(data=[go.Pie(labels=["Black", "Asian/Pacific Islander", "White", "White Hispanic", "Black Hispanic", "Unknown", "American Indian/Alaskan Native", "Other"],
                               values=[2430760, 204862, 603822, 1301339, 4012441, 50239, 11029, 1363])])
    )
elif first_level_variable == 'age group':
    st.markdown(
        """
        Total arrests based on age groups from 2006-2019
        """
    )
    st.plotly_chart(
        go.Figure(data=[go.Pie(labels=["<18", "18-24", "25-44", "45-64", "65+"],
                               values=[411809, 1315665, 2313082, 923100, 42004])])
    )


st.subheader("How does the number of arrests depend on location from 2006-2019?")
st.markdown(
    """
    The following map shows the locations of arrest in New York City during 2006-2019. 
    Due to the concerns on runtime, a random sampling of 5% has been applied to the dataset.
    """
)
df_sample = data.sample(frac=0.05)
st.map(df_sample[["latitude", "longitude"]])

st.subheader(
    "What are the most common reasons for arrests for all races from 2006-2019?"
)
st.markdown(
    "The following bar chart shows the top 20 offenses for all races across the years 2006-2019. "
)
# All races graph:
k = list(data["offense description"].value_counts().index[0:20])
v = list(data["offense description"].value_counts().values[0:20])
df = {"offense": k, "count": v}
df = pd.DataFrame.from_dict(df)
df = df.set_index("offense")
st.bar_chart(df)

st.header("Secondary Visual Analysis")
st.markdown(
    "In this secondary visual analysis, the year in which the crime occurrs becomes the independant variable along the other variables, race, gender, age group, and location. The number of arrests will remain the dependant varibale. "
)
st.subheader(
    "What are the most common locations of arrest in New York City for each year?"
)
st.markdown(
    "The following map shows the most frequent locations of crimes in New York City from 2006-2019. The most common locations of arrests on the New York City map are calculated with the Kmeans algorithm, where it distributes the NYPD crime dataset into Kpre-defined unique non-overlapping clusters. Due to concerns about the lengthy runtime caused by the size of the dataset, a 10% random sampling has been applied to the 5005855 data points. "
)
cluster_slider = st.slider(
    min_value=1, max_value=50, value=10, label="Number of clusters: "
)
centroids = kmeans(data[["latitude", "longitude"]], cluster_slider=cluster_slider)
st.map(centroids)


st.subheader("How do the number of arrests depend on XXX and year?")
st.markdown(
    "The following bar chart shows the relationship between the number of arrests in New York City and correlating factors such as age, sex, race, and year from 2006-2019.  "
)
variable = st.selectbox("Choose Parameter:", VARIABLES, 2)
st.bar_chart(get_hist(data, variable))

st.subheader("How do the offense description of arrests depend on race and year?")
st.markdown(
    "The following bar chart shows the dependence between the top 20 criminal offense committed and the offender's race from 2006-2019. The year slidebar can be toggled to see how the offenses for each race varies depending on the year in which the crime occurred. "
)
# Dropdown to choose race
race_param = st.selectbox("Choose race:", PARAMS["race"], 0)
st.bar_chart(offense_race(data, race_param))
