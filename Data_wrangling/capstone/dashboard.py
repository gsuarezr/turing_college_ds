import streamlit as st
import pandas as pd
import numpy as np
import glob
import plotly.express as px
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils import *
from plotly.subplots import make_subplots
import os

pd.options.plotting.backend = "plotly"
st.sidebar.image(os.path.dirname(os.path.abspath(__file__)) + "/logo.svg")
data_dict = wrangle("data")


st.title("Covid cases in South Korea")

st.sidebar.title("Covid Cases in Korea")
panel = st.sidebar.selectbox(
    "EDA Section, There is one for each dataset please select a panel",
    (
        "Time vs Age",
        "Time vs Region",
        "Region Metadata",
        "Time Gender",
        "Test",
        "Path",
        "Weather",
        "Patient Info",
        "Search Trend",
        "Policies",
        "Seoul",
    ),
)

st.sidebar.title("Conclusions and Final Analysis")

panel2 = st.sidebar.radio(
    "Select",
    ("Reset", "Forecasting", "PCA", "Conclusions"),
)
st.sidebar.markdown(
    "###### made by [Gerardo Suarez](https://mcditoos.github.io), You can find me on Github as [@mcditoos](https://github.com/mcditoos)"
)
if panel2 == "Reset":
    panel2 = 0

if panel2 == "Conclusions":
    st.subheader(
        "We may summarize what we have learned so far from the data, for future virus and new waves in other countries:"
    )
    st.markdown(
        "- Gender bares no strong correlation to infections, the assymetry in the infections can be explained through the mean floating population in korea"
    )
    st.markdown(
        "- Age plays a major role on mortality, the death rate gets higher the older the people, so special measurements need to be taken for the elderly"
    )
    st.markdown(
        "- Social and religious events played a major role in S. Korea's infections, it is important to keep this in mind for next ocassions so regulations don't come to late"
    )
    st.markdown(
        "- The search trends and infections time series have a great correlation, in the future is a good idea to keep this data as well as other tech related data \
        (most sport bands measure $Sp0_2$) to make adecuate policies before it's too late"
    )
    st.markdown(
        "- Most policies where established after the pandemic peak, so they were a mitigation strategy and needed to be more radical, in the future data analysis can lead\
            to making better policies in time"
    )
if panel2 == "PCA":
    st.subheader(
        "For the bonus challenge we used PCA for dimensionality reduction, we used it on a combination of the search trend and tests datasets"
    )
    col1, col2 = st.columns(2)
    data = pca_data(data=True)
    with col1:
        st.dataframe(data.head(8))
    with col2:
        st.dataframe(data.describe())
    st.caption(
        "After PCA we reduce all of the columns of the data to 2 (except for confirmed which is the target), we plot those 2 new variables as a scatter plot"
    )
    st.plotly_chart(
        pca_data(),
    )
if panel2 == "Forecasting":
    st.title("Preliminaries")
    st.caption(
        'One of the bonus challenges was to use a model to do some forecasting, when doing a model an important part of it is "Feature selection" or basicly,\
        indicating the variables of the model, after the previous analysis, we decided that the best bet was to build a model based on the search trends from the naver engine'
        + "we decided this based on the fact that this data seems to be the one that conveys the most information about new cases and is also the most complete one, to see the correlation \
            between the search trends and the confirmed cases let us look at the next plot"
    )
    st.plotly_chart(
        get_trend_con(),
    )
    st.caption(
        "We can see that peaks in the search trend happen about a month before than in the confirmed cases, indicating that it may be useful data to predict new\
        Covid outburst. Both time series seem to exhibit similar trends"
    )
    st.subheader("Linear regression")

    st.caption(
        "At least in physics, when dealing with data one tries to make a model as simple as possible, but not simpler than it is. Meaning that one should look for the\
        simplest model that reproduces the data. The simplest model there is, is a linear model, so that is the first model we are going to try on our data, we are going"
        + "to introduce a lag in our model, meaning that the data from n days before will be used to predict today's value, the green line indicates the end of our training data"
    )
    lag = st.slider("How many lag days?", 0, 21, 7)
    st.plotly_chart(
        get_trend_con(True, lag),
    )

    st.subheader("ARIMA Model")
    st.caption(
        "ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average, in this models we use walk-foward validation, meaning we feed the testing\
        data into the model as we predict, making our training set larger, this process is iterative and we need to keep feeding the data to our algorithm for it"
        + "to keep forecasting adequatly (in this case daily)"
    )
    st.plotly_chart(
        arima(),
    )
    st.caption(
        "If we stop the walk forward validation and we instead fit our predictions, the forecast become completely off after a while"
    )
    st.plotly_chart(
        arima(True),
    )
if not panel2:
    if panel == "Time vs Age":
        st.write(f"{data_dict}")
        # To make this panel better get information of the population in south korea
    #       st.subheader("Time age Histogram by date")
    #      start_time = st.slider(
    #         "When do you start?",
    #        value=datetime.datetime(2020, 3, 20),
    #       min_value=datetime.datetime(2020, 3, 2),
    #      max_value=datetime.datetime(2020, 6, 30),
    #     format="YY/MM/DD",
    # )
    # z = str(start_time).split("-")
    # z[-1] = z[-1][:2]
    # start = "-".join(z)
    # st.write("Start time:", start)
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.dataframe(data_dict["timeage"].head(8))
    # with col2:
    #     st.dataframe(data_dict["timeage"].describe())
    # st.subheader("Confirmed and Deceased people by date")
    # st.plotly_chart(
    #     get_hist_by_date(start),
    # )
    # st.caption(
    #    "In this dataset, there is data for each of the age groups for 121 days from 2020-03-02 to 2020-06-30, there are 9 age groups \
    #    [0s,10s,20s,30s,40s,50s,60s,70s,80s,90s], through all of the days, most of the confirmed are always in the 20s age group, this "
    #    + "may be due to the fact that this people are the most socially active (universities, dating, hanging out) they are also the group which tends to be\
    #    the most irresponsible, also a psychological factor is that covid is not really mortal for this age group"
    # )
    # st.subheader("Cummulative information (Trends)")
    # st.plotly_chart(
    #   get_cumulative(),
    #  use_container_width=True,
    # )
    #   st.caption(
    #      "As we mentioned before, and can be seen here the mortality rate for the 20s age group is null, and the older you get the more deadly the \
    #         disease is, more than covid this may be due to pre-existing conditions. As per the end date of the data the curves seems to be flattening out"
    #    + "so we may think that at this point everything is controlled, however we know now new waves ocurred"
    # )
    if panel == "Time vs Region":
        st.subheader("Confirmed Cases by region")

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data_dict["timeprovince"].head(8))
        with col2:
            st.dataframe(data_dict["timeprovince"].describe())

        st.caption(
            f'This dataset contains data from {len(data_dict["timeprovince"]["date"].unique())} days from {len(data_dict["timeprovince"].province.unique())} provinces'
        )
        pie_option = st.radio(
            "Which plot do you want to see?",
            ("%Released", "%Dead", "%Confirmed"),
            horizontal=True,
        )

        pie_jey = {"%Released": 0, "%Dead": 1, "%Confirmed": 2}
        st.subheader("Distribution at the final time")

        st.plotly_chart(
            get_pie(pie_jey[pie_option]),
            use_container_width=True,
        )

        st.caption(
            f"From here we can see that on June 2020, Daegu and Gyeongsangbuk-do have around 70% of the cases, when compared to the capital Seoul,\
             the percentages seem like too big in comparison, so something may have happened here that did not happen in Seoul, from the rest of the EDA"
            + "and some post analysis investigation we conclude that it seems to be due to a religious event in Daegu which caused a massive spread"
        )
        st.subheader("Cummulative information in time")

        st.plotly_chart(
            get_cum_province(pie_jey[pie_option]),
        )
        st.caption(
            "We can see from this previous plot that the slope for the curve of Daegu is way steeper than in the rest of the provinces, ant it starts in \
            february 2020, this was atributed to the [Shincheonji church meeting](https://www.nytimes.com/2020/02/21/world/asia/south-korea-coronavirus-shincheonji.html), and the data from other datasets confirms it "
            + "The cases skyrocketed after february 2020, this may be due to massive spread events like that one from the church, or due to increase testing\
                a bit more about will be said in next sections"
        )
    if panel == "Region Metadata":
        st.subheader("Region Metadata")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data_dict["region"].head(8))
        with col2:
            st.dataframe(data_dict["region"].describe())
        st.subheader("Elderly population, maybe plot a bit more in this section")
        option = st.radio(
            "How do you want to sort the order?",
            ("Percentage", "Population"),
            horizontal=True,
        )
        st.plotly_chart(
            elderly_pop(option),
            use_container_width=True,
        )
        st.caption(
            "The number of elderly people does not seem to bear any correlation to confirmed infections, this seems to indicate that transmission\
             is the same regardless of age, the bars are ordered by confirmed cases"
        )
    if panel == "Time Gender":
        st.subheader("Data about gender in time")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data_dict["timegender"].head(8))
        with col2:
            st.dataframe(data_dict["timegender"].describe())
        st.caption(
            f'This dataset contains data from {len(data_dict["timegender"]["date"].unique())} days '
        )
        option = st.radio(
            "Which plots do you want to see?",
            ("Confirmed", "Deceased"),
            horizontal=True,
        )
        st.subheader("Final date, infected by gender")
        st.plotly_chart(
            get_sex(option),
            use_container_width=True,
        )
        st.subheader("Cumulative infected by date, by gender")
        st.plotly_chart(
            get_cum_sex(option),
            use_container_width=True,
        )
        st.caption(
            "We can see that there were always more female than male patients, one may think this is due to population inbalance but \
             Korea's population is about 49.9% female and 50.1% male according to [worldbank](https://data.worldbank.org/indicator/SP.POP.TOTL.FE.ZS?locations=KR)"
            + ". Surprisingly, there are more males deceased than females, of course this may be due to other things, like males usually being smokers while females \
                are not as often due to cultural customs"
        )
    if panel == "Test":
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data_dict["time"].head(8))
        with col2:
            st.dataframe(data_dict["time"].describe())

        st.caption(
            f'This dataset contains data from {len(data_dict["time"]["date"].unique())} days about Covid-19 tests and their results'
        )
        st.subheader("Tests and results by date")
        option = st.radio(
            "Which plot do you want to see?",
            ("Daily", "Cummulative"),
            horizontal=True,
        )

        st.plotly_chart(
            get_cum_test(option),
            use_container_width=True,
        )
        st.caption(
            f"We can see that the number of test spiked around 2020/02/18, right the same date the massive spread seems to have started, this points to the fact\
                that the infections may have been there, however not enough testing was done do the necessary measures were not applied untill it was two late "
            + ", No testing was advised by that time if asymptomatic and the virus can live in your body 14 days before showing any \
                [signs](https://eu.usatoday.com/in-depth/news/2020/03/13/what-coronavirus-does-body-covid-19-infection-process-symptoms/5009057002/)"
        )

    if panel == "Path":
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data_dict["case"].head(8))
        with col2:
            st.dataframe(data_dict["case"].describe())

        st.caption(
            f'This dataset contains data from {len(data_dict["case"])}  Covid-19  cases, about the way they got infected and whether it\
                was a group infection or not'
        )
        st.subheader("Path of Infection")
        option = st.radio(
            "Which plot do you want to see?",
            ("Cases", "City"),
            horizontal=True,
        )
        n = st.slider("How many individually?", 1, 10, 4)
        st.plotly_chart(
            path(n, option),
            use_container_width=True,
        )

        st.caption(
            f"This piece of data confirms that a huge part of the covid massive spread was due to the Shincheonji church as it was mentioned\
                while looking at other data, the other two big components are people coming from abroad and contact with patients. This seems to"
            + "indicate that forbiding events that involve many people is the best measure."
        )

        st.subheader("Group vs individual infections")
        st.plotly_chart(
            group(),
            use_container_width=True,
        )
        st.caption(
            f"We see that most cases where infected individually meaning by contact with someone else, however still group infections are a big \
            percentage, and involve way more people, making them a big worry, however since there are many individual cases as well, it seems to point"
            + " to more strict isolation norms being helpful"
        )

    if panel == "Weather":
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data_dict["weather"].head(8))
        with col2:
            st.dataframe(data_dict["weather"].describe())
        st.caption(
            f'This dataset contains weather data  from {list(data_dict["weather"].date)[0]} to {list(data_dict["weather"].date)[-1]}\
            specifically we are going to take a look at temperature, humidity, and wind speed, it contains data from {len(data_dict["weather"].province.unique())} provinces'
        )
        option = st.radio(
            "Which plot do you want to see?",
            (
                "Average Temperature",
                "Min Temperature",
                "Max Temperature",
                "Relative Humidity",
                "Wind Speed",
                "Precipitation",
            ),
            horizontal=True,
        )

        if "Temperature" in option:
            st.subheader(f"Average Temperature (Since January 2016)")
            st.plotly_chart(
                temp(),
                use_container_width=True,
            )
        else:
            st.subheader(f"Average {option} (Since January 2016)")
            st.plotly_chart(
                weather(option),
                use_container_width=True,
            )
        if "Temperature" in option:
            st.subheader(f"Temperature by Date")
        else:
            st.subheader(f"{option} by Date")
        start_time = st.slider(
            "What date do you want to see?",
            value=datetime.datetime(2020, 3, 2),
            format="YY/MM/DD",
        )
        z = str(start_time).split("-")
        z[-1] = z[-1][:2]
        start = "-".join(z)
        st.write("Start time:", start)

        if "Temperature" in option:
            st.plotly_chart(
                temp(date=start),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                weather(option=option, date=start),
                use_container_width=True,
            )
        # May add direction as a vector on the map

        st.subheader("Time series")

        st.plotly_chart(
            get_cum_weather(option=option),
            use_container_width=True,
        )
        st.caption(
            f"There doesn't seem to be any strong correlation between the weather data and confirmed cases, perhaps a correlation between higher wind speeds\
                and the infections, since both peaks are really close, however there may have been way more important factors besides weather "
        )
    if panel == "Patient Info":
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data_dict["patientinfo"].head(8))
        with col2:
            st.dataframe(data_dict["patientinfo"].describe())
        st.caption(
            f'This dataset contains data  from {len(data_dict["patientinfo"])} covid cases across Korea \
             across {len(data_dict["patientinfo"].province.unique())} provinces and {len(data_dict["patientinfo"].city.unique())} cities'
            + f' unfortunately only {len(data_dict["patientinfo"])-data_dict["patientinfo"].contact_number.isna().sum()} cases have info\
                about the contact number'
        )
        st.subheader("Contact number distribution")
        st.caption(
            f"The contact number data does not seem to be too reliable, for once the maximum (without filtering outliers) is 6100000099, which is \
                obviously a mistake since this is just a bit short from earth's population, we decided to use the IQR to deal with outliers, so "
            + "we disregarded contact numbers higher than 32 for this dataset, another thing to notice is that the dataset had lots of of zeros as contact\
                number which is also ilogical, since the person had to be in contact with at least one patient to be sick, however we took a look at shifting"
            + "this and nothing intereting was found "
        )
        cut = st.slider(
            "Where do you cut ouliers?", min_value=0, max_value=500, step=10, value=32
        )
        option = st.radio("Shift the minimum", ("False", "True"), horizontal=True)
        col1, col2 = st.columns(2)
        with col1:
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.dataframe(contact(cut=cut, describe=True))
        with col2:
            st.plotly_chart(
                contact(cut=cut, min=option),
            )
        # Show the data set of the people with most cases and do a little visualization, also show percentages by state (deceseased,released,isolated)
        st.subheader("Released and confirmed dates")
        st.plotly_chart(
            contact(cut=cut, confirmed=True, min=option), use_container_width=True
        )
        st.caption(
            f"We can see that there are multiple spikes in the mean number of confirmed cases, but the spikes are smaller than in te released cases\
                this points out to some people quarantining less than others, the distance between the peaks however indicates they quarantined long"
            + " enough, all of the outliers occurred in near the peaks in april and june, they may have happened because many cases where incoming and \
                the data entry got slopy"
        )
    if panel == "Search Trend":
        st.caption(
            "This file contains data about the search volume (relative interest) of diferent diseases (Cold, Flu, Pneumonia and Coronavirus)\
            from the naver search engine"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data_dict["searchtrend"].head(8))
        with col2:
            st.dataframe(data_dict["searchtrend"].describe())

        start_time = st.slider(
            "Select your time window",
            value=(datetime.datetime(2019, 11, 1), datetime.datetime(2020, 6, 1)),
            format="YY-MM-DD",
        )

        date1, date2 = (
            str(start_time[0]).split(" ")[0],
            str(start_time[1]).split(" ")[0],
        )
        st.plotly_chart(searchtrend(date1, date2))

        st.caption(
            "We see that the pattern somehow resambles the confirmed cases, this makes sense because in this era once people hear about something\
            they tend to look it up, so as symptoms began to be felt, the search must have gone up, we will take a look at this relation later"
        )
    if panel == "Policies":
        st.header("Policies")
        st.caption(
            f"This Dataset contains information about the dfferent types of policies and their start and end dates, overall they where {len(data_dict['policy'])}\
            policies of one of the following types {', '.join(data_dict['policy']['type'].unique())}, of which {data_dict['policy']['end_date'].isna().sum()}"
            + " are still active"
        )

        st.dataframe(data_dict["policy"].head(8))
        st.caption(
            "From the next plot we can see that most policies concern education and immigration, which makes sense because those are both areas where\
                lots of different people with different backgrounds get together in crowded spaces thus making the propagation of the virus easy"
            + "an interesting though controversial point is that there were no regulations about religious gatherings, even when they seem to be one of the main\
            causes of infection in S. Korea, as seen in the Path data."
        )
        st.plotly_chart(
            policies(),
        )
        st.caption(
            "The next two plots show the policies start dates in time, we can see that most of the policies started after the peak of the pandemic\
             (2020/02/29) indicating that they may have been a little to late."
        )
        st.plotly_chart(
            policies_dates(),
        )
        st.caption(
            "We stress the previous point looking at the policies per month, this leads to conclude that policies were about mitigating the spread\
            not about preventing it"
        )
        st.plotly_chart(
            policies_month(),
        )
        option = st.radio(
            "Select policy type",
            (
                "Alert",
                "Immigration",
                "Health",
                "Social",
                "Education",
                "Technology",
                "Administrative",
                "Transformation",
            ),
            horizontal=True,
        )
        data = data_dict["policy"]
        mask = data["type"] == option
        dates = list(data[mask].start_date)
        fig = get_trend_con()
        for i in dates:
            fig.add_vline(
                x=i,
                line_width=3,
                line_dash="dash",
                line_color="green",
            )
        st.plotly_chart(
            fig,
        )
        st.caption(
            f"The green lines are located on the dates {list(data[mask].start_date)}, which is when each policy of the {option} type began, "
        )

    if panel == "Seoul":
        st.subheader("Seoul Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data_dict["seoulfloating"].head(8))
        with col2:
            st.dataframe(data_dict["seoulfloating"].describe())
        st.caption(
            f'This dataset contains data from the floating population in seoul from {list(data_dict["seoulfloating"].date)[0]} to {list(data_dict["seoulfloating"].date)[-1]}\
            it contains information about {len(data_dict["seoulfloating"].city.unique())} cities, mostly about  the floating population number, sex, and age.'
        )
        option = st.radio(
            "Which plot do you want to see?",
            (
                "Age",
                "Gender",
                "Population",
            ),
            horizontal=True,
        )
        st.plotly_chart(
            seoulpop(option),
        )
        st.caption(
            f"From the previous plots we can get some insights about the floating population in Seoul, in average there are less males on the streets, which is probably why \
                there is such an asymmetry in the number of confirmed cases, there doesn't seem to be a strong correlation with age, indicating that social meetings "
            + "or some other aspects are more important when looking at particular demographics"
        )
