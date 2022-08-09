import string
import streamlit as st
import pandas as pd
import numpy as np
import glob
import plotly.express as px
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import urllib
import requests
import os

colors = px.colors.qualitative.Plotly
pd.options.plotting.backend = "plotly"

DataDict = dict[pd.DataFrame]
Figure = go.Figure


def wrangle(folder: str) -> DataDict:
    """
    This function takes a string as a directory and loads all the csv files present into DataFrames,
    it create a key from  the names of the files and generates a dictionary with all of the DataFrames
    Example:
    >>> data_dict=wrangle('data')

    Stores a dictionary of Dataframes for each of the csv files in the data directory
    """
    data_list = glob.glob(f"Data_wrangling/capstone/{folder}/*.csv")
    data_list = sorted(data_list)
    names = [
        "case",
        "patientinfo",
        "policy",
        "region",
        "searchtrend",
        "seoulfloating",
        "time",
        "timeage",
        "timegender",
        "timeprovince",
        "weather",
    ]
    data_dict = {}
    for i in range(len(data_list)):
        data_dict[names[i]] = pd.read_csv(data_list[i])
    return data_dict


def get_nominatim_geocode(address: str) -> tuple[str, str]:
    """
    This function takes a string (intended to be a location) and returns the longitue and latitude as a tuple
    Example:
    >>> get_nominatim_geocode('Gyeongsangnam-do')
        ('128.6925', '35.2382')
    """
    url = (
        "https://nominatim.openstreetmap.org/search/"
        + urllib.parse.quote(address)
        + "?format=json"
    )
    try:
        response = requests.get(url).json()
        return response[0]["lon"], response[0]["lat"]
    except Exception as e:
        # print(e)
        return None, None


def get_hist_by_date(date: str) -> Figure:
    """
    This function takes a date as a string in the format YY/MM/DD and returns a histogram
    of the confirmed and diseased cases accumulated on that particular date
    """
    data_dict = wrangle("data")
    df = data_dict["timeage"]
    mask = df["date"] == date
    data = df[mask].drop(columns=["date", "time"])
    fig = px.histogram(
        data_frame=data,
        y="age",
        x=["confirmed", "deceased"],
        barmode="group",
    )
    fig.update_layout(
        plot_bgcolor="#e5e5e5",
        yaxis_title="Age Group",
        xaxis_title="Counts",
        width=800,
        height=500,
    )
    return fig


def get_cumulative() -> Figure:
    """
    This function takes a date as a string in the format YY/MM/DD and returns a histogram
    of the confirmed and diseased cases accumulated on that particular date
    """
    data_dict = wrangle("data")
    dataa = data_dict["timeage"]
    new = dataa.drop(columns="time")
    new["deceased_rate"] = new.deceased / new.confirmed * 100.0
    pivot = new.pivot(index="date", columns="age", values="confirmed")
    pivot2 = new.pivot(index="date", columns="age", values="deceased")
    pivot3 = new.pivot(index="date", columns="age", values="deceased_rate")
    fig = make_subplots(rows=3, cols=1)
    j = 0
    for i in pivot.columns:
        fig.append_trace(
            go.Scatter(
                x=pivot.index,
                y=list(pivot[i].values),
                name=f"{i}",
                legendgroup=f"{i}",
                showlegend=False,
                line=dict(color=f"{colors[j]}"),
            ),
            row=1,
            col=1,
        )

        fig.append_trace(
            go.Scatter(
                x=pivot3.index,
                y=list(pivot2[i].values),
                name=f"{i}",
                legendgroup=f"{i}",
                showlegend=False,
                line=dict(color=f"{colors[j]}"),
            ),
            row=2,
            col=1,
        )

        fig.append_trace(
            go.Scatter(
                x=pivot3.index,
                y=list(pivot3[i].values),
                name=f"{i}",
                legendgroup=f"{i}",
                line=dict(color=f"{colors[j]}"),
            ),
            row=3,
            col=1,
        )
        j = j + 1

    fig.update_layout(
        height=800, width=900, title_text="Cumulative Plots", plot_bgcolor="#e5e5e5"
    )
    fig["layout"]["xaxis3"]["title"] = "Date"
    fig["layout"]["yaxis2"]["title"] = "Deceased"
    fig["layout"]["yaxis3"]["title"] = "Mortality Rate"
    fig["layout"]["yaxis"]["title"] = "Confirmed Cases"
    return fig


def remove_outliers(
    df: pd.DataFrame, col_list: list[str], q1: float = 0.25, q2: float = 0.75
) -> pd.DataFrame:
    """This function removes outliers based on the interquantile range by default it takes the q1 25% and q2 75% quantiles
    for the IQR calculation
    Example:
    >>>"""
    Q1 = df[col_list].quantile(q1)
    Q2 = df[col_list].quantile(q2)
    IQR = Q2 - Q1
    bot = (df[col_list] < Q1 - 1.5 * IQR).any(axis=1)
    top = (df[col_list] > Q2 + 1.5 * IQR).any(axis=1)
    df = df[~(top | bot)]
    return df


def get_pie(option: int, percentage: float = 5) -> Figure:
    """This function calculates and generates a pie plor for the time province data, it shows the
    percentages of dead, released and confirmed on the final date"""
    data_dict = wrangle("data")
    new = data_dict["timeprovince"]
    new = new[new["date"] == "2020-06-30"].drop(columns=["time", "date"])
    new["percentage_confirmed"] = new["confirmed"] / new["confirmed"].sum() * 100
    new["percentage_dead"] = new["deceased"] / new["deceased"].sum() * 100
    new["percentage_released"] = new["released"] / new["released"].sum() * 100
    options = ["percentage_released", "percentage_dead", "percentage_confirmed"]
    mask = new[options[option]] > percentage
    other = (new[options[option]][~mask]).sum()
    new = new[mask]
    new.loc["Other"] = other
    new.loc["Other", "province"] = "Other"
    fig = px.pie(new, values=options[option], names="province", hole=0.4)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_color="#0d0c0c",
        marker=dict(line=dict(color="#000000", width=2)),
    )
    return fig


def get_cum_province(option: int) -> Figure:
    """ "
    This function takes an int from 0 to 2, and creates a plot for the cummulative number of released, deceased or confirmed cases
    by province
    """
    data_dict = wrangle("data")
    intime = data_dict["timeprovince"]
    options = ["released", "deceased", "confirmed"]
    pivot = intime.pivot(index="date", columns="province", values=options[option])
    pivot["total"] = pivot.sum(axis=1)
    fig = px.line(data_frame=pivot, x=pivot.index, y=list(pivot.columns))
    fig.update_layout(
        plot_bgcolor="#e5e5e5",
        xaxis_title="Date",
        yaxis_title=f"{options[option].capitalize()} Counts",
    )

    return fig


def elderly_pop(sorted: bool) -> Figure:
    """
    This function takes a boolean to indicate the sorting of the histogram, when false it is according to the elderly population,
    otherwise acording to the confirmed cases, it then creates a bar plot of the percentage of elderly people in that province population
    """
    data_dict = wrangle("data")
    new = data_dict["timeprovince"]
    region = data_dict["region"]
    mask = region["province"] != "Korea"
    sorter = list(new.sort_values(by="confirmed", ascending=False)["province"])
    new_region = (
        region[mask]
        .groupby("province")
        .mean()
        .elderly_population_ratio.sort_values()
        .to_frame()
        .reset_index()
    )
    if sorted == "Population":
        new_region = new_region.set_index("province").loc[sorter].reset_index()
    fig = new_region.plot.hist(x="province", y="elderly_population_ratio")
    fig.update_layout(
        plot_bgcolor="#e5e5e5",
        yaxis_title=f"Elderly Population ({sorted})",
        xaxis_title="Province",
    )
    return fig


def get_sex(option: str) -> Figure:
    """
    This function takes a string (either confirmed or deceased) and computes a pie chart of the percentages of male and females cases
    on the final date
    """
    data_dict = wrangle("data")
    data = data_dict["timegender"]
    mask = data["date"] == "2020-06-30"
    fig = px.pie(data[mask], values=option.lower(), names=["Male", "Female"], hole=0.4)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_color="#0d0c0c",
        marker=dict(line=dict(color="#000000", width=2)),
    )
    fig.update_layout(font={"size": 18})
    return fig


def get_cum_sex(option: str) -> Figure:
    """
    This function takes a string (either confirmed or deceased) and computes a line plot of the cumulative cases for males, females and the total in time
    """
    data_dict = wrangle("data")
    intime = data_dict["timegender"]
    pivot = intime.pivot(index="date", columns="sex", values=option.lower())
    pivot["Both"] = pivot.sum(axis=1)
    pivot.rename(columns={"female": "Female", "male": "Male"}, inplace=True)
    fig = px.line(
        data_frame=pivot,
        x=pivot.index,
        y=list(pivot.columns),
    )
    fig.update_layout(
        plot_bgcolor="#e5e5e5",
        yaxis_title="Infected",
        xaxis_title="Date",
        legend_title_text="Gender",
    )
    return fig


def get_cum_test(daily: str = "Cummulative") -> Figure:
    """
    This function takes a string, if it is equal to daily then it computes the
    daily results of tests and plots them, otherwise it does the same for the
    cummulative results
    """
    data_dict = wrangle("data")
    intime = data_dict["time"]
    df = intime.drop(columns="time").set_index("date")
    if daily == "Daily":
        df = df.diff().dropna()
    df.rename(
        columns={
            "test": "All Test",
            "confirmed": "Positive Tests",
            "negative": "Negative tests",
            "released": "Infected released",
            "deceased": "Dead due to Covid",
        },
        inplace=True,
    )
    fig = px.line(data_frame=df, x=df.index, y=df.columns)
    fig.update_layout(
        plot_bgcolor="#e5e5e5",
        yaxis_title="Count",
        xaxis_title="Date",
        legend_title_text="Result",
    )
    return fig


def group() -> Figure:
    """
    This function compues a pie about infection cases, specifically the percentages of group vs individual infections
    """
    data_dict = wrangle("data")
    data = data_dict["case"]
    fig = px.pie(
        data["group"].value_counts(),
        values="group",
        names=["Individual infection", "Group infection"],
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_color="#0d0c0c",
        marker=dict(line=dict(color="#000000", width=2)),
    )
    fig.update_layout(font={"size": 18})

    return fig


def path(n: int, location: str = False):
    """
    This function returns a plot about the most common paths of infection, it's arguments are
    n which denotes the number of elements to consider (the rest are set to the other category)
    and location which when set to city looks at the paths of infection by city, not globally
    """
    data_dict = wrangle("data")
    data = data_dict["case"]
    rrr = (
        data[["infection_case", "confirmed"]]
        .groupby("infection_case")
        .sum()
        .sort_values(by="confirmed", ascending=False)
        .reset_index()
    )
    if location == "City":
        rrr = (
            data[["infection_case", "city", "confirmed"]]
            .groupby(["infection_case", "city"])
            .sum()
            .sort_values(by="confirmed", ascending=False)
            .reset_index()
        )
        rrr["with_loc"] = rrr["infection_case"] + " " + rrr["city"]
        new = rrr.drop(columns=["infection_case", "city"])
        new = new.rename(columns={"with_loc": "infection_case"})
        rrr = (
            new[["infection_case", "confirmed"]]
            .groupby("infection_case")
            .sum()
            .sort_values(by="confirmed", ascending=False)
            .reset_index()
        )

    mask = rrr.loc[n - 1, "confirmed"] > rrr["confirmed"]
    new = rrr[~mask]
    new.loc[n] = ["Other", rrr[mask]["confirmed"].sum()]
    fig = px.pie(new, values="confirmed", names="infection_case", hole=0.4)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_color="#0d0c0c",
        marker=dict(line=dict(color="#000000", width=2)),
    )
    fig.update_layout(font={"size": 18})

    return fig


def weather(option: string = "Humidity", date: bool = False) -> Figure:
    """
    This function plots the average whete if bool equals false, otherwise it does so by date,
    option denotes the weather characteristic to plot.
    """
    data_dict = wrangle("data")
    weatherdf = data_dict["weather"]
    options = {
        "Relative Humidity": "avg_relative_humidity",
        "Wind Speed": "max_wind_speed",
        "Precipitation": "precipitation",
    }
    avgw = (
        weatherdf.groupby("province")
        .mean()
        .reset_index()
        .sort_values(by=[options[option]], ascending=False)
    )
    if date:
        mask = weatherdf["date"] == date
        avgw = weatherdf[mask].sort_values(by=[options[option]], ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=avgw["province"],
            x=avgw[options[option]],
            name=options[option],
            orientation="h",
            marker_color=px.colors.qualitative.Set3[3],
        )
    )
    fig.update_layout(
        barmode="overlay",
        xaxis_title=option,
        yaxis_title="Province",
        plot_bgcolor="#e5e5e5",
    )
    return fig


def temp(date: bool = False) -> Figure:
    """
    This function plots the average temperature, plus the max and min over the Covid period unless date is set to True,
    Then it plots the average,max and min by date
    """
    data_dict = wrangle("data")
    weather = data_dict["weather"]
    avgw = (
        weather.groupby("province")
        .mean()
        .reset_index()
        .sort_values(by=["avg_temp"], ascending=False)
    )
    if date:
        mask = weather["date"] == date
        avgw = weather[mask].sort_values(by=["avg_temp"], ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=avgw["province"],
            x=avgw["max_temp"],
            name="Max Temperature",
            orientation="h",
            marker_color=px.colors.qualitative.Set3[3],
        )
    )
    fig.add_trace(
        go.Bar(
            y=avgw["province"],
            x=avgw["avg_temp"],
            name="Mean Temperature",
            orientation="h",
            marker_color=px.colors.qualitative.Set3[4],
        )
    )
    fig.add_trace(
        go.Bar(
            y=avgw["province"],
            x=avgw["min_temp"],
            name="Min Temperature",
            orientation="h",
            marker_color=px.colors.qualitative.Set3[5],
        )
    )
    fig.update_layout(barmode="overlay", plot_bgcolor="#e5e5e5")
    return fig


def get_cum_weather(option: str) -> Figure:
    """
    This function plots the time series data of the weather
    """
    data_dict = wrangle("data")
    options = {
        "Relative Humidity": "avg_relative_humidity",
        "Wind Speed": "max_wind_speed",
        "Precipitation": "precipitation",
        "Average Temperature": "avg_temp",
        "Min Temperature": "min_temp",
        "Max Temperature": "max_temp",
    }
    intime = data_dict["weather"]
    pivot = intime.pivot(index="date", columns="province", values=options[option])
    fig = px.line(
        data_frame=pivot,
        x=pivot.index,
        y=list(pivot.columns),
        color_discrete_sequence=px.colors.qualitative.Light24_r,
    )
    fig.update_layout(
        yaxis_title=option,
        xaxis_title="Date",
        plot_bgcolor="#e5e5e5",
        legend_title_text="Province",
        xaxis_range=[datetime.datetime(2020, 1, 16), datetime.datetime(2020, 3, 5)],
    )
    return fig


def contact(
    cut: int = 7e10, describe: bool = False, confirmed: bool = False, min: bool = False
) -> Figure:
    """
    This function plots the distribution for the contact number of infected patients, if describe
    equals true it instead returns a dataframe with the summary of the data, if confirmed equals true
    it returns a bar instead of a violin/jittered/box plot. if min is true then it sets the minimum contact
    number to 1
    """
    data_dict = wrangle("data")
    patient = data_dict["patientinfo"]
    patient.contact_number = patient.contact_number.replace("-", None)
    mask = ~patient.contact_number.isna()
    contact = patient[mask]
    contact.contact_number = list(map(int, contact.contact_number))
    if min == "True":
        maskara = contact.contact_number == 0
        contact.contact_number[maskara] = 1

    maska = contact.contact_number < cut
    if cut == 0:
        maska = contact.contact_number < 7e10
    if confirmed:
        bar = (
            contact[maska][["contact_number", "confirmed_date"]]
            .groupby("confirmed_date")
            .mean()
            .reset_index()
        )
        bar2 = (
            contact[maska][["contact_number", "released_date"]]
            .groupby("released_date")
            .mean()
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=bar["confirmed_date"],
                y=bar["contact_number"],
                name="Confirmed date",
            )
        )
        fig.add_trace(
            go.Bar(
                x=bar2["released_date"].dropna(),
                y=bar2["contact_number"],
                name="Released date",
            )
        )
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Date",
            yaxis_title="Count",
            plot_bgcolor="#e5e5e5",
        )
        return fig
    if describe:
        return contact["contact_number"][maska].describe()
    fig = go.Figure(
        data=go.Violin(
            y=contact[maska]["contact_number"],
            box_visible=True,
            line_color="black",
            meanline_visible=True,
            fillcolor=px.colors.qualitative.Set3[4],
            opacity=0.6,
            x0="Distribution",
        )
    )
    fig.update_traces(
        points="all",  # show all points
        jitter=0.4,  # add some jitter on points for better visibility
        scalemode="count",
    )
    fig.update_layout(
        yaxis_zeroline=False, plot_bgcolor="#e5e5e5", yaxis_title="Contact Number"
    )
    return fig


def searchtrend(
    date1: datetime,
    date2: datetime,
) -> Figure:
    """
    This function takes two dates as input and then plots the search trend time series on naver (of infectious diseases)
    between those two dates
    """
    data_dict = wrangle("data")
    search = data_dict["searchtrend"]
    mask = (search.date >= date1) & (search.date <= date2)
    search.rename(columns=str.capitalize, inplace=True)
    fig = px.line(search[mask], x="Date", y=["Cold", "Flu", "Pneumonia", "Coronavirus"])
    fig.update_layout(
        yaxis_title="Relative interest (0-100)",
        plot_bgcolor="#e5e5e5",
        legend_title_text="Disease",
    )
    return fig


def policies() -> Figure:
    """
    This function plots a pie plot about the types of policies that were implemented in S. Korea during the COVID-19
    pandemic
    """
    data_dict = wrangle("data")
    pol = data_dict["policy"]
    first = pol[["policy_id", "type"]].groupby("type").count().reset_index()
    fig = px.pie(first, values="policy_id", names="type", hole=0.4)
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_color="#0d0c0c",
        marker=dict(line=dict(color="#000000", width=2)),
    )
    fig.update_layout(font={"size": 18})
    return fig


def policies_dates() -> Figure:
    """
    This function plots a violin plot and a jittered plot of the distribution of dates on which policies started
    """
    data_dict = wrangle("data")
    pol = data_dict["policy"]
    fig = go.Figure(
        data=go.Violin(
            x=pol["start_date"],
            box_visible=True,
            line_color="black",
            meanline_visible=True,
            fillcolor=px.colors.qualitative.Set3[4],
            opacity=0.6,
            x0="Contact Number",
            name="Policy start date distribution",
        )
    )
    fig.update_traces(
        points="all",  # show all points
        jitter=0.2,  # add some jitter on points for better visibility
        scalemode="count",
    )
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="#e5e5e5")
    return fig


def policies_month() -> Figure:
    """
    This function returns a bar chart about the number of policies implemented each month during the COVID 19
    pandemic in south korea
    """
    data_dict = wrangle("data")
    pol = data_dict["policy"]
    pol["start_date"] = pd.to_datetime(
        pol["start_date"], format="%Y-%m-%d", errors="ignore"
    )
    data = (
        pol.set_index("start_date")["policy_id"]
        .groupby(pd.Grouper(freq="M"))
        .count()
        .reset_index()
    )
    fig = px.bar(data, x="start_date", y="policy_id")
    fig.update_layout(
        yaxis_zeroline=False,
        plot_bgcolor="#e5e5e5",
        xaxis_title="Start date",
        yaxis_title="Number of Policies",
    )
    return fig


def get_trend_con(regression: bool = False, lag: int = 0) -> Figure:
    """
    This function plots the search trend of coronavirus on the naver platform and the confirmed number
    of cases in S.Korea, if regression equals true it also builds a linear regression model shifted by lag days
    and returns a plot of the model and the real data
    """
    data_dict = wrangle("data")
    one = data_dict["searchtrend"]
    mask = one["date"] > "2020-01-20"
    one = one[mask].set_index("date")
    one = one.add_suffix("_search")
    three = data_dict["time"].set_index("date")
    three = three.add_suffix("_time")
    final = one.join(three)
    final = final.fillna(0)
    final.index = pd.to_datetime(final.index)
    newf = final.drop(columns="time_time")
    data_dict = wrangle("data")
    intime = data_dict["time"]
    df = intime.drop(columns="time").set_index("date")
    df = df.diff().dropna()
    if regression:
        model = LinearRegression()
        end_date = pd.to_datetime("2020-05-25")
        start_date = pd.to_datetime("2020-02-15")
        mask = (df.index < str(end_date)) & (df.index > str(start_date))
        mask3 = df.index > str(start_date)
        end_date = end_date - timedelta(days=lag)
        date = end_date - timedelta(days=len(df[mask]) + 1)
        mask2 = (newf.index > str(date)) & (newf.index < str(end_date))
        tofit = newf[mask2]
        model.fit(tofit, df["confirmed"][mask])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        mask4 = df[mask3].index < str(end_date)
        mask5 = newf.index > str(start_date)
        mask6 = newf[mask5].index < str(end_date)
        fig.add_trace(
            go.Scatter(
                x=df[mask3].index,
                y=list(df[mask3][mask4].confirmed)
                + list(model.predict(newf[mask5][~mask6])),
                name="Linear Regression Model",
            )
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df.confirmed, name="Real Confirmed cases")
        )
        fig.add_trace(
            go.Scatter(
                x=df[mask3][~mask4].index,
                y=abs(df[mask3][~mask4].confirmed - model.predict(newf[mask5][~mask6])),
                name="Error",
                line=dict(dash="dot"),
            )
        )
        fig.update_layout(
            plot_bgcolor="#e5e5e5",
            yaxis_title="Confirmed Cases",
            xaxis_title="Date",
            xaxis_range=[datetime.datetime(2020, 2, 16), datetime.datetime(2020, 7, 5)],
        )
        fig.add_vline(
            x=end_date,
            line_width=3,
            line_dash="dash",
            line_color="green",
        )
        return fig
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=newf.index, y=newf.coronavirus_search, name="Search Trend")
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df.confirmed, name="Confirmed cases"), secondary_y=True
    )
    fig.update_layout(
        plot_bgcolor="#e5e5e5", yaxis_title="Relative Interest (%)", xaxis_title="Date"
    )
    fig.update_yaxes(title_text="Confirmed cases", secondary_y=True)
    return fig


def arima(update: bool = False) -> Figure:
    """
    This function performs AutoRegressive Integrated Moving Average and walk forward validation (if update is True)
    on the confirmed daily covid cases in S. Korea, we then use this to predict future values with or withouth walking forward validation.
    It then returns a plot of the model
    """
    data_dict = wrangle("data")
    intime = data_dict["time"]
    df = intime.drop(columns="time").set_index("date")
    df = df.diff().dropna()
    X = df["confirmed"].values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size : len(X)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(0, len(test)):
        model = ARIMA(history, order=(10, 3, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        if update:
            obs = yhat
        history.append(obs)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["confirmed"], name="Confirmed Cases"))
    fig.add_trace(go.Scatter(x=df.index, y=train.tolist() + predictions, name="ARIMA"))
    fig.add_vline(
        x=df.confirmed.index.tolist()[int(len(X) * 0.66)],
        line_width=3,
        line_dash="dash",
        line_color="green",
    )
    fig.update_layout(
        plot_bgcolor="#e5e5e5", yaxis_title="Confirmed cases", xaxis_title="Date"
    )
    return fig


def pca_data(data: bool = False) -> Figure:
    """
    This function creates a dataset from the csv files we believe are useful for forecasting and applies pca on them. if data=True
    then it returns the dataset. otherwise it returns a plot of the PCA components
    """
    data_dict = wrangle("data")
    one = data_dict["searchtrend"]
    mask = one["date"] > "2020-01-01"
    one = one[mask].set_index("date")
    one = one.add_suffix("_search")
    three = data_dict["time"].set_index("date")
    three = three.add_suffix("_time")
    final = one.join(three)
    final = final.fillna(0)
    final.index = pd.to_datetime(final.index)
    if data:
        return final
    pca = PCA(n_components=2)
    sc = StandardScaler()
    X_train = sc.fit_transform(final)
    X_train = pca.fit_transform(X_train)
    fig = px.scatter(X_train[:, 0], X_train[:, 1])
    fig.update_layout(plot_bgcolor="#e5e5e5", yaxis_title="PCA 1", xaxis_title="PCA 2")
    return fig


def seoulpop(x: str = "Population") -> Figure:
    """
    This function returns a chart about seouls floating population according to the variable x. if x is population
    then it returns a barchar of the mean floating population by date, if gender then it returns a pie plot about the genders
    of the mean floating point population and if age it returns a pie plot of their age group
    """
    data_dict = wrangle("data")
    data = data_dict["seoulfloating"]
    if x == "Population":
        fig = px.bar(
            x=data.date.unique(),
            y=data.groupby(["date", "hour"])
            .sum()
            .groupby("date")
            .fp_num.mean()
            .apply(lambda x: x / 1e6),
        )
        fig.update_layout(
            plot_bgcolor="#e5e5e5",
            yaxis_title="Floating population (millions)",
            xaxis_title="Date",
        )

    if x == "Gender":
        labels = ["Female", "Male"]
        seoul_female, seoul_male = data[data.sex == "female"], data[data.sex == "male"]
        values = [seoul_female.fp_num.sum(), seoul_male.fp_num.sum()]
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    textinfo="label+percent",
                    insidetextorientation="radial",
                    hole=0.4,
                )
            ]
        )
        fig.update_layout(font={"size": 18})

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont_color="#0d0c0c",
            marker=dict(line=dict(color="#000000", width=2)),
        )
    if x == "Age":
        float_age_order = list(
            data.groupby("birth_year")
            .sum()
            .sort_values("fp_num", ascending=False)
            .index
        )
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=float_age_order,
                    values=data.groupby("birth_year")
                    .sum()
                    .sort_values("fp_num", ascending=False)
                    .fp_num,
                    textinfo="label+percent",
                    insidetextorientation="radial",
                    hole=0.4,
                )
            ]
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont_color="#0d0c0c",
            marker=dict(line=dict(color="#000000", width=2)),
        )
        fig.update_layout(font={"size": 18})

    return fig
