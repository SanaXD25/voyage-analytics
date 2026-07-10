
import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_prep import load_raw, build_user_features  # noqa: E402
from recommender import recommend_for_user  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def get_raw_data():
    return load_raw()


@st.cache_data
def get_user_features(_users, _flights, _hotels):
    return build_user_features(_users, _flights, _hotels)


@st.cache_resource
def get_models():
    models = {}
    for name, fname in [
        ("flight_price", "flight_price_model.joblib"),
        ("hotel_price", "hotel_price_model.joblib"),
        ("segmentation", "segmentation_model.joblib"),
        ("gender", "gender_model.joblib"),
        ("recommender", "recommender_artifacts.joblib"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        models[name] = joblib.load(path) if os.path.exists(path) else None
    return models


users, flights, hotels = get_raw_data()
user_features = get_user_features(users, flights, hotels)
models = get_models()

CITIES = sorted(flights["from"].unique())
FLIGHT_CLASSES = sorted(flights["flightType"].unique())
AGENCIES = sorted(flights["agency"].unique())
HOTEL_NAMES = sorted(hotels["name"].unique())
PLACES = sorted(hotels["place"].unique())
COMPANIES = sorted(users["company"].unique())
GENDERS = sorted(users["gender"].unique())

MISSING_MODEL_MSG = (
    "Model file not found. Run the corresponding script in `src/` "
    "(e.g. `python src/train_price_model.py`) to train and save it first."
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("✈️ Voyage Analytics")
st.sidebar.caption("MLOps in Travel — Productionization of ML Systems")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Price Predictor", "Customer Segments", "Gender Classifier", "Recommender"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dataset**\n\n"
    f"- {len(users):,} users\n"
    f"- {len(flights):,} flight legs\n"
    f"- {len(hotels):,} hotel bookings\n"
    f"- {len(CITIES)} cities"
)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Voyage Analytics — Overview")
    st.write(
        "This dashboard serves four ML models trained on a Brazilian travel "
        "platform's booking data: flight/hotel price prediction, customer "
        "segmentation, gender classification, and a destination recommender."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", f"{len(users):,}")
    c2.metric("Flight Legs", f"{len(flights):,}")
    c3.metric("Hotel Bookings", f"{len(hotels):,}")
    c4.metric("Total Flight Revenue", f"${flights['price'].sum():,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Flights by Route (Top 10)")
        route_counts = (
            (flights["from"] + " → " + flights["to"]).value_counts().head(10).reset_index()
        )
        route_counts.columns = ["route", "count"]
        fig = px.bar(route_counts, x="count", y="route", orientation="h")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Flight Price Distribution by Class")
        fig = px.box(flights, x="flightType", y="price", color="flightType")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Bookings by Agency")
        fig = px.pie(flights, names="agency", title="Flight bookings by agency")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Avg Hotel Price by City")
        avg_hotel = hotels.groupby("place")["price"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_hotel, x="place", y="price")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Price Predictor
# ---------------------------------------------------------------------------
elif page == "Price Predictor":
    st.title("Price Predictor")
    tab1, tab2 = st.tabs(["Flight Price", "Hotel Price"])

    with tab1:
        st.subheader("Predict a flight fare")
        if models["flight_price"] is None:
            st.warning(MISSING_MODEL_MSG)
        else:
            c1, c2, c3 = st.columns(3)
            origin = c1.selectbox("From", CITIES, index=0)
            dest = c2.selectbox("To", [c for c in CITIES if c != origin], index=0)
            flight_class = c3.selectbox("Class", FLIGHT_CLASSES)

            c4, c5, c6 = st.columns(3)
            agency = c4.selectbox("Agency", AGENCIES)
            distance = c5.number_input("Distance (km)", min_value=50.0, max_value=3000.0, value=676.5)
            time_hr = c6.number_input("Flight time (hours)", min_value=0.3, max_value=6.0, value=1.76)

            c7, c8 = st.columns(2)
            month = c7.slider("Month", 1, 12, 6)
            day_of_week = c8.slider("Day of week (0=Mon)", 0, 6, 2)

            if st.button("Predict flight price", type="primary"):
                X = pd.DataFrame([{
                    "distance": distance, "time": time_hr, "month": month, "day_of_week": day_of_week,
                    "from": origin, "to": dest, "flightType": flight_class, "agency": agency,
                }])
                pred = models["flight_price"].predict(X)[0]
                st.success(f"Predicted flight price: **${pred:,.2f}**")

    with tab2:
        st.subheader("Predict a hotel rate")
        if models["hotel_price"] is None:
            st.warning(MISSING_MODEL_MSG)
        else:
            c1, c2, c3 = st.columns(3)
            hotel_name = c1.selectbox("Hotel", HOTEL_NAMES)
            place = c2.selectbox("City", PLACES)
            days = c3.number_input("Stay length (days)", min_value=1, max_value=14, value=3)

            c4, c5 = st.columns(2)
            month = c4.slider("Month ", 1, 12, 6)
            day_of_week = c5.slider("Day of week (0=Mon) ", 0, 6, 2)

            if st.button("Predict hotel price", type="primary"):
                X = pd.DataFrame([{
                    "days": days, "month": month, "day_of_week": day_of_week,
                    "name": hotel_name, "place": place,
                }])
                pred = models["hotel_price"].predict(X)[0]
                st.success(f"Predicted nightly hotel price: **${pred:,.2f}**  →  total for {days} nights: **${pred*days:,.2f}**")

# ---------------------------------------------------------------------------
# Page: Customer Segments
# ---------------------------------------------------------------------------
elif page == "Customer Segments":
    st.title("Customer Segmentation")
    if models["segmentation"] is None:
        st.warning(MISSING_MODEL_MSG)
    else:
        seg_path = os.path.join(MODEL_DIR, "user_segments.csv")
        if not os.path.exists(seg_path):
            st.warning("Run `python src/train_segmentation.py` to generate segment assignments.")
        else:
            df = pd.read_csv(seg_path)
            st.write("Each user is grouped into one of four behavioural segments based on trip "
                     "frequency, spend, and stay patterns.")

            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(df, names="segment_name", title="Segment sizes")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(
                    df, x="n_trips", y="total_spend", color="segment_name",
                    hover_data=["name", "age", "company"], title="Trips vs. Total Spend"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Segment profiles (averages)")
            profile_cols = ["age", "n_trips", "avg_flight_price", "total_spend", "avg_stay_days", "days_active"]
            st.dataframe(df.groupby("segment_name")[profile_cols].mean().round(1))

            st.subheader("Look up a user")
            user_code = st.selectbox("User code", df["userCode"].sort_values().tolist())
            row = df[df["userCode"] == user_code].iloc[0]
            st.info(f"**{row['name']}** ({row['company']}) → Segment: **{row['segment_name']}** | "
                    f"{int(row['n_trips'])} trips | ${row['total_spend']:,.0f} total spend")

# ---------------------------------------------------------------------------
# Page: Gender Classification
# ---------------------------------------------------------------------------
elif page == "Gender Classifier":
    st.title("Gender Classifier")
    st.caption(
        "Predicts a user's gender (male / female / none) from demographics "
        "and travel behaviour — trip frequency, spend, favourite flight "
        "class and agency."
    )
    if models["gender"] is None:
        st.warning(MISSING_MODEL_MSG)
    else:
        st.warning(
            "⚠️ Honest note: this model's accuracy is ~31% on held-out data, "
            "vs. a ~33% random-guess baseline across three balanced classes. "
            "Travel behaviour in this dataset doesn't meaningfully predict "
            "gender. It's included as a working pipeline end-to-end (data → "
            "features → model → serving) per the project spec, not as a "
            "reliable classifier — this dataset simply doesn't carry that "
            "signal."
        )

        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 80, 35)
        company = c2.selectbox("Company", COMPANIES)
        fav_flight_class = c3.selectbox("Favourite flight class", FLIGHT_CLASSES)

        c4, c5, c6 = st.columns(3)
        fav_agency = c4.selectbox("Favourite agency", AGENCIES)
        n_trips = c5.number_input("Number of trips", 0, 300, 20)
        avg_flight_price = c6.number_input("Avg flight price ($)", 0.0, 5000.0, 800.0)

        c7, c8, c9 = st.columns(3)
        total_flight_spend = c7.number_input("Total flight spend ($)", 0.0, 500000.0, 15000.0)
        avg_distance = c8.number_input("Avg flight distance (km)", 0.0, 3000.0, 700.0)
        n_hotel_bookings = c9.number_input("Number of hotel bookings", 0, 300, 15)

        c10, c11 = st.columns(2)
        avg_hotel_price = c10.number_input("Avg hotel price ($)", 0.0, 1000.0, 150.0)
        total_hotel_spend = c11.number_input("Total hotel spend ($)", 0.0, 500000.0, 5000.0)
        avg_stay_days = st.number_input("Avg stay length (days)", 0.0, 14.0, 3.0)

        if st.button("Predict gender", type="primary"):
            X = pd.DataFrame([{
                "age": age, "n_trips": n_trips, "avg_flight_price": avg_flight_price,
                "total_flight_spend": total_flight_spend, "avg_distance": avg_distance,
                "n_hotel_bookings": n_hotel_bookings, "avg_hotel_price": avg_hotel_price,
                "total_hotel_spend": total_hotel_spend, "avg_stay_days": avg_stay_days,
                "company": company, "fav_flight_class": fav_flight_class, "fav_agency": fav_agency,
            }])
            pred = models["gender"].predict(X)[0]
            probs = models["gender"].predict_proba(X)[0]
            classes = models["gender"].classes_
            st.metric("Predicted gender", pred.capitalize())
            st.write({c: f"{p:.1%}" for c, p in zip(classes, probs)})

# ---------------------------------------------------------------------------
# Page: Recommender
# ---------------------------------------------------------------------------
elif page == "Recommender":
    st.title("Destination & Hotel Recommender")
    st.caption(
        "Recommends new destinations (or, if a user has already visited every "
        "city in the network, the ones worth revisiting) matched to their "
        "typical hotel budget, plus the cheapest hotel option there."
    )
    if models["recommender"] is None:
        st.warning(MISSING_MODEL_MSG)
    else:
        user_code = st.selectbox("User code", users["code"].sort_values().tolist())
        user_row = users[users["code"] == user_code].iloc[0]
        st.write(f"**{user_row['name']}** — {user_row['company']}, age {user_row['age']}")

        n = st.slider("Number of recommendations", 1, 5, 3)
        if st.button("Get recommendations", type="primary"):
            recs = recommend_for_user(user_code, models["recommender"], n=n)
            if not recs:
                st.info("No recommendations available for this user.")
            for r in recs:
                with st.container(border=True):
                    st.markdown(f"### {r['place']}")
                    st.write(f"Popularity score: {r['popularity']:,} | "
                             f"Avg hotel price here: ${r['avg_hotel_price_in_place']:,.2f}")
                    st.write(f"Suggested hotel: **{r['suggested_hotel']}** "
                             f"(${r['suggested_hotel_price']:,.2f}/night)")
                    st.caption(r["reason"].capitalize())
