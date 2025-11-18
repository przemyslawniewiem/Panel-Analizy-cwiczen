import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests

# ================ SCRAPING DANYCH ================

def scrape_calories_table():
    """
    Funkcja scrapująca tabelę spalania kalorii ze strony bieganie.pl

    Zwraca:
        pandas.DataFrame: DataFrame zawierający dane z tabeli spalania kalorii
    """
    # URL strony
    url = 'https://bieganie.pl/zdrowie/tabela-spalania-kalorii/'

    # Pobranie zawartości strony
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Znajdź tabelę
    table = soup.find('table')

    # Inicjalizacja list na dane
    activities = []
    calories_per_hour = []
    time_per_100kcal = []

    # Przejście przez wszystkie wiersze tabeli (pomijając nagłówek)
    rows = table.find_all('tr')[1:]  # Pomijamy pierwszy wiersz (nagłówek)

    for row in rows:
        cols = row.find_all('td')
        if len(cols) == 3:  # Upewniamy się, że wiersz ma 3 kolumny
            activities.append(cols[0].text.strip())
            calories_per_hour.append(cols[1].text.strip())
            time_per_100kcal.append(cols[2].text.strip())

    # Utworzenie DataFrame
    df = pd.DataFrame({
        'Rodzaj aktywności': activities,
        'Kalorie na godzinę [kcal]': calories_per_hour,
        'Czas na spalenie 100 kcal': time_per_100kcal
    })

    return df


# Wywołanie funkcji i wyświetlenie wyników
df = scrape_calories_table()
print(df)

# ================ PRZETWARZANIE DANYCH ================

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Wczytanie i oczyszczenie danych dotyczących ćwiczeń członków siłowni
    """
    # Wczytaj plik CSV
    df = pd.read_csv(file_path)

    # Konwersja typów danych na numeryczne
    numeric_columns = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
                       'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
                       'Fat_Percentage', 'Water_Intake (liters)',
                       'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Dodanie nowych cech
    # Wynik intensywności = (Maksymalne tętno / Tętno spoczynkowe) * (Spalone kalorie / Czas treningu)
    df['Intensity_Score'] = (df['Max_BPM'] / df['Resting_BPM']) * (
                df['Calories_Burned'] / df['Session_Duration (hours)'])
    df['Fitness_Level'] = df.apply(calculate_fitness_level, axis=1)

    return df


def calculate_fitness_level(row) -> str:
    """
    Obliczanie poziomu sprawności na podstawie wielu wskaźników
    """
    score = 0

    # Punktacja BMI
    if 18.5 <= row['BMI'] <= 24.9:
        score += 2  # Zdrowe BMI
    elif 25 <= row['BMI'] <= 29.9:
        score += 1  # Lekka nadwaga

    # Punktacja tętna spoczynkowego
    if row['Resting_BPM'] < 60:
        score += 2  # Świetna wydolność
    elif 60 <= row['Resting_BPM'] <= 70:
        score += 1  # Dobra wydolność

    # Punktacja poziomu doświadczenia
    score += row['Experience_Level']

    # Punktacja częstotliwości treningów
    if row['Workout_Frequency (days/week)'] >= 4:
        score += 2  # Częste treningi
    elif row['Workout_Frequency (days/week)'] >= 3:
        score += 1  # Umiarkowane treningi

    # Określenie poziomu sprawności
    if score >= 6:
        return 'Zaawansowany'
    elif score >= 4:
        return 'Średniozaawansowany'
    else:
        return 'Początkujący'


def get_statistics(df: pd.DataFrame) -> Dict:
    """
    Obliczanie kluczowych statystyk z datasetu
    """
    stats = {
        'total_members': len(df),
        'avg_age': df['Age'].mean(),
        'avg_bmi': df['BMI'].mean(),
        'workout_distribution': df['Workout_Type'].value_counts().to_dict(),
        'avg_calories_burned': df['Calories_Burned'].mean(),
        'fitness_levels': df['Fitness_Level'].value_counts().to_dict()
    }
    return stats


def analyze_workout_effectiveness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza efektywności treningów na podstawie spalonych kalorii i intensywności
    """
    effectiveness = df.groupby('Workout_Type').agg({
        'Calories_Burned': 'mean',
        'Session_Duration (hours)': 'mean',
        'Intensity_Score': 'mean'
    }).round(2)

    # Dodanie Efficiency Score (kalorie/czas)
    effectiveness['Efficiency_Score'] = (
            effectiveness['Calories_Burned'] / effectiveness['Session_Duration (hours)']
    ).round(2)

    return effectiveness


# ================ FUNKCJE ANALIZY I WIZUALIZACJI ================

def create_workout_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """
    Tworzenie wykresu kołowego pokazującego rozkład typów treningów
    """
    workout_dist = df['Workout_Type'].value_counts()
    fig = px.pie(
        values=workout_dist.values,
        names=workout_dist.index,
        title='Rozkład Typów Treningów',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    return fig


def create_calories_boxplot(df: pd.DataFrame) -> go.Figure:
    """
    Tworzenie wykresu pudełkowego pokazującego spalone kalorie według typu treningu
    """
    fig = px.box(
        df,
        x='Workout_Type',
        y='Calories_Burned',
        title='Rozkład Spalonych Kalorii według Typu Treningu',
        color='Workout_Type'
    )
    return fig


def create_experience_bmi_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Tworzenie wykresu pudełkowego pokazującego BMI vs Poziom Doświadczenia
    """
    # Mapowanie poziomów doświadczenia na etykiety
    experience_labels = {
        1: 'Początkujący',
        2: 'Średniozaawansowany',
        3: 'Ekspert'
    }

    df_temp = df.copy()
    df_temp['Experience_Label'] = df_temp['Experience_Level'].map(experience_labels)

    fig = px.box(
        df_temp,
        x='Experience_Label',
        y='BMI',
        title='Rozkład BMI według Poziomu Doświadczenia',
        color='Experience_Label',
        category_orders={'Experience_Label': ['Początkujący', 'Średniozaawansowany', 'Ekspert']}
    )

    # Dostosowanie układu
    fig.update_layout(
        xaxis_title="Poziom Doświadczenia",
        yaxis_title="BMI",
        showlegend=False  # Ukrycie legendy, która jest zbędna w tym przypadku
    )

    return fig


def create_age_distribution_plot(df: pd.DataFrame) -> go.Figure:
    """
    Tworzenie histogramu pokazującego rozkład wieku
    """
    fig = px.histogram(
        df,
        x='Age',
        color='Gender',
        title='Rozkład Wieku według Płci',
        nbins=20
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Tworzenie mapy cieplnej korelacji dla zmiennych numerycznych
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        title='Mapa Korelacji',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    return fig


def create_custom_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """
    Tworzenie własnego wykresu punktowego z wybranymi zmiennymi
    """
    color_map = {
        'Male': '#2E86C1',  # niebieski dla mężczyzn
        'Female': '#FF69B4'  # różowy dla kobiet
    }

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color='Gender',
        title=f'{y_col} vs {x_col}',
        color_discrete_map=color_map,
        opacity=0.6,
        labels={'Gender': 'Płeć'}
    )

    # Dostosowanie układu
    fig.update_traces(marker=dict(size=8))  # Stały rozmiar punktów
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col
    )

    return fig


def generate_insights(df: pd.DataFrame) -> List[str]:
    """
    Generowanie kluczowych spostrzeżeń z danych
    """
    insights = []

    # Spostrzeżenia związane z wiekiem
    avg_age = df['Age'].mean()
    insights.append(f"Średni wiek członków to {avg_age:.1f} lat")

    # Efektywność treningów
    most_calories = df.groupby('Workout_Type')['Calories_Burned'].mean()
    most_effective = most_calories.idxmax()
    insights.append(f"{most_effective} spala średnio najwięcej kalorii")

    # Rozkład poziomu doświadczenia
    exp_dist = df['Experience_Level'].value_counts(normalize=True) * 100
    insights.append(f"{exp_dist[1]:.1f}% członków to początkujący")

    # Spostrzeżenia dotyczące BMI
    healthy_bmi = df[df['BMI'].between(18.5, 24.9)].shape[0] / len(df) * 100
    insights.append(f"{healthy_bmi:.1f}% członków ma zdrowe BMI")

    return insights


# ================ APLIKACJA STREAMLIT ================

st.set_page_config(page_title="Analiza Członków Siłowni", layout="wide")


def main():
    st.title("Panel Analizy Ćwiczeń Członków Siłowni")

    # Wczytanie i przetworzenie danych
    df = load_and_clean_data('gym_members_exercise_tracking.csv')

    # Filtry w pasku bocznym
    st.sidebar.header("Filtry")
    selected_gender = st.sidebar.multiselect(
        "Wybierz Płeć",
        options=df['Gender'].unique(),
        default=df['Gender'].unique()
    )

    age_range = st.sidebar.slider(
        "Wybierz Zakres Wieku",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )

    # Filtrowanie danych
    filtered_df = df[
        (df['Gender'].isin(selected_gender)) &
        (df['Age'].between(age_range[0], age_range[1]))
        ]

    # Wyświetlanie kluczowych metryk
    st.header("Kluczowe Metryki")
    col1, col2, col3, col4 = st.columns(4)

    stats = get_statistics(filtered_df)
    col1.metric("Liczba Członków", stats['total_members'])
    col2.metric("Średni Wiek", f"{stats['avg_age']:.1f}")
    col3.metric("Średnie BMI", f"{stats['avg_bmi']:.1f}")
    col4.metric("Średnio Spalonych Kalorii", f"{stats['avg_calories_burned']:.0f}")

    # Wyświetlanie wykresów
    st.header("Wizualizacja Danych")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Analiza Treningów",
        "Demografia Członków",
        "Korelacje",
        "Wykres Własny"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_workout_distribution_plot(filtered_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_calories_boxplot(filtered_df), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_age_distribution_plot(filtered_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_experience_bmi_scatter(filtered_df), use_container_width=True)

    with tab3:
        st.plotly_chart(create_correlation_heatmap(filtered_df), use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        # Wybór zmiennych do wykresu własnego
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        with col1:
            x_variable = st.selectbox("Wybierz zmienną na osi X:", numeric_columns)
        with col2:
            y_variable = st.selectbox("Wybierz zmienną na osi Y:", numeric_columns)

        st.plotly_chart(
            create_custom_scatter(filtered_df, x_variable, y_variable),
            use_container_width=True
        )

    # Wyświetlanie spostrzeżeń
    st.header("Kluczowe Spostrzeżenia")
    insights = generate_insights(filtered_df)
    for insight in insights:
        st.info(insight)

    # Wyświetlanie analizy efektywności treningów
    st.header("Analiza Efektywności Treningów")
    effectiveness_df = analyze_workout_effectiveness(filtered_df)
    st.dataframe(effectiveness_df)

    # Wyświetlanie pełnej tabeli danych
    st.header("Pełne Dane")
    show_full_data = st.checkbox("Pokaż wszystkie dane")
    if show_full_data:
        st.dataframe(filtered_df)


if __name__ == "__main__":
    main()