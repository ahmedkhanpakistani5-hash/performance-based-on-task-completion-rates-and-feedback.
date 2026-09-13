import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from groq import Groq


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Intern Performance Predictor",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="title">📊 Intern Performance Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning + Groq AI to predict intern performance'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# CREATE SAMPLE TRAINING DATA
# --------------------------------------------------

@st.cache_data
def create_dataset():

    np.random.seed(42)

    data = []

    for i in range(300):

        completion_rate = np.random.randint(50, 101)
        completion_time = np.random.uniform(1, 12)
        feedback_rating = np.random.uniform(1, 5)
        attendance = np.random.randint(60, 101)
        tasks_completed = np.random.randint(5, 31)
        late_tasks = np.random.randint(0, 8)

        performance = (
            completion_rate * 0.35
            + feedback_rating * 10 * 0.25
            + attendance * 0.20
            + tasks_completed * 1.2
            - completion_time * 1.5
            - late_tasks * 2
        )

        performance += np.random.normal(0, 5)

        performance = max(0, min(100, performance))

        data.append([
            completion_rate,
            completion_time,
            feedback_rating,
            attendance,
            tasks_completed,
            late_tasks,
            performance
        ])

    columns = [
        "Task_Completion_Rate",
        "Average_Completion_Time",
        "Feedback_Rating",
        "Attendance",
        "Tasks_Completed",
        "Late_Tasks",
        "Performance_Score"
    ]

    return pd.DataFrame(data, columns=columns)


df = create_dataset()


# --------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

features = [
    "Task_Completion_Rate",
    "Average_Completion_Time",
    "Feedback_Rating",
    "Attendance",
    "Tasks_Completed",
    "Late_Tasks"
]

X = df[features]
y = df["Performance_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Intern Information")

task_completion = st.sidebar.slider(
    "Task Completion Rate (%)",
    0,
    100,
    80
)

completion_time = st.sidebar.slider(
    "Average Task Completion Time (hours)",
    1.0,
    20.0,
    5.0,
    0.5
)

feedback = st.sidebar.slider(
    "Feedback Rating",
    1.0,
    5.0,
    4.0,
    0.1
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    0,
    100,
    85
)

tasks_completed = st.sidebar.number_input(
    "Tasks Completed",
    min_value=0,
    max_value=100,
    value=15
)

late_tasks = st.sidebar.number_input(
    "Late Tasks",
    min_value=0,
    max_value=50,
    value=2
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

input_data = pd.DataFrame({
    "Task_Completion_Rate": [task_completion],
    "Average_Completion_Time": [completion_time],
    "Feedback_Rating": [feedback],
    "Attendance": [attendance],
    "Tasks_Completed": [tasks_completed],
    "Late_Tasks": [late_tasks]
})

prediction = model.predict(input_data)[0]

prediction = max(0, min(100, prediction))


# --------------------------------------------------
# PERFORMANCE CATEGORY
# --------------------------------------------------

if prediction >= 75:
    category = "Likely to Excel"
    emoji = "🟢"

elif prediction >= 55:
    category = "Average Performance"
    emoji = "🟡"

else:
    category = "Likely to Struggle"
    emoji = "🔴"


# --------------------------------------------------
# MAIN RESULTS
# --------------------------------------------------

st.subheader("🎯 Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted Performance",
        f"{prediction:.1f}/100"
    )

with col2:
    st.metric(
        "Prediction Category",
        category
    )

with col3:
    st.metric(
        "Model R² Score",
        f"{r2:.2f}"
    )


st.progress(int(prediction))

st.markdown(
    f"## {emoji} {category}"
)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

with st.expander("📈 Model Performance"):

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Mean Absolute Error",
            f"{mae:.2f}"
        )

    with col2:
        st.metric(
            "R² Score",
            f"{r2:.2f}"
        )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader("🔍 What Influenced the Prediction?")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)


# --------------------------------------------------
# GROQ AI ANALYSIS
# --------------------------------------------------

st.subheader("🤖 AI Performance Analysis")

def get_groq_client():

    try:
        api_key = st.secrets["GROQ_API_KEY"]

    except Exception:

        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


if st.button("Generate AI Analysis"):

    client = get_groq_client()

    if client is None:

        st.warning(
            "Groq API key not found. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )

    else:

        prompt = f"""
You are an HR analytics AI assistant.

Analyze the following intern performance prediction.

Predicted performance score: {prediction:.1f}/100
Category: {category}

Task completion rate: {task_completion}%
Average completion time: {completion_time} hours
Feedback rating: {feedback}/5
Attendance: {attendance}%
Tasks completed: {tasks_completed}
Late tasks: {late_tasks}

Provide:

1. A short performance summary.
2. The strongest positive factors.
3. The main areas of concern.
4. Three practical recommendations for the intern.
5. A short manager recommendation.

Keep the response professional, concise, and easy to understand.

Important:
Do not claim that this prediction is certain.
Treat it as an ML-based estimate.
"""

        try:

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HR analytics assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=800
            )

            ai_response = response.choices[0].message.content

            st.markdown(ai_response)

        except Exception as e:

            st.error(
                f"Groq API error: {str(e)}"
            )


# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

with st.expander("📊 View Training Dataset"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Intern Performance Predictor | "
    "Random Forest Regression + Groq AI"
)
