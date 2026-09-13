import streamlit as st
import pandas as pd
import joblib
import os

from groq import Groq


# ------------------------------------------------
# PAGE
# ------------------------------------------------

st.set_page_config(
    page_title="Intern Performance Predictor",
    page_icon="📊",
    layout="wide"
)


# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()


# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("📊 Intern Performance Predictor")

st.write(
    "Predict intern performance using Machine Learning "
    "and generate an AI-powered performance analysis."
)


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("Intern Information")

task_completion = st.sidebar.slider(
    "Task Completion Rate (%)",
    0,
    100,
    80
)

completion_time = st.sidebar.number_input(
    "Average Task Completion Time (hours)",
    min_value=0.5,
    max_value=20.0,
    value=5.0,
    step=0.5
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


# ------------------------------------------------
# PREDICT
# ------------------------------------------------

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


# ------------------------------------------------
# CATEGORY
# ------------------------------------------------

if prediction >= 75:

    category = "Likely to Excel"
    emoji = "🟢"

elif prediction >= 55:

    category = "Average Performance"
    emoji = "🟡"

else:

    category = "Likely to Struggle"
    emoji = "🔴"


# ------------------------------------------------
# RESULTS
# ------------------------------------------------

st.header("🎯 Prediction Result")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Predicted Performance",
        f"{prediction:.1f}/100"
    )

with col2:

    st.metric(
        "Prediction",
        category
    )


st.progress(int(prediction))

st.subheader(f"{emoji} {category}")


# ------------------------------------------------
# INPUT SUMMARY
# ------------------------------------------------

st.header("📋 Intern Summary")

summary = pd.DataFrame({
    "Metric": [
        "Task Completion",
        "Completion Time",
        "Feedback Rating",
        "Attendance",
        "Tasks Completed",
        "Late Tasks"
    ],

    "Value": [
        f"{task_completion}%",
        f"{completion_time} hours",
        f"{feedback}/5",
        f"{attendance}%",
        tasks_completed,
        late_tasks
    ]
})

st.table(summary)


# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------

st.header("🔍 Important Factors")

features = [
    "Task Completion Rate",
    "Completion Time",
    "Feedback Rating",
    "Attendance",
    "Tasks Completed",
    "Late Tasks"
]

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


# ------------------------------------------------
# GROQ AI
# ------------------------------------------------

st.header("🤖 AI Performance Analysis")

if st.button("Generate AI Analysis"):

    try:

        api_key = st.secrets["GROQ_API_KEY"]

    except Exception:

        api_key = os.getenv("GROQ_API_KEY")


    if not api_key:

        st.error(
            "Groq API key not found. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

    else:

        client = Groq(api_key=api_key)

        prompt = f"""
You are an expert HR analytics assistant.

Analyze this intern performance prediction.

Performance score: {prediction:.1f}/100
Category: {category}

Task completion rate: {task_completion}%
Average completion time: {completion_time} hours
Feedback rating: {feedback}/5
Attendance: {attendance}%
Tasks completed: {tasks_completed}
Late tasks: {late_tasks}

Give:

1. Performance summary
2. Strengths
3. Areas for improvement
4. Three recommendations
5. Manager recommendation

Keep it professional and concise.

Do not say the prediction is certain.
It is only an ML-based estimate.
"""

        try:

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an HR analytics expert."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=700
            )

            answer = response.choices[0].message.content

            st.markdown(answer)

        except Exception as e:

            st.error(f"Groq error: {e}")


# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.divider()

st.caption(
    "Intern Performance Predictor | "
    "Random Forest Regression + Groq AI"
)
