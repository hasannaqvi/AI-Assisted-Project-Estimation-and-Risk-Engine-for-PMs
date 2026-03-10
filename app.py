import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from z3 import Optimize, Int, Or, Sum

PROJECT_TYPE_CONFIG = {
    "Small":  {"variance": 0.15, "risk_multiplier": 0.7}, #duration variance low, risk multiplier low
    "Medium": {"variance": 0.30, "risk_multiplier": 1.0},
    "Large":  {"variance": 0.50, "risk_multiplier": 1.3}
}


# ====================================
# SECTION 1: CPM FUNCTION
# ====================================
def compute_cpm(tasks, durations):
    ES, EF = {}, {}

    def forward(task):
        if task in ES:
            return
        if not tasks[task]["deps"]:
            ES[task] = 0
        else:
            ES[task] = max(EF[d] for d in tasks[task]["deps"])
        EF[task] = ES[task] + durations[task]

    for t in tasks:
        forward(t)

    project_duration = max(EF.values())

    LS, LF = {}, {}
    for t in tasks:
        LF[t] = project_duration

    for t in reversed(list(tasks.keys())):
        successors = [s for s in tasks if t in tasks[s]["deps"]]
        if successors:
            LF[t] = min(LS[s] for s in successors)
        LS[t] = LF[t] - durations[t]

    critical_path = [
        t for t in tasks if abs(LS[t] - ES[t]) < 1e-6
    ]

    return project_duration, critical_path


# ====================================
# SECTION 2: MONTE CARLO SIMULATION
# ====================================
def run_monte_carlo(tasks, n_sim=500):
    durations = []
    critical_counts = {t: 0 for t in tasks}

    for _ in range(n_sim):
        sampled = {
            t: np.random.triangular(
                tasks[t]["min"],
                tasks[t]["mean"],
                tasks[t]["max"]
            )
            for t in tasks
        }

        total_duration, critical_path = compute_cpm(tasks, sampled)
        durations.append(total_duration)

        for t in critical_path:
            critical_counts[t] += 1

    return durations, critical_counts


# ====================================
# SECTION 3: INTERVENTION INDEX
# ====================================
def compute_intervention_index(critical_counts, tasks, n_sim):
    rework_probability = {                                       #Dynamic rework probability based on task
        t: tasks[t]["rework_prob"]
        for t in tasks
    }

    criticality = {
        t: critical_counts[t] / n_sim
        for t in tasks
    }

    intervention = {
        t: criticality[t] * rework_probability[t]
        for t in tasks
    }

    return intervention


# ====================================
# SECTION 4: Z3 OPTIMIZATION
# ====================================
# Define mitigation options
MITIGATIONS = {
    "REQ": {"risk_reduction": 0.20, "cost": 40},
    "API": {"risk_reduction": 0.30, "cost": 80},
    "FE":  {"risk_reduction": 0.20, "cost": 60},
    "INT": {"risk_reduction": 0.35, "cost": 70},
    "QA":  {"risk_reduction": 0.15, "cost": 50}
}

def optimize_mitigation(intervention_index, budget):
    opt = Optimize()
    vars = {t: Int(t) for t in MITIGATIONS}

    for v in vars.values():
        opt.add(Or(v == 0, v == 1))

    opt.add(
        Sum([vars[t] * MITIGATIONS[t]["cost"] for t in MITIGATIONS]) <= budget
    )

    opt.maximize(
        Sum([
            vars[t] * MITIGATIONS[t]["risk_reduction"] * intervention_index[t]
            for t in MITIGATIONS
        ])
    )

    opt.check()
    model = opt.model()

    return [t for t in MITIGATIONS if model[vars[t]].as_long() == 1]


# ====================================
# SECTION 5: APPLY MITIGATIONS
# ====================================
def apply_mitigations(tasks, selected):
    mitigations = {
        "REQ": 0.20,
        "API": 0.30,
        "FE":  0.20,
        "INT": 0.35,
        "QA":  0.15
    }

    new_tasks = {}
    for t, params in tasks.items():
        params = params.copy()
        if t in selected:
            reduction = mitigations[t]
            params["max"] = params["mean"] + (params["max"] - params["mean"]) * (1 - reduction)
        new_tasks[t] = params

    return new_tasks

#------------------Dynamically Building Tasks based on Project Type & Deadline------------------
BASE_REWORK_RISK = {
    "REQ": 0.30,
    "ARCH": 0.20,
    "API": 0.45,
    "FE": 0.35,
    "INT": 0.50,
    "QA": 0.25,
    "DEP": 0.10
}

def build_tasks(task_hours, project_type, go_live_days):
    config = PROJECT_TYPE_CONFIG[project_type]

    mean_durations = {
        t: h / 8 for t, h in task_hours.items()
    }

    planned_duration = sum(mean_durations.values())
    pressure_factor = min(max(planned_duration / go_live_days, 0.7), 1.5)

    tasks = {}

    for t, mean in mean_durations.items():
        variance = config["variance"]
        min_dur = mean * (1 - variance)
        max_dur = mean * (1 + variance)

        rework_prob = (
            BASE_REWORK_RISK[t]
            * config["risk_multiplier"]
            * pressure_factor
        )
        rework_prob = min(rework_prob, 0.85)

        tasks[t] = {
            "mean": mean,
            "min": min_dur,
            "max": max_dur,
            "deps": task_dependencies[t],
            "rework_prob": rework_prob
        }

    return tasks, pressure_factor



# ====================================
# PAGE CONFIG & SIDEBAR
# ====================================
st.set_page_config(page_title="Software Project Risk Planner", layout="wide")

st.title("📊 Software Project Risk & Delivery Planner")
st.caption("A lightweight decision-support tool for PMs and Engineering Leaders")

#************************************
# Sidebar: PM Inputs section
#************************************
st.sidebar.header("Project Inputs")
#     NEW PM INPUTS (Project Type, Go-Live Deadline)

#--------------------------------------- User Inputs for Inputs Req, Dev, QA

st.sidebar.subheader("Task Effort Estimates (hours)")

task_hours = {
    "REQ": st.sidebar.number_input(
        "Requirements Analysis",
        min_value=8, max_value=200, value=32, step=4,
        help="Estimated effort for requirements gathering and clarification."
    ),
    "ARCH": st.sidebar.number_input(
        "Architecture & Design",
        min_value=8, max_value=200, value=24, step=4,
        help="Effort required for system architecture and technical design."
    ),
    "API": st.sidebar.number_input(
        "Backend / API Development",
        min_value=8, max_value=400, value=64, step=8,
        help="Development effort for backend services and APIs."
    ),
    "FE": st.sidebar.number_input(
        "Frontend Development",
        min_value=8, max_value=400, value=56, step=8,
        help="UI and frontend implementation effort."
    ),
    "INT": st.sidebar.number_input(
        "Integration & System Testing",
        min_value=8, max_value=300, value=40, step=4,
        help="Effort required to integrate components and resolve integration issues."
    ),
    "QA": st.sidebar.number_input(
        "Quality Assurance",
        min_value=8, max_value=300, value=32, step=4,
        help="Testing, validation, and defect resolution effort."
    ),
    "DEP": st.sidebar.number_input(
        "Deployment & Release",
        min_value=4, max_value=100, value=16, step=2,
        help="Final deployment, release preparation, and go-live activities."
    )
}
#---------------------------------------End of User Inputs for Inputs Req, Dev, QA
project_type = st.sidebar.selectbox(
    "Project Type",
    ["Small", "Medium", "Large"],
    help="Represents overall project complexity and coordination overhead."
)

go_live_days = st.sidebar.number_input(
    "Go-Live Deadline (days)",
    min_value=10,
    max_value=365,
    value=90,
    step=5,
    help="Target delivery deadline used to assess schedule pressure and delivery risk."
)
#-------------------------------
BUDGET = st.sidebar.slider(     #// added PM terminology for budget
    "Mitigation Budget (effort units)",
    min_value=50,
    max_value=250,
    value=120,
    step=10,
    help=(
        "Represents how much managerial effort or resources can be invested "
        "to proactively reduce project risk (e.g., spikes, tooling, early validation). "
        "Higher values allow more or stronger risk mitigations."
    )
)

N_SIM = st.sidebar.slider(  #// added PM terminology for monte carlo simulations
    "Monte Carlo Simulations",
    min_value=200,
    max_value=1000,
    value=500,
    step=100,
    help=(
        "Number of simulated project scenarios. "
        "Higher values produce more stable and reliable risk estimates, "
        "at the cost of longer computation time."
    )
)


confidence = st.sidebar.selectbox(   #// added PM terminology for deadline commitments
    "Delivery Confidence Level",
    ["P50 (Likely)", "P90 (Conservative)"],
    help=(
        "Defines how conservative the delivery commitment should be. "
        "P50 reflects a likely outcome, while P90 represents a safer date "
        "with only a 10% risk of overrun."
    )
)


# Static project template (MVP)
#task_hours = {
#    "REQ": 32,
#    "ARCH": 24,
#    "API": 64,
#    "FE": 56,
#    "INT": 40,
#    "QA": 32,
#    "DEP": 16
#}

task_dependencies = {
    "REQ": [],
    "ARCH": ["REQ"],
    "API": ["ARCH"],
    "FE": ["ARCH"],
    "INT": ["API", "FE"],
    "QA": ["INT"],
    "DEP": ["QA"]
}


# ====================================
# RUN MODEL & DISPLAY RESULTS
# ====================================

tasks, pressure_factor = build_tasks(     #Dynamic task building based on PM inputs
    task_hours,
    project_type,
    go_live_days
)
durations, critical_counts = run_monte_carlo(tasks, N_SIM)

p50 = np.percentile(durations, 50)
p90 = np.percentile(durations, 90)

intervention_index = compute_intervention_index(
    critical_counts, tasks, N_SIM
)

selected = optimize_mitigation(intervention_index, BUDGET)

mitigated_tasks = apply_mitigations(tasks, selected)
post_durations, _ = run_monte_carlo(mitigated_tasks, N_SIM)
post_p90 = np.percentile(post_durations, 90)

# ====================================
# DELIVERY FORECAST METRICS
# ====================================
st.subheader("📅 Delivery Forecast")  #//added caption below for PM/Management interpretation
st.caption(
    "P50 indicates a likely completion date, while P90 represents a conservative "
    "commitment suitable for executive or external stakeholders."
)


c1, c2, c3 = st.columns(3)

c1.metric("P50 (Likely)", f"{p50:.1f} days")
c2.metric("P90 (Safe)", f"{p90:.1f} days")
c3.metric("P90 After Mitigation", f"{post_p90:.1f} days")
st.subheader("⚙️ Project Context Indicators")

c4, c5 = st.columns(2)

c4.metric(
    "Project Size",
    project_type,
    help="Overall complexity level affecting uncertainty and coordination risk."
)

c5.metric(
    "Schedule Pressure",
    f"{pressure_factor:.2f}",
    help=(
        "Ratio between planned effort and go-live deadline. "
        "Values above 1 indicate high delivery pressure."
    )
)

# ====================================
# CHARTS SECTION (COLLAPSIBLE)
# ====================================
with st.expander("📊 View Charts", expanded=False):
    # ====================================
    # DELIVERY RISK DISTRIBUTION CHART
    # ====================================
    st.subheader("📈 Delivery Risk Distribution")

    fig, ax = plt.subplots(figsize=(4, 2))

    ax.hist(durations, bins=30, alpha=0.6, label="Before Mitigation", edgecolor='black')
    ax.hist(post_durations, bins=30, alpha=0.6, label="After Mitigation", edgecolor='black')

    ax.axvline(p90, linestyle="--", label="P90 Before", color='orange')
    ax.axvline(post_p90, linestyle="--", label="P90 After", color='green')

    ax.set_xlabel("Project Duration (days)", fontsize='small')
    ax.set_ylabel("Frequency", fontsize='small')
    ax.legend(fontsize='xx-small')
    ax.tick_params(axis='both', which='major', labelsize='x-small')

    st.pyplot(fig)

    # ====================================
    # RECOMMENDED INTERVENTIONS
    # ====================================
    st.subheader("🔥 Intervention Priority Index")

    df_intervention = (
        pd.DataFrame.from_dict(intervention_index, orient="index", columns=["Score"])
        .sort_values("Score", ascending=False)
    )

    fig2, ax2 = plt.subplots(figsize=(3, 2))

    ax2.bar(df_intervention.index, df_intervention["Score"])
    ax2.set_ylabel("Intervention Priority Score", fontsize='small')
    ax2.set_xlabel("Task", fontsize='small')
    ax2.tick_params(axis='both', which='major', labelsize='x-small')

    st.pyplot(fig2)

st.subheader("💰 Recommended Mitigations")
st.write(f"**Selected interventions:** {', '.join(selected) if selected else 'None within budget'}")
budget_used = sum([MITIGATIONS.get(t, {}).get('cost', 0) for t in selected])
st.write(f"**Budget used:** {budget_used} / {BUDGET}")

st.success("""
With current budget, invest in:
- High-priority interventions based on criticality
- Early risk reduction in foundational tasks

Expected outcome:
- Reduced tail risk
- Higher delivery confidence
""")
