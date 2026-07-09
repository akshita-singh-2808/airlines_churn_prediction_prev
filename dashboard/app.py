import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# CONFIG
st.set_page_config(
    page_title = "Airline Loyalty — Churn Intelligence",
    page_icon  = "✈️",
    layout     = "wide"
)

# LOAD DATA

@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir /"Dataset/Segmented_Customers.csv"
    df = pd.read_csv(data_path)
    return df

df = load_data()

# SIDEBAR
st.sidebar.image("https://img.icons8.com/?size=100&id=48275&format=png&color=000000", width=80)
st.sidebar.title("Loyalty Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "🚨 At-Risk Members",
    "🔍 Member Lookup",
    "🎯 Retention Playbook"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Data:** 2017 Activity | **Model:** XGBoost (Optuna-tuned) | **PR-AUC selected**")

# PAGE 1 — OVERVIEW

if page == "📊 Overview":

    st.title("📊 Loyalty Program - Churn Intelligence Dashboard")
    st.markdown("Real-time view of member health, churn risk, and segment breakdown.")
    st.divider()

    # ── KPI Row ───────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Members",      f"{len(df):,}")
    k2.metric("🔴 High Risk",       f"{(df['Churn_Risk']=='High').sum():,}")
    k3.metric("🟡 Medium Risk",     f"{(df['Churn_Risk']=='Medium').sum():,}")
    k4.metric("🟢 Low Risk",        f"{(df['Churn_Risk']=='Low').sum():,}")
    k5.metric("Avg Churn Prob",     f"{df['Churn_Prob'].mean()*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    # Churn risk pie chart
    with col1:
        st.subheader("Churn Risk Distribution")
        risk_counts = df['Churn_Risk'].value_counts().reset_index()
        risk_counts.columns = ['Risk', 'Count']
        fig = px.pie(
            risk_counts, values='Count', names='Risk',
            color='Risk',
            color_discrete_map={
                'High'  : '#e74c3c',
                'Medium': '#f39c12',
                'Low'   : '#2ecc71'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    # Segment bar chart
    with col2:
        st.subheader("Members by Segment")
        seg_counts = df['Segment_Name'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig = px.bar(
            seg_counts, x='Count', y='Segment',
            orientation='h', color='Segment',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    # CLV by segment boxplot
    with col1:
        st.subheader("CLV Distribution by Segment")
        fig = px.box(
            df, x='Segment_Name', y='CLV',
            color='Segment_Name',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(showlegend=False, xaxis_title="Segment")
        st.plotly_chart(fig, use_container_width=True)

    # Churn probabity by segment
    with col2:
        st.subheader("Avg Churn Probability by Segment")
        avg_churn = df.groupby('Segment_Name')['Churn_Prob'].mean().reset_index()
        avg_churn.columns = ['Segment', 'Avg_Churn_Prob']
        avg_churn = avg_churn.sort_values('Avg_Churn_Prob', ascending=False)
        fig = px.bar(
            avg_churn, x='Segment', y='Avg_Churn_Prob',
            color='Segment',
            color_discrete_sequence=px.colors.qualitative.Set2,
            text=avg_churn['Avg_Churn_Prob'].apply(lambda x: f"{x*100:.1f}%")
        )
        fig.update_layout(showlegend=False, yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Segment x Churn Risk heatmap
    st.subheader("Segment × Churn Risk Breakdown")
    crosstab = pd.crosstab(df['Segment_Name'], df['Churn_Risk'])
    fig = px.imshow(
        crosstab,
        text_auto=True,
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)

# PAGE 2: AT RISK MEMBERS
elif page == "🚨 At-Risk Members":

    st.title("🚨 At-Risk Members")
    st.markdown("Filter and identify members who need immediate attention.")
    st.divider()

    # Filters 
    col1, col2, col3 = st.columns(3)

    with col1:
        risk_filter = st.multiselect(
            "Churn Risk Level",
            options=['High', 'Medium', 'Low'],
            default=['High']
        )
    with col2:
        seg_filter = st.multiselect(
            "Segment",
            options=df['Segment_Name'].dropna().unique().tolist(),
            default=df['Segment_Name'].dropna().unique().tolist()
        )
    with col3:
        clv_min, clv_max = st.slider(
            "CLV Range",
            min_value=int(df['CLV'].min()),
            max_value=int(df['CLV'].max()),
            value=(int(df['CLV'].min()), int(df['CLV'].max()))
        )

    # Apply filters
    filtered = df[
        (df['Churn_Risk'].isin(risk_filter)) &
        (df['Segment_Name'].isin(seg_filter)) &
        (df['CLV'] >= clv_min) &
        (df['CLV'] <= clv_max)
    ].sort_values('Churn_Prob', ascending=False)

    st.metric(f"Customers Matching Criteria", f"{len(filtered):,}")
    st.divider()
    filtered = filtered.copy()
    filtered["Churn_Prob"] = filtered["Churn_Prob"] * 100

    # Display table
    display_cols = [
        'Loyalty Number', 'CLV', 'Churn_Prob', 'Churn_Risk',
        'Segment_Name', 'Total_Flights',
        'Months_Since_Last_Flight', 'Action_Code',
        'Recommended_Action', 'Timing'
    ]

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Churn_Prob": st.column_config.NumberColumn(
            "Churn Probability",
            format="%.2f%%"
            ),
            "CLV": st.column_config.NumberColumn(
                "CLV", format="$%.0f"
            )
        }
    )

    # Download
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button(
        label     = "⬇️ Download List as CSV",
        data      = csv,
        file_name = "at_risk_members.csv",
        mime      = "text/csv"
    )

# PAGE 3 — MEMBER LOOKUP

elif page == "🔍 Member Lookup":

    st.title("🔍 Member Lookup")
    st.markdown("Enter a Loyalty Number to see full profile and recommended action.")
    st.divider()

    loyalty_input = st.number_input(
        "Enter Loyalty Number",
        min_value=0, step=1,
        value= 480934
    )

    if loyalty_input > 0:
        member = df[df['Loyalty Number'] == loyalty_input]

        if len(member) == 0:
            st.error(f"❌ Member {loyalty_input} not found.")
        else:
            member = member.iloc[0]

            # Risk color
            risk_color = {
                'High'  : '🔴',
                'Medium': '🟡',
                'Low'   : '🟢'
            }.get(member['Churn_Risk'], '⚪')

            st.subheader(f"Member with Loyalty Number- {int(member['Loyalty Number'])} : {risk_color} {member['Churn_Risk']} Risk")
            st.divider()

            # Profile metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CLV",              f"${member['CLV']:,.0f}")
            col2.metric("Churn Probability", f"{member['Churn_Prob']*100:.1f}%")
            col3.metric("Total Flights",    int(member['Total_Flights']))
            col4.metric("Months Inactive",  int(member['Months_Since_Last_Flight']))

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Segment",          member['Segment_Name'])
            col2.metric("Tenure (years)",   int(member['Tenure_At_Cutoff']))
            col3.metric("Activity Rate",    f"{member['Activity_Rate']*100:.0f}%")
            col4.metric("Redemption Ratio", f"{member['Redemption_Ratio']*100:.1f}%")

            st.divider()

            # Seasonal flights bar chart
            st.subheader("Seasonal Flight Pattern")
            seasonal = pd.DataFrame({
                'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
                'Flights': [
                    member['Flights_Q1'], member['Flights_Q2'],
                    member['Flights_Q3'], member['Flights_Q4']
                ]
            })
            fig = px.bar(
                seasonal, x='Quarter', y='Flights',
                color='Quarter',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Recommended action card
            st.subheader("💡 Recommended Action")

            action_code = member['Action_Code']

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Action Code:** `{action_code}`")
                st.markdown(f"**What to do:** {member['Recommended_Action']}")
                st.markdown(f"**When:** {member['Timing']}")
            with col2:
                if action_code == 'VIP WIN-BACK':
                    st.error("🔴 Urgent- personal outreach required within 7 days")
                elif action_code in ['WIN-BACK', 'RE-ENGAGE', 'VALUE SAVE']:
                    st.warning("🟡 Action needed within 14 days")
                elif action_code == 'PROACTIVE SAVE':
                    st.info("🔵 Schedule for start of next quarter")
                else:
                    st.success("🟢 Low urgency- include in regular nurture cycle")

# PAGE 4 — RETENTION PLAYBOOK

elif page == "🎯 Retention Playbook":

    st.title("🎯 Retention Playbook")
    st.markdown("Specific actions for each segment- ready to hand to operations.")
    st.divider()

    # NOTE: "who"/stats text below is now generated from the live dataframe
    # further down (col3 metrics), so the numbers you see always match the
    # actual data, not a fixed example.
    playbook = {
        "New At-Risk": {
            "icon"   : "🔴",
            "who"    : "Recently enrolled members (≤1 year tenure) with elevated churn risk",
            "action" : "Win-back email campaign with time-limited offer",
            "offer"  : "2x points on next flight booked within 30 days",
            "timing" : "Within 7 days of identification",
            "owner"  : "CRM Team",
            "why"    : "Short tenure means low switching cost — act before habit forms elsewhere"
        },
        "Dormant Members": {
            "icon"   : "🟠",
            "who"    : "Members with zero flights in the last year, still enrolled",
            "action" : "Personalised re-engagement push notification",
            "offer"  : "Discounted fare on their historically top 3 routes",
            "timing" : "Immediate - before they lapse permanently",
            "owner"  : "Marketing Team",
            "why"    : "Still enrolled but disengaged-a relevant offer can reactivate them"
        },
        "High Value Vulnerables": {
            "icon"   : "🟡",
            "who"    : "Top-tier CLV members with elevated churn risk",
            "action" : "Personal account manager call + exclusive retention offer",
            "offer"  : "Upgrade voucher + 5,000 bonus points + priority boarding for 3 months",
            "timing" : "Within 7 days - highest financial risk if lost",
            "owner"  : "Account Management Team",
            "why"    : "Losing one High Value Vulnerable costs more than retaining many Dormant members"
        },
        "Stable Regulars": {
            "icon"   : "🟢",
            "who"    : "Largest segment, steady flyers with low churn risk",
            "action" : "Loyalty deepening - tier upgrade challenge",
            "offer"  : "Fly 3 more times this quarter to unlock next card tier benefits",
            "timing" : "Start of each quarter",
            "owner"  : "Loyalty Team",
            "why"    : "Already engaged - small push converts them to higher lifetime value"
        }
    }

    for segment, details in playbook.items():
        with st.expander(f"{details['icon']} {segment}", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**Who:** {details['who']}")
                st.markdown(f"**Action:** {details['action']}")
                st.markdown(f"**Offer:** {details['offer']}")
                st.markdown(f"**Why now:** {details['why']}")

            with col2:
                st.markdown(f"**Timing:** {details['timing']}")
                st.markdown(f"**Owner:** {details['owner']}")

            with col3:
                seg_data  = df[df['Segment_Name'] == segment]
                total     = len(seg_data)
                high_risk = (seg_data['Churn_Risk'] == 'High').sum()
                avg_clv   = seg_data['CLV'].mean()

                st.metric("Members",    f"{total:,}")
                st.metric("High Risk",  f"{high_risk:,}")
                st.metric("Avg CLV",    f"${avg_clv:,.0f}")
