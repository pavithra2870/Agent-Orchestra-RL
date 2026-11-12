"""
Streamlit frontend for Agent Orchestra RL Project
Review 2 Demo - Clean, professional interface for showcasing RL results
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Agent Orchestra RL",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional styling with Inter font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Main background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Typography - Inter font */
    html, body, [class*="css"], h1, h2, h3, h4, h5, h6, p, div, span {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a;
        font-weight: 700;
        letter-spacing: -0.8px;
        font-size: 2.5rem !important;
    }
    
    h2 {
        color: #2c2c2c;
        font-weight: 600;
        letter-spacing: -0.5px;
        font-size: 1.75rem !important;
    }
    
    h3 {
        color: #3c3c3c;
        font-weight: 600;
        letter-spacing: -0.3px;
        font-size: 1.35rem !important;
    }
    
    h4 {
        color: #4c4c4c;
        font-weight: 500;
        font-size: 1.1rem !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #1a1a1a;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        color: #666;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tables */
    .dataframe {
        font-size: 13px !important;
        font-weight: 400;
    }
    
    .dataframe thead th {
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        background-color: #f8f9fa !important;
        color: #666 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: white;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 54px;
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 0px 28px;
        font-weight: 500;
        font-size: 15px;
        border: none;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a73e8;
        color: white !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 14px 36px;
        font-weight: 600;
        font-size: 15px;
        letter-spacing: 0.3px;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #1557b0;
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.25);
        transform: translateY(-1px);
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stMultiSelect {
        font-weight: 500;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e8e8e8;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border: none;
        font-size: 14px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #1a73e8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Agent Orchestra RL")
    st.markdown("**Reinforcement Learning for Tool-Use Orchestration**")
with col2:
    st.markdown("<div style='text-align: right; padding-top: 20px;'><span style='background-color: #e8f4fd; padding: 8px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; color: #1a73e8;'>Review 2 Demo</span></div>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## Navigation")
    st.markdown("Explore pre-computed results or run live simulations")
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    Multi-phase RL project comparing:
    - **Tabular RL**: Q-Learning, SARSA
    - **Deep RL**: DQN, PPO
    - **Offline RL**: CQL
    - **Multi-Agent**: Coordination
    """)
    
    st.markdown("---")
    st.markdown("### Key Highlights")

    # Derived from results/final/results_table.csv so the sidebar never
    # drifts from whatever the most recent evaluation run produced.
    _final_csv = RESULTS_DIR / "final" / "results_table.csv"
    if _final_csv.exists():
        _final_df = pd.read_csv(_final_csv)
        _best_regret_row = _final_df.loc[_final_df["regret_vs_oracle"].idxmin()]
        st.info(
            f"**{_best_regret_row['agent'].upper()}** achieves the lowest regret "
            f"({_best_regret_row['regret_vs_oracle']:.2f}) vs. the oracle solver"
        )

        _sample_eff = _final_df.dropna(subset=["sample_efficiency_episodes"])
        _sample_eff = _sample_eff[_sample_eff["sample_efficiency_episodes"] > 0]
        if len(_sample_eff) >= 2:
            _most_eff = _sample_eff.loc[_sample_eff["sample_efficiency_episodes"].idxmin()]
            _least_eff = _sample_eff.loc[_sample_eff["sample_efficiency_episodes"].idxmax()]
            if _most_eff["agent"] != _least_eff["agent"]:
                _ratio = _least_eff["sample_efficiency_episodes"] / _most_eff["sample_efficiency_episodes"]
                st.success(
                    f"**{_most_eff['agent'].upper()}** is {_ratio:.1f}x more sample-efficient "
                    f"than **{_least_eff['agent'].upper()}**"
                )
    else:
        st.caption("Run the final evaluation harness to populate key highlights.")

# Main tabs
tab1, tab2, tab3 = st.tabs(["Pre-computed Results", "Live Simulation", "Generalization Test"])

# ==================== TAB 1: Pre-computed Results ====================
with tab1:
    st.markdown("## Pre-computed Results")
    st.markdown("Comprehensive view of all training phases and evaluation metrics")
    
    # Phase selector
    phase_options = {
        "Final Summary": "final",
        "Phase 1: Tabular RL": "phase1",
        "Phase 2: Deep RL (DQN)": "phase2", 
        "Phase 3: PPO & CQL": "phase3",
        "Phase 4: Multi-Agent": "phase4"
    }
    
    selected_phase = st.selectbox("**Select Phase**", list(phase_options.keys()), index=0)
    phase_dir = RESULTS_DIR / phase_options[selected_phase]
    
    st.markdown("---")
    
    # Display based on selected phase
    if selected_phase == "Final Summary":
        st.markdown("### Cross-Method Comparison")
        
        # Load final results
        final_csv = RESULTS_DIR / "final" / "results_table.csv"
        if final_csv.exists():
            df = pd.read_csv(final_csv)
            
            # Key metrics in cards
            col1, col2, col3, col4 = st.columns(4)
            
            best_success = df.loc[df['success_rate'].idxmax()]
            best_reward = df.loc[df['avg_reward'].idxmax()]
            best_efficiency = df.loc[df['tool_call_efficiency'].idxmax()]
            best_regret = df.loc[df['regret_vs_oracle'].idxmin()]
            
            with col1:
                st.metric(
                    "Best Success Rate",
                    f"{best_success['success_rate']*100:.1f}%",
                    delta=best_success['agent'],
                    delta_color="off"
                )
            
            with col2:
                st.metric(
                    "Best Avg Reward",
                    f"{best_reward['avg_reward']:.2f}",
                    delta=best_reward['agent'],
                    delta_color="off"
                )
            
            with col3:
                st.metric(
                    "Best Efficiency",
                    f"{best_efficiency['tool_call_efficiency']:.3f}",
                    delta=best_efficiency['agent'],
                    delta_color="off"
                )
            
            with col4:
                st.metric(
                    "Lowest Regret",
                    f"{best_regret['regret_vs_oracle']:.3f}",
                    delta=best_regret['agent'],
                    delta_color="off"
                )
            
            st.markdown("---")
            
            # Results table with styling
            st.markdown("#### Detailed Comparison")
            
            # Format the dataframe
            display_df = df.copy()
            display_df['success_rate'] = (display_df['success_rate'] * 100).round(1).astype(str) + '%'
            display_df['avg_reward'] = display_df['avg_reward'].round(2)
            display_df['avg_cost_per_task'] = display_df['avg_cost_per_task'].round(2)
            display_df['avg_latency_per_task'] = display_df['avg_latency_per_task'].round(2)
            display_df['tool_call_efficiency'] = display_df['tool_call_efficiency'].round(3)
            display_df['regret_vs_oracle'] = display_df['regret_vs_oracle'].round(3)
            
            st.dataframe(
                display_df,
                width='stretch',
                hide_index=True,
            )
            
            # Visualizations
            st.markdown("---")
            st.markdown("#### Visual Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Success rate comparison
                fig1 = go.Figure(data=[
                    go.Bar(
                        x=df['agent'],
                        y=df['success_rate'] * 100,
                        marker_color='#1a73e8',
                        text=df['success_rate'].apply(lambda x: f"{x*100:.1f}%"),
                        textposition='auto',
                        textfont=dict(size=12, weight='bold')
                    )
                ])
                fig1.update_layout(
                    title="Success Rate by Agent",
                    title_font=dict(size=16, weight='bold'),
                    xaxis_title="Agent",
                    yaxis_title="Success Rate (%)",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Inter, sans-serif", size=12),
                    height=400
                )
                fig1.update_xaxes(showgrid=False)
                fig1.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
                st.plotly_chart(fig1, width='stretch')
            
            with col2:
                # Regret comparison
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=df['agent'],
                        y=df['regret_vs_oracle'],
                        marker_color='#ea4335',
                        text=df['regret_vs_oracle'].apply(lambda x: f"{x:.2f}"),
                        textposition='auto',
                        textfont=dict(size=12, weight='bold')
                    )
                ])
                fig2.update_layout(
                    title="Regret vs Oracle",
                    title_font=dict(size=16, weight='bold'),
                    xaxis_title="Agent",
                    yaxis_title="Regret",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Inter, sans-serif", size=12),
                    height=400
                )
                fig2.update_xaxes(showgrid=False)
                fig2.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
                st.plotly_chart(fig2, width='stretch')
            
            # Summary grid image
            summary_img = RESULTS_DIR / "final" / "summary_grid.png"
            if summary_img.exists():
                st.markdown("---")
                st.markdown("#### Summary Grid")
                st.image(str(summary_img), width='stretch')
        
        # Load findings
        findings_path = RESULTS_DIR / "FINDINGS.md"
        if findings_path.exists():
            st.markdown("---")
            st.markdown("### Key Findings")
            with st.expander("View Detailed Findings", expanded=False):
                findings = findings_path.read_text()
                st.markdown(findings)
    
    else:
        # Phase-specific results
        st.markdown(f"### {selected_phase}")
        
        logs_dir = phase_dir / "logs"
        plots_dir = phase_dir / "plots"
        
        # CSV File Viewer
        if logs_dir.exists():
            csv_files = list(logs_dir.glob("*.csv"))
            if csv_files:
                st.markdown("#### CSV Data Files")
                csv_tabs = st.tabs([f.stem.replace('_', ' ').title() for f in csv_files])
                
                for csv_tab, csv_file in zip(csv_tabs, csv_files):
                    with csv_tab:
                        df_csv = pd.read_csv(csv_file)
                        st.markdown(f"**{csv_file.name}** ({len(df_csv)} rows)")
                        
                        # Show first few rows
                        st.dataframe(df_csv.head(100), width='stretch', hide_index=True)
                        
                        # Download button
                        csv_data = df_csv.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download {csv_file.name}",
                            data=csv_data,
                            file_name=csv_file.name,
                            mime='text/csv',
                        )
                
                st.markdown("---")
        
        # Load summary
        if (logs_dir / "summary.csv").exists():
            summary_df = pd.read_csv(logs_dir / "summary.csv")
            
            st.markdown("#### Performance Summary")
            
            # Display metrics in columns
            num_agents = len(summary_df)
            cols = st.columns(min(num_agents, 4))
            
            for idx, row in summary_df.iterrows():
                with cols[idx % 4]:
                    st.markdown(f"**{row['agent'].upper()}**")
                    st.metric("Success Rate", f"{row['success_rate']*100:.1f}%")
                    st.metric("Avg Reward", f"{row['avg_reward']:.2f}")
                    if 'regret_vs_oracle' in row:
                        st.metric("Regret", f"{row['regret_vs_oracle']:.3f}")
            
            st.markdown("---")
            
            # Full table
            st.markdown("#### Detailed Metrics")
            display_summary = summary_df.copy()
            display_summary['success_rate'] = (display_summary['success_rate'] * 100).round(1).astype(str) + '%'
            for col in ['avg_reward', 'avg_cost', 'avg_latency', 'avg_steps', 'regret_vs_oracle']:
                if col in display_summary.columns:
                    display_summary[col] = display_summary[col].round(3)
            
            st.dataframe(display_summary, width='stretch', hide_index=True)
        
        # Display plots
        if plots_dir.exists():
            st.markdown("---")
            st.markdown("#### Learning Curves & Visualizations")
            
            plot_files = list(plots_dir.glob("*.png"))
            if plot_files:
                # Show plots in grid
                cols_per_row = 2
                for i in range(0, len(plot_files), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, plot_file in enumerate(plot_files[i:i+cols_per_row]):
                        with cols[j]:
                            st.image(str(plot_file), caption=plot_file.stem.replace('_', ' ').title())
            else:
                st.info("No plots available for this phase")

# ==================== TAB 2: Live Simulation ====================
with tab2:
    st.markdown("## Live Simulation Runner")
    st.markdown("Run evaluation episodes with trained models and compare performance")
    
    # Task metadata is derived live from src/envs/tasks.py's TASK_REGISTRY
    # (the actual source of truth the environment uses) instead of being
    # duplicated here by hand, so this panel can't drift out of sync with it.
    from src.envs.tasks import TASK_REGISTRY, NUM_TASKS
    from src.envs.tools import TOOL_NAMES

    task_templates = {t.task_id: t.name for t in TASK_REGISTRY}

    # Free-text flavor descriptions aren't derivable from the registry, so
    # they're kept as hand-written copy, looked up by task name (not id) so
    # a registry reorder can't silently mismatch a task to the wrong text.
    _task_description_by_name = {
        "check_order_status": "Check order status via internal DB or external search API",
        "resolve_billing_issue": "Resolve billing dispute - verify identity, check account, optionally issue refund",
        "process_refund": "Process customer refund after identity verification",
        "update_contact_email": "Update customer's email address after verification",
        "escalate_complaint": "Escalate customer complaint to human agent (with or without verification)",
        "search_product_info": "Search for product information in local catalog or external API",
        "verify_and_notify": "Verify customer identity and send notification email",
        "fraud_check_and_block": "Investigate fraud, verify identity, check records, escalate to human",
        "password_reset": "Reset customer password - verify identity and send reset link",
        "full_account_review": "Complete account review - verify, check DB, search external, escalate",
    }
    task_descriptions = {
        t.task_id: _task_description_by_name.get(t.name, t.name.replace("_", " ").title())
        for t in TASK_REGISTRY
    }

    def _difficulty_label(t) -> str:
        min_len, max_len = t.get_min_path_length(), t.get_max_path_length()
        steps_label = f"{min_len}-{max_len}" if min_len != max_len else f"{min_len}"
        if t.max_steps <= 5 and max_len <= 2:
            tier = "Easy"
        elif t.max_steps <= 6:
            tier = "Medium"
        else:
            tier = "Hard"
        return f"{tier} ({steps_label} steps, budget {t.max_budget:.1f})"

    task_difficulty = {t.task_id: _difficulty_label(t) for t in TASK_REGISTRY}

    # Required tools per task (what agents see in metadata) - union of every
    # success path's tools, exactly as ToolOrchestrationEnv computes it.
    task_required_tools = {
        t.task_id: [
            name for name, flag in zip(TOOL_NAMES, t.get_required_tools(TOOL_NAMES)) if flag == 1.0
        ]
        for t in TASK_REGISTRY
    }
    
    # Available algorithms
    available_algorithms = [
        "rule_based",
        "dqn",
        "ppo",
        "cql",
        "q_learning",
        "sarsa",
    ]
    
    algorithm_info = {
        "rule_based": "Hand-coded heuristic (baseline)",
        "dqn": "Deep Q-Network (Phase 2)",
        "ppo": "Proximal Policy Optimization (Phase 3)",
        "cql": "Conservative Q-Learning - Offline RL (Phase 3)",
        "q_learning": "Tabular Q-Learning (Phase 1)",
        "sarsa": "Tabular SARSA (Phase 1)",
    }
    
    col_config, col_hyperparam = st.columns([1, 1])
    
    with col_config:
        st.markdown("### Configuration")
        
        # Task metadata toggle (NEW)
        st.markdown("####  Task Generalization Features")
        include_metadata = st.checkbox(
            "**Enable Task Metadata Features**",
            value=True,  # DEFAULT TO TRUE since models are now retrained
            help="Includes semantic task information (required tools, path lengths, constraints) in observations. YOUR RETRAINED MODELS EXPECT THIS TO BE ENABLED."
        )
        
        if include_metadata:
            st.success("✓ Agents receive rich task metadata (34-dim observations)")
            st.caption("✓ Compatible with your retrained models | Enables generalization")
        else:
            st.error("⚠️ METADATA DISABLED - Your retrained models expect 34-dim observations!")
            st.caption("This will cause 0% success. Enable the checkbox above.")
        
        st.markdown("---")
        
        # Task selection
        selected_task_id = st.selectbox(
            "**Task Template**",
            options=list(task_templates.keys()),
            index=1,  # Default to task 1 (medium difficulty)
            format_func=lambda x: f"Task {x}: {task_templates[x]}"
        )
        
        # Show detailed task information
        with st.expander("📋 Task Details", expanded=True):
            st.markdown(f"**Name**: {task_templates[selected_task_id]}")
            st.markdown(f"**Description**: {task_descriptions[selected_task_id]}")
            st.markdown(f"**Difficulty**: {task_difficulty[selected_task_id]}")
            st.markdown(f"**Required Tools** (metadata agents receive):")
            tools_list = ", ".join(f"`{tool}`" for tool in task_required_tools[selected_task_id])
            st.markdown(f"  - {tools_list}")
            
            if include_metadata:
                st.success("✓ Agents will receive this tool information in observations")
            else:
                st.warning("⚠ Agents will NOT see which tools are required (legacy mode)")
        
        st.caption(f"**Tip**: Tasks 7 & 9 are hardest (3-4 steps). Task 1 has multiple solution paths.")
        
        # Algorithm selection
        selected_algorithms = st.multiselect(
            "**Algorithms to Compare**",
            options=available_algorithms,
            default=["rule_based", "dqn", "ppo", "cql"],
            format_func=lambda x: algorithm_info.get(x, x)
        )
        
        # Number of episodes
        num_episodes = st.slider("**Number of Episodes**", 10, 200, 50, 10)
        st.caption(f"**Recommended**: 30-50 eps for quick demo (~10-30 sec), 100-150 eps for robust comparison (~1-2 min)")
        if selected_algorithms:
            est_time_min = len(selected_algorithms) * num_episodes * 0.5
            est_time_max = len(selected_algorithms) * num_episodes * 1.0
            st.caption(f"**Estimated time**: {est_time_min:.0f}-{est_time_max:.0f} seconds")
        
        # Show what agents actually see (NEW)
        if include_metadata:
            st.markdown("---")
            with st.expander("👁️ What Agents See (Observation Preview)", expanded=False):
                from src.envs.tasks import get_task_metadata_stats

                _selected_task = TASK_REGISTRY[selected_task_id]
                _stats = get_task_metadata_stats()
                required_binary = [1 if t in task_required_tools[selected_task_id] else 0 for t in TOOL_NAMES]
                min_path_norm = _selected_task.get_min_path_length() / _stats['max_max_path_length']
                num_paths_norm = _selected_task.get_num_success_paths() / _stats['max_num_paths']
                budget_norm = _selected_task.max_budget / _stats['max_budget_global']
                steps_norm = _selected_task.max_steps / _stats['max_steps_global']

                obs_dim = NUM_TASKS + 2 * len(TOOL_NAMES) + 2 + len(TOOL_NAMES) + 4
                st.markdown(f"**Observation vector ({obs_dim} dimensions):**")

                st.code(f"""Task one-hot:     [0,0,...,1,...,0]  ({NUM_TASKS} dim) - identifies task {selected_task_id}
Attempted mask:   [0,0,0,0,0,0]      ({len(TOOL_NAMES)} dim)  - which tools tried
Succeeded mask:   [0,0,0,0,0,0]      ({len(TOOL_NAMES)} dim)  - which tools succeeded
Budget remaining: [1.0]              (1 dim)  - normalized budget (resets to 1.0 at episode start)
Steps remaining:  [1.0]              (1 dim)  - normalized steps (resets to 1.0 at episode start)

METADATA (agents use this to generalize):
Required tools:   {required_binary}  ({len(TOOL_NAMES)} dim)  - which tools needed
Min path length:  [{min_path_norm:.2f}]         (1 dim)  - shortest success path, normalized
Num paths:        [{num_paths_norm:.2f}]         (1 dim)  - alternative success paths, normalized
Task max budget:  [{budget_norm:.2f}]         (1 dim)  - this task's budget vs. hardest task
Task max steps:   [{steps_norm:.2f}]         (1 dim)  - this task's step limit vs. hardest task""", language="text")
                st.caption("✅ This metadata enables agents to reason about unseen tasks!")
        
    
    with col_hyperparam:
        st.markdown("### Hyperparameters")
        
        enable_hyperparam = st.checkbox("Enable Hyperparameter Tuning", value=False)
        
        if enable_hyperparam:
            st.markdown("**Environment Parameters**")
            success_reward = st.number_input("Success Reward", 5.0, 20.0, 10.0, 1.0)
            lambda_param = st.slider("Lambda (cost weight)", 0.0, 0.5, 0.1, 0.05)
            mu_param = st.slider("Mu (latency weight)", 0.0, 0.5, 0.1, 0.05)
            failure_penalty = st.number_input("Failure Penalty", 0.0, 5.0, 2.0, 0.5)
            
            hyperparams = {
                'success_reward': success_reward,
                'lambda': lambda_param,
                'mu': mu_param,
                'failure_penalty': failure_penalty
            }
        else:
            st.info("Using default hyperparameters from training")
            hyperparams = {}
    
    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_simulation = st.button("Run Simulation", width='stretch', type="primary")
    
    st.markdown("---")
    
    # Simulation output
    if run_simulation:
        if not selected_algorithms:
            st.warning("Please select at least one algorithm")
        else:
            # Import simulation runner
            try:
                from src.utils.streamlit_runner import run_evaluation_simulation
                
                st.markdown("### Running Simulation...")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run simulation
                results = {}
                for i, algo in enumerate(selected_algorithms):
                    status_text.markdown(f"**Running {algo}...** ({i+1}/{len(selected_algorithms)})")
                    progress_bar.progress((i) / len(selected_algorithms))
                    
                    try:
                        result = run_evaluation_simulation(
                            algorithm=algo,
                            task_id=selected_task_id,
                            num_episodes=num_episodes,
                            hyperparams=hyperparams if enable_hyperparam else None,
                            include_metadata=include_metadata  # NEW: pass metadata flag
                        )
                        results[algo] = result
                    except Exception as e:
                        st.error(f"Error running {algo}: {str(e)}")
                        results[algo] = None
                    
                    progress_bar.progress((i + 1) / len(selected_algorithms))
                
                progress_bar.progress(1.0)
                status_text.markdown("**Simulation Complete!**")
                
                # Filter out failed runs
                results = {k: v for k, v in results.items() if v is not None}
                
                if not results:
                    st.error("All simulations failed. Please check the logs.")
                else:
                    # Display results
                    st.markdown("---")
                    st.markdown("### Results")
                    
                    # Create comparison dataframe
                    comparison_data = []
                    for algo, res in results.items():
                        comparison_data.append({
                            'Algorithm': algo,
                            'Success Rate': f"{res['success_rate']*100:.1f}%",
                            'Avg Reward': f"{res['avg_reward']:.2f}",
                            'Std Reward': f"{res['std_reward']:.2f}",
                            'Avg Cost': f"{res['avg_cost']:.2f}",
                            'Avg Latency': f"{res['avg_latency']:.2f}",
                            'Avg Steps': f"{res['avg_steps']:.1f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, width='stretch', hide_index=True)
                    
                    # Download results
                    csv_results = comparison_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_results,
                        file_name=f"simulation_task_{selected_task_id}_results.csv",
                        mime='text/csv',
                    )
                    
                    # Visualizations
                    st.markdown("---")
                    st.markdown("#### Performance Comparison")
                    
                    # Extract numeric values for plotting
                    plot_data = []
                    for algo, res in results.items():
                        plot_data.append({
                            'Algorithm': algo,
                            'Success Rate (%)': res['success_rate'] * 100,
                            'Avg Reward': res['avg_reward'],
                            'Avg Cost': res['avg_cost'],
                            'Avg Latency': res['avg_latency']
                        })
                    
                    plot_df = pd.DataFrame(plot_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = go.Figure(data=[
                            go.Bar(
                                x=plot_df['Algorithm'],
                                y=plot_df['Success Rate (%)'],
                                marker_color='#34a853',
                                text=plot_df['Success Rate (%)'].apply(lambda x: f"{x:.1f}%"),
                                textposition='auto',
                                textfont=dict(size=12, weight='bold')
                            )
                        ])
                        fig1.update_layout(
                            title="Success Rate",
                            title_font=dict(size=16, weight='bold'),
                            yaxis_title="Success Rate (%)",
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(family="Inter, sans-serif", size=12),
                            height=350
                        )
                        fig1.update_xaxes(showgrid=False)
                        fig1.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
                        st.plotly_chart(fig1, width='stretch')
                    
                    with col2:
                        fig2 = go.Figure(data=[
                            go.Bar(
                                x=plot_df['Algorithm'],
                                y=plot_df['Avg Reward'],
                                marker_color='#1a73e8',
                                text=plot_df['Avg Reward'].apply(lambda x: f"{x:.2f}"),
                                textposition='auto',
                                textfont=dict(size=12, weight='bold')
                            )
                        ])
                        fig2.update_layout(
                            title="Average Reward",
                            title_font=dict(size=16, weight='bold'),
                            yaxis_title="Avg Reward",
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(family="Inter, sans-serif", size=12),
                            height=350
                        )
                        fig2.update_xaxes(showgrid=False)
                        fig2.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
                        st.plotly_chart(fig2, width='stretch')
                    
                    # Cost vs Latency scatter
                    st.markdown("---")
                    st.markdown("#### Cost vs Latency Trade-off")
                    
                    fig3 = go.Figure(data=[
                        go.Scatter(
                            x=plot_df['Avg Cost'],
                            y=plot_df['Avg Latency'],
                            mode='markers+text',
                            marker=dict(size=15, color='#fbbc04', line=dict(width=2, color='white')),
                            text=plot_df['Algorithm'],
                            textposition='top center',
                            textfont=dict(size=11, weight='bold')
                        )
                    ])
                    fig3.update_layout(
                        title="Cost-Latency Trade-off",
                        title_font=dict(size=16, weight='bold'),
                        xaxis_title="Average Cost",
                        yaxis_title="Average Latency",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Inter, sans-serif", size=12),
                        height=400
                    )
                    fig3.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
                    fig3.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
                    st.plotly_chart(fig3, width='stretch')
            
            except ImportError as e:
                st.error(f"Failed to import simulation runner: {e}")
                st.info("Make sure you're running from the project root and all dependencies are installed.")
    else:
        st.info("Configure simulation parameters and click **Run Simulation** to start")

# ==================== TAB 3: Generalization Test ====================
with tab3:
    st.markdown("## Generalization Test (Seen vs Unseen Tasks)")
    st.markdown("Test whether agents can generalize to tasks they've never seen during training")
    
    st.info("""
    **What this tests**: Can agents handle novel tasks based on semantic understanding, or do they just memorize task IDs?
    
    **With metadata**: Agents receive task structure info (required tools, path lengths, constraints)  
    **Without metadata**: Agents only see task ID number (no semantic meaning)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Configuration")
        
        # Agent selection
        test_agent = st.selectbox(
            "**Agent to Test**",
            options=["ppo", "dqn", "q_learning", "sarsa"],
            index=0,
            format_func=lambda x: {"ppo": "PPO", "dqn": "DQN", "q_learning": "Q-Learning", "sarsa": "SARSA"}[x]
        )
        
        # Train/test split
        st.markdown("**Task Split**")
        train_tasks_str = st.text_input("Training tasks (comma-separated)", "0,1,2,3,4,5,6,7")
        test_tasks_str = st.text_input("Test tasks (unseen)", "8,9")
        
        # Episodes
        train_episodes = st.slider("Training episodes", 500, 5000, 2000, 500)
        eval_episodes = st.slider("Evaluation episodes per task", 20, 200, 50, 10)
        
        # Metadata toggle
        use_metadata = st.checkbox("Enable Task Metadata", value=True, help="With metadata, agents receive semantic task info for generalization")
    
    with col2:
        st.markdown("### Expected Results")
        
        if use_metadata:
            st.success("""
            **With Metadata (Expected):**
            - Seen tasks: ~80-90% success
            - Unseen tasks: ~65-75% success
            - Gap: ~10-15% (GOOD!)
            
            Small gap indicates true understanding of task structure.
            """)
        else:
            st.error("""
            **Without Metadata (Expected):**
            - Seen tasks: ~80-90% success
            - Unseen tasks: ~0-30% success  
            - Gap: ~50-85% (BAD!)
            
            Large gap indicates memorization, not generalization.
            """)
        
        st.markdown("---")
        st.markdown("**Estimated Time**")
        est_time = (train_episodes * 0.3 + eval_episodes * 12 * 0.5) / 60  # rough estimate
        st.caption(f"~{est_time:.1f} minutes for training + evaluation")
    
    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_generalization = st.button("Run Generalization Test", width='stretch', type="primary")
    
    st.markdown("---")
    
    if run_generalization:
        try:
            import subprocess
            import time
            
            # Parse task lists
            train_tasks = [int(x.strip()) for x in train_tasks_str.split(",")]
            test_tasks = [int(x.strip()) for x in test_tasks_str.split(",")]
            
            st.markdown("### Running Generalization Test...")
            st.markdown(f"**Training on tasks:** {train_tasks}")
            st.markdown(f"**Testing on tasks:** {test_tasks}")
            
            # Build command
            cmd = [
                "python", "evaluate_generalization.py",
                "--agent", test_agent,
                "--train_tasks", train_tasks_str,
                "--eval_tasks", test_tasks_str,
                "--episodes", str(train_episodes),
                "--eval_episodes", str(eval_episodes),
            ]
            
            if not use_metadata:
                cmd.append("--no_metadata")
            
            # Show command
            with st.expander("Command being executed"):
                st.code(" ".join(cmd))
            
            # Run subprocess
            status_placeholder = st.empty()
            output_placeholder = st.empty()
            
            status_placeholder.info("🔄 Training agent...")
            
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                status_placeholder.success("✅ Generalization test complete!")
                
                # Show output
                with st.expander("Full Output", expanded=False):
                    st.code(result.stdout)
                
                # Try to load results
                import pandas as pd
                from pathlib import Path
                
                metadata_suffix = "with_metadata" if use_metadata else "no_metadata"
                results_dir = Path("results/generalization")
                summary_file = results_dir / f"{test_agent}_summary_{metadata_suffix}.csv"
                
                if summary_file.exists():
                    st.markdown("---")
                    st.markdown("### Results")
                    
                    df = pd.read_csv(summary_file)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    seen_row = df[df['split'] == 'train (seen)'].iloc[0]
                    unseen_row = df[df['split'] == 'test (unseen)'].iloc[0]
                    
                    with col1:
                        st.metric("Seen Tasks Success", f"{seen_row['success_rate']*100:.1f}%")
                        st.caption(f"Reward: {seen_row['avg_reward']:.2f}")
                    
                    with col2:
                        st.metric("Unseen Tasks Success", f"{unseen_row['success_rate']*100:.1f}%")
                        st.caption(f"Reward: {unseen_row['avg_reward']:.2f}")
                    
                    with col3:
                        gap = (seen_row['success_rate'] - unseen_row['success_rate']) * 100
                        st.metric("Generalization Gap", f"{gap:.1f}%", delta_color="inverse")
                        
                        if gap < 20:
                            st.caption("✅ Good generalization!")
                        elif gap < 40:
                            st.caption("⚠️ Moderate generalization")
                        else:
                            st.caption("❌ Poor generalization")
                    
                    st.markdown("---")
                    
                    # Full table
                    st.markdown("#### Detailed Comparison")
                    display_df = df.copy()
                    display_df['success_rate'] = (display_df['success_rate'] * 100).round(1).astype(str) + '%'
                    display_df['avg_reward'] = display_df['avg_reward'].round(2)
                    display_df['avg_cost'] = display_df['avg_cost'].round(2)
                    display_df['avg_latency'] = display_df['avg_latency'].round(2)
                    st.dataframe(display_df, width='stretch', hide_index=True)
                    
                    # Download results
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results",
                        data=csv_data,
                        file_name=f"generalization_{test_agent}_{metadata_suffix}.csv",
                        mime='text/csv'
                    )
                    
                    # Interpretation
                    st.markdown("---")
                    st.markdown("#### Interpretation")
                    
                    if use_metadata:
                        if gap < 20:
                            st.success(f"""
                            **Great!** With metadata, your agent shows strong generalization.
                            The {gap:.1f}% gap means it can handle unseen tasks reasonably well
                            by understanding task structure (which tools are needed, path lengths, etc.)
                            """)
                        else:
                            st.warning(f"""
                            **Hmm.** Even with metadata, the {gap:.1f}% gap is larger than expected.
                            This might mean:
                            - Agent needs more training episodes
                            - Test tasks are significantly different from training tasks
                            - Agent hasn't fully learned to use the metadata features
                            """)
                    else:
                        if gap > 40:
                            st.error(f"""
                            **As expected:** Without metadata, agents struggle on unseen tasks ({gap:.1f}% gap).
                            They've memorized task IDs rather than learning task structure.
                            Try enabling metadata to see the improvement!
                            """)
                        else:
                            st.info(f"""
                            **Interesting:** The agent shows some generalization ({gap:.1f}% gap) even without metadata.
                            This might be because:
                            - Test tasks are very similar to training tasks
                            - Agent learned general patterns from tool sequences
                            - Lucky task embedding in neural networks
                            """)
                else:
                    st.warning("Results file not found, but test completed successfully. Check console output above.")
            else:
                status_placeholder.error("❌ Generalization test failed")
                st.error("Error during execution:")
                st.code(result.stderr)
                
        except subprocess.TimeoutExpired:
            st.error("⏱️ Test timed out (>10 minutes). Try reducing training episodes.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("Configure parameters and click **Run Generalization Test** to start")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 13px; padding: 20px;'>
    <b>Agent Orchestra RL Project</b> • Review 2 Demo • Built with Streamlit
</div>
""", unsafe_allow_html=True)
