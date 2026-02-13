import streamlit as st
import subprocess
import re
import os
import time
import pandas as pd
import plotly.express as px

# --- UI Configuration & Custom Animated CSS ---
st.set_page_config(page_title="Neural Bloom Engine", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Dark Theme Background */
    [data-testid="stAppViewContainer"] { 
        background-color: #0b0f19; 
        color: #e6e8eb;
    }
    
    /* ANIMATION 1: Smooth Fade & Slide In */
    @keyframes slideUpFade {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Apply fade-in to the main blocks */
    div[data-testid="stVerticalBlock"] {
        animation: slideUpFade 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }

    /* ANIMATION 2: Hover pop on Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #151a28;
        border: 1px solid #2a3143;
        padding: 20px 20px 20px 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        border: 1px solid #00C9FF;
        box-shadow: 0 10px 25px rgba(0, 201, 255, 0.2);
    }
    
    /* ANIMATION 3: Continuous Breathing Glow on Primary Button */
    @keyframes breathingGlow {
        0% { box-shadow: 0 0 10px rgba(0, 201, 255, 0.4); }
        50% { box-shadow: 0 0 25px rgba(146, 254, 157, 0.7); }
        100% { box-shadow: 0 0 10px rgba(0, 201, 255, 0.4); }
    }
    
    /* Cyberpunk Gradient Primary Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #0b0f19;
        font-weight: 900;
        letter-spacing: 1px;
        border: none;
        border-radius: 8px;
        padding: 14px 24px;
        transition: transform 0.2s ease;
        animation: breathingGlow 3s infinite alternate;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #92FE9D 0%, #00C9FF 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- Bulletproof Backend Execution ---
@st.cache_data(show_spinner=False)
def run_cpp_engine():
    engine_path = './bloom_engine'
    if not os.path.exists(engine_path):
        return None, f"Error: '{engine_path}' not found. Did you compile it?"
    try:
        result = subprocess.run([engine_path], capture_output=True, text=True, check=True)
        return result.stdout, None
    except Exception as e:
        return None, f"Execution failed: {str(e)}"

def safe_extract(pattern, text, is_float=False):
    match = re.search(pattern, text)
    if match: return float(match.group(1)) if is_float else int(match.group(1))
    return 0.0 if is_float else 0

def parse_metrics(output):
    metrics = {}
    sections = output.split('2. False Positive')
    mem_section = sections[0]
    rest = sections[1] if len(sections) > 1 else ""
    sections_2 = rest.split('3. Query Latency')
    fpr_section, lat_section = (sections_2[0], sections_2[1]) if len(sections_2) > 1 else ("", "")

    metrics['std_mem'] = safe_extract(r'Standard BF:\s+(\d+)\s+bits', mem_section)
    metrics['lrn_mem'] = safe_extract(r'Learned BF:\s+(\d+)\s+bits', mem_section)
    metrics['std_fpr'] = safe_extract(r'Standard BF:\s+([0-9.]+)', fpr_section, is_float=True)
    metrics['lrn_fpr'] = safe_extract(r'Learned BF:\s+([0-9.]+)', fpr_section, is_float=True)
    metrics['std_lat'] = safe_extract(r'Standard BF:\s+(\d+)\s+ns', lat_section)
    metrics['lrn_lat'] = safe_extract(r'Learned BF:\s+(\d+)\s+ns', lat_section)
    return metrics

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Engine Specs")
    st.info("High-speed C++ executable paired with bitwise Machine Learning heuristics.")
    st.divider()
    st.metric("Target False Positive", "≤ 1.0%")
    st.metric("Architecture", "x86_64 Native")
    st.metric("Compiler Flag", "g++ -O3 (Max Speed)")
    st.metric("Status", "🟢 Online")

# --- Main Dashboard Layout ---
st.title("🛡️ Neural Data Structure Engine")
st.markdown("<p style='color: #00C9FF; font-size: 1.1rem; margin-top:-15px; font-weight: 600;'>Sandwiched Learned Bloom Filter • Hardware Inference Monitor</p>", unsafe_allow_html=True)
st.divider()

# Organize content into modern Tabs
tab1, tab2 = st.tabs(["🚀 Live Dashboard", "💻 C++ Terminal Logs"])

with tab1:
    col1, col2 = st.columns([1, 2.5], gap="large")

    with col1:
        st.markdown("### Control Panel")
        st.caption("Initialize the native C++ engine to process streaming queries against the bitwise decision tree.")
        
        if st.button("⚡ EXECUTE BENCHMARK", use_container_width=True):
            # Simulated UI progress bar for dramatic loading effect
            st.caption("Booting C++ binary sequence...")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.003) # Super fast fake loading
                progress_bar.progress(i + 1)
            
            raw_output, error = run_cpp_engine()
            progress_bar.empty()
            
            if error:
                st.error(error)
            elif raw_output:
                st.session_state['metrics'] = parse_metrics(raw_output)
                st.session_state['raw_output'] = raw_output
                st.toast("Hardware benchmark completed successfully!", icon="✅")

    with col2:
        if 'metrics' in st.session_state and st.session_state['metrics']:
            m = st.session_state['metrics']
            st.markdown("### 📊 Live Benchmark Metrics")
            
            # Row 1: Memory & Speed
            m1, m2, m3 = st.columns(3)
            m1.metric("Standard Memory", f"{m['std_mem']:,} bits")
            compression = (1 - (m['lrn_mem'] / m['std_mem'])) * 100 if m['std_mem'] > 0 else 0
            m2.metric("Learned Memory", f"{m['lrn_mem']:,} bits", f"-{compression:.0f}% Size", delta_color="inverse")
            latency_diff = m['lrn_lat'] - m['std_lat']
            m3.metric("Latency (Learned)", f"{m['lrn_lat']} ns", f"{latency_diff} ns vs Std", delta_color="inverse")

            # Row 2: Accuracy
            st.divider()
            m4, m5 = st.columns(2)
            m4.metric("Standard FPR", f"{m['std_fpr']}%")
            m5.metric("Learned FPR", f"{m['lrn_fpr']}%")

            # Interactive Plotly Chart
            st.markdown("### 📈 Memory Compression Analysis")
            df = pd.DataFrame({
                "Architecture": ["Standard Bloom Filter", "Learned Bloom Filter"],
                "Memory (Bits)": [m['std_mem'], m['lrn_mem']]
            })
            
            fig = px.bar(df, x="Architecture", y="Memory (Bits)", color="Architecture",
                         color_discrete_sequence=["#2a3143", "#00C9FF"], text_auto=True)
            
            fig.update_traces(textfont_size=16, textangle=0, textposition="outside", cliponaxis=False)
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FFF"), showlegend=False,
                margin=dict(t=30, b=20, l=20, r=20),
                yaxis=dict(showgrid=True, gridcolor='#2a3143')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("System standing by. Click 'Execute Benchmark' to generate real-time metrics.")

with tab2:
    st.markdown("### C++ Engine Execution Logs")
    if 'raw_output' in st.session_state:
        st.code(st.session_state['raw_output'], language="cpp")
    else:
        st.caption("No logs available. Run the benchmark first.")