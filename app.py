import streamlit as st
import subprocess
import re
import os
import pandas as pd

# --- UI Configuration ---
st.set_page_config(page_title="Learned Bloom Filter", page_icon="🛡️", layout="wide")

st.title("🛡️ Sandwiched Learned Bloom Filter")
st.markdown("### Real-Time Inference & Benchmarking Dashboard")
st.markdown("This dashboard interfaces directly with a native C++ hardware-level inference engine to benchmark memory compression and query latency.")
st.divider()

# --- Bulletproof Backend Execution ---
@st.cache_data(show_spinner=False)
def run_cpp_engine():
    engine_path = './bloom_engine'
    
    # Error Check 1: Does the file exist?
    if not os.path.exists(engine_path):
        return None, f"Error: '{engine_path}' not found. Did you compile the C++ code?"
    
    try:
        # Executes the compiled C++ engine
        result = subprocess.run([engine_path], capture_output=True, text=True, check=True)
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        return None, f"C++ Engine crashed during execution. Error: {e.stderr}"
    except PermissionError:
        return None, "Permission denied. Run 'chmod +x bloom_engine' in your terminal."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# --- Safe Parsing Function ---
def safe_extract(pattern, text, is_float=False):
    """Safely extracts regex matches without crashing if the pattern fails."""
    match = re.search(pattern, text)
    if match:
        return float(match.group(1)) if is_float else int(match.group(1))
    return 0.0 if is_float else 0

def parse_metrics(output):
    metrics = {}
    
    # Splitting into sections to avoid cross-matching
    sections = output.split('2. False Positive')
    mem_section = sections[0]
    rest = sections[1] if len(sections) > 1 else ""
    
    sections_2 = rest.split('3. Query Latency')
    fpr_section = sections_2[0] if len(sections_2) > 0 else ""
    lat_section = sections_2[1] if len(sections_2) > 1 else ""

    # Safe Extraction
    metrics['std_mem'] = safe_extract(r'Standard BF:\s+(\d+)\s+bits', mem_section)
    metrics['lrn_mem'] = safe_extract(r'Learned BF:\s+(\d+)\s+bits', mem_section)
    
    metrics['std_fpr'] = safe_extract(r'Standard BF:\s+([0-9.]+)', fpr_section, is_float=True)
    metrics['lrn_fpr'] = safe_extract(r'Learned BF:\s+([0-9.]+)', fpr_section, is_float=True)
    
    metrics['std_lat'] = safe_extract(r'Standard BF:\s+(\d+)\s+ns', lat_section)
    metrics['lrn_lat'] = safe_extract(r'Learned BF:\s+(\d+)\s+ns', lat_section)
    
    return metrics

# --- Dashboard Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Control Panel")
    st.info("Execute the native C++ engine to process 5,000 queries through the bitwise decision tree and standard hash functions.")
    
    if st.button("🚀 Run Hardware Benchmark", use_container_width=True, type="primary"):
        with st.spinner("Executing C++ binaries..."):
            raw_output, error = run_cpp_engine()
            
            if error:
                st.error(error)
            elif raw_output:
                st.session_state['metrics'] = parse_metrics(raw_output)
                st.session_state['raw_output'] = raw_output
                st.success("Benchmark completed successfully!")
            else:
                st.warning("Engine ran, but returned no output.")

# --- Results Rendering ---
if 'metrics' in st.session_state and st.session_state['metrics']:
    m = st.session_state['metrics']
    
    with col2:
        st.markdown("### 📊 Live Benchmark Metrics")
        
        # Row 1: Memory & Compression
        m1, m2, m3 = st.columns(3)
        m1.metric("Standard Memory", f"{m['std_mem']:,} bits")
        
        # Avoid division by zero safely
        compression = 0
        if m['std_mem'] > 0:
            compression = (1 - (m['lrn_mem'] / m['std_mem'])) * 100
            
        m2.metric("Learned Memory", f"{m['lrn_mem']:,} bits", f"-{compression:.0f}% Size", delta_color="inverse")
        
        latency_diff = m['lrn_lat'] - m['std_lat']
        m3.metric("Latency (Learned)", f"{m['lrn_lat']} ns", f"{latency_diff} ns vs Std", delta_color="inverse")

        # Row 2: Accuracy
        st.divider()
        m4, m5 = st.columns(2)
        m4.metric("Standard FPR", f"{m['std_fpr']}%")
        m5.metric("Learned FPR", f"{m['lrn_fpr']}%")

    # --- Visualizations ---
    st.markdown("### 📈 Memory Footprint Comparison")
    
    df = pd.DataFrame({
        "Data Structure": ["Standard Bloom Filter", "Sandwiched Learned Filter"],
        "Memory (Bits)": [m['std_mem'], m['lrn_mem']]
    })
    
    st.bar_chart(df, x="Data Structure", y="Memory (Bits)", color="#0068c9")
    
    with st.expander("Show Raw C++ Terminal Output"):
        st.code(st.session_state['raw_output'], language="text")