import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ---    streamlit run p.py   ---




# ---    https://statistics-tool-2gnsqckaeejxn5qxhmyte2.streamlit.app/   ---





# --- Page Configuration ---
st.set_page_config(page_title="Statistics Tool - Dr. Mohamed Sobh ", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: dimgray;
}
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
def draw_header():
    col1, col2 = st.columns([1, 5])
    with col1:
    
        st.image("dr_mohamed.png", width=200) 
    with col2:
        st.title("Advanced Statistics & Probability Tool")
        st.subheader("Under Supervision of: Dr. Mohamed Sobh")
        st.markdown("### **By Student: Ahmed Elshaar**")
    st.divider()

draw_header()


# --- Advanced Plotting Engine (Fixed Two-Tailed Visualization) ---
def plot_statistics(dist_type, test_stat, critical_val, alpha, tail_type, df=None, mode="Testing", ci_lower=None, ci_upper=None, ci_crit_low=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    fig.patch.set_facecolor('#F0F4F8')
    ax.set_facecolor('#F8FAFC')

    # ── CI MODE ────────────────────────────────────────────────────────────
    if mode == "CI" and ci_lower is not None and ci_upper is not None:

        if dist_type in ("z", "t"):
            center = (ci_lower + ci_upper) / 2
            se     = (ci_upper - ci_lower) / (2 * abs(critical_val))
            x = np.linspace(center - 4*se, center + 4*se, 1000)
            y = stats.norm.pdf(x, center, se) if dist_type == "z" else stats.t.pdf((x - center)/se, df)/se
            dist_label = "Sampling Distribution (Z)" if dist_type == "z" else f"Sampling Distribution (t, df={df})"

        else:  # chi2
            c1 = ci_crit_low
            c2 = critical_val
            upper_limit = max(30, c2 * 1.5)
            x = np.linspace(0, upper_limit, 1000)
            y = stats.chi2.pdf(x, df)
            dist_label = f"Chi-Square Distribution (df={df})"

        ax.plot(x, y, '#1A237E', lw=2.5, label=dist_label)

        if dist_type in ("z", "t"):
            ax.fill_between(x, y, where=((x >= ci_lower) & (x <= ci_upper)), color='#27AE60', alpha=0.35, label='Confidence Interval Region')
            ax.axvline(ci_lower, color='#8E44AD', linestyle='--', lw=2)
            ax.axvline(ci_upper, color='#8E44AD', linestyle='--', lw=2)
            ax.text(ci_lower, max(y)*0.2, f'{ci_lower:.4f}', color='#8E44AD', fontweight='bold', ha='right')
            ax.text(ci_upper, max(y)*0.2, f'{ci_upper:.4f}', color='#8E44AD', fontweight='bold', ha='left')
        else:
            ax.fill_between(x, y, where=((x >= c1) & (x <= c2)), color='#27AE60', alpha=0.35, label='Confidence Interval Region')
            ax.axvline(c1, color='#8E44AD', linestyle='--', lw=2)
            ax.axvline(c2, color='#8E44AD', linestyle='--', lw=2)
            ax.text(c1, max(y)*0.2, f'{ci_upper:.4f}', color='#8E44AD', fontweight='bold', ha='right')
            ax.text(c2, max(y)*0.2, f'{ci_lower:.4f}', color='#8E44AD', fontweight='bold', ha='left')

        ax.set_title(f"Confidence Interval Plot", fontsize=15)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        return

    # ── TESTING MODE (unchanged) ───────────────────────────────────────────
    if dist_type == "z":
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        dist_label = "Standard Normal Distribution (Z)"
    elif dist_type == "t":
        x = np.linspace(-4, 4, 1000)
        y = stats.t.pdf(x, df)
        dist_label = f"t-Distribution (df={df})"
    else:
        upper_limit = max(30, (critical_val if critical_val else 10) * 1.5)
        x = np.linspace(0, upper_limit, 1000)
        y = stats.chi2.pdf(x, df)
        dist_label = f"Chi-Square Distribution (df={df})"

    ax.plot(x, y, '#1A237E', lw=2.5, label=dist_label)

    if "Two" in tail_type:
        cv = abs(critical_val)
        ax.fill_between(x, y, where=(x >  cv), color='#8E44AD', alpha=0.45, label='Rejection Region')
        ax.fill_between(x, y, where=(x < -cv), color='#8E44AD', alpha=0.45)
        ax.axvline( cv, color='#8E44AD', linestyle='--', lw=2)
        ax.axvline(-cv, color='#8E44AD', linestyle='--', lw=2)
        ax.text( cv, max(y)*0.2, f'+{cv:.3f}', color='#8E44AD', fontweight='bold', ha='left')
        ax.text(-cv, max(y)*0.2, f'-{cv:.3f}', color='#8E44AD', fontweight='bold', ha='right')
    elif "Right" in tail_type or ">" in tail_type:
        ax.fill_between(x, y, where=(x > critical_val), color='#8E44AD', alpha=0.45, label='Rejection Region')
        ax.axvline(critical_val, color='#8E44AD', linestyle='--', lw=2)
        ax.text(critical_val, max(y)*0.2, f'{critical_val:.3f}', color='#8E44AD', fontweight='bold', ha='left')
    elif "Left" in tail_type or "<" in tail_type:
        ax.fill_between(x, y, where=(x < critical_val), color='#8E44AD', alpha=0.45, label='Rejection Region')
        ax.axvline(critical_val, color='#8E44AD', linestyle='--', lw=2)
        ax.text(critical_val, max(y)*0.2, f'{critical_val:.3f}', color='#8E44AD', fontweight='bold', ha='right')

    ax.axvline(test_stat, color='#E67E22', linestyle='-.', lw=3, label=f'Test Stat: {test_stat:.3f}')
    ax.scatter([test_stat], [0.005], color='#E67E22', s=150, marker='D', zorder=5)
    ax.annotate(f'Test Value: {test_stat:.2f}', xy=(test_stat, 0.01), xytext=(test_stat, max(y)*0.8),
                 arrowprops=dict(facecolor='#E67E22', shrink=0.05), color='#E67E22', fontweight='bold', ha='center')

    ax.set_title(f"Statistical Distribution Plot ({tail_type})", fontsize=15)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

# --- Decision Logic Helper ---
def display_decision(p_val, alpha, test_stat, crit_val, tail):
    st.write(f"---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Statistic (Calculated)", f"{test_stat:.4f}")
        st.metric("P-Value", f"{p_val:.4f}")
    with col2:
        if p_val < alpha:
            st.error("Decision: REJECT H0")
            st.warning("Result: STATISTICALLY SIGNIFICANT")
        else:
            st.success("Decision: FAIL TO REJECT H0 (ACCEPT)")
            st.info("Result: NOT STATISTICALLY SIGNIFICANT")

# --- Main Navigation ---
main_choice = st.radio("Select Process:", ["Confidence Interval", "Hypothesis Testing"], horizontal=True)

# ---------------------------------------------------------
# 1. CONFIDENCE INTERVAL SECTION
# ---------------------------------------------------------
if main_choice == "Confidence Interval":
    st.header("🎯 Confidence Interval Mode")
    param = st.selectbox("Select Parameter:", ["Mean", "Variance", "Proportion"])
    n, x_bar, std_dev, s_sq, p_hat, calculated = 0, 0, 0, 0, 0, False

    if param == "Mean":
        sigma_known = st.radio("Is Sigma (σ) Known?", ["Yes", "No"])
        input_way = st.radio("Input Method:", ["Summary Statistics", "Raw Data Input"])
        if input_way == "Summary Statistics":
            x_bar = st.number_input("Mean (x̄):")
            val = st.number_input("Dispersion (s or σ):", value=1.0)
            is_sq = st.checkbox("Is this value Squared (Variance)?")
            std_dev = np.sqrt(val) if is_sq else val
            n = st.number_input("n:", min_value=1, value=30)
        else:
            raw_meth = st.radio("Format:", ["Values", "Sums"])
            if raw_meth == "Values":
                raw = st.text_input("Data (comma separated):")
                if raw:
                    data = [float(i) for i in raw.split(",")]
                    n = len(data); sx = sum(data); sx2 = sum([i**2 for i in data])
                    x_bar = sx / n; s_sq = (sx2 - (sx**2 / n)) / (n - 1); std_dev = np.sqrt(s_sq)
            else:
                sx, sx2, n = st.number_input("Σx:"), st.number_input("Σx²:"), st.number_input("n:", min_value=2, value=10)
                x_bar = sx/n; s_sq = (sx2 - (sx**2/n))/(n-1); std_dev = np.sqrt(s_sq)

        cl = st.slider("Confidence Level:", 0.80, 0.99, 0.95)
        if st.button("Calculate CI"):
            alpha = 1-cl; use_dist = "z" if (sigma_known=="Yes" or n>=30) else "t"
            crit = stats.norm.ppf(1-alpha/2) if use_dist=="z" else stats.t.ppf(1-alpha/2, n-1)
            margin = crit * (std_dev / np.sqrt(n))
            lower, upper = x_bar - margin, x_bar + margin
            st.success(f"CI: ({lower:.4f} , {upper:.4f})")
            col_plot, col_vals = st.columns([4, 1])
            with col_plot:
                plot_statistics(use_dist, x_bar, crit, alpha, "Two-Tailed", df=n-1 if use_dist=="t" else None, mode="CI", ci_lower=lower, ci_upper=upper)
            with col_vals:
                st.metric("Lower Bound", f"{lower:.4f}")
                st.metric("Upper Bound", f"{upper:.4f}")
                st.metric("Critical Value", f"± {crit:.4f}")
                st.divider()
                st.markdown("**خطوات الحل:**")
                st.markdown(f"**1.** α = 1 - CL")
                st.markdown(f"α = 1 - {cl} = **{alpha:.4f}**")
                st.markdown(f"**2.** القيمة الحرجة")
                if use_dist == "z":
                    st.markdown(f"z(α/2) = **±{crit:.4f}**")
                else:
                    st.markdown(f"t(α/2,{int(n-1)}) = **±{crit:.4f}**")
                st.markdown(f"**3.** هامش الخطأ E")
                if use_dist == "z":
                    st.markdown(f"E = z × σ/√n")
                else:
                    st.markdown(f"E = t × s/√n")
                st.markdown(f"E = {crit:.4f} × {std_dev:.4f}/√{int(n)}")
                st.markdown(f"E = **{margin:.4f}**")
                st.markdown(f"**4.** الفترة = x̄ ± E")
                st.markdown(f"= {x_bar:.4f} ± {margin:.4f}")
                st.markdown(f"= (**{lower:.4f}** , **{upper:.4f}**)")
            calculated = True

    elif param == "Variance":
        v_input = st.radio("Input Method:", ["Summary", "Raw Data Input"])
        if v_input == "Summary":
            v_val = st.number_input("s:", value=1.0)
            s_sq = v_val if st.checkbox("Is Squared (s²)?") else v_val**2
            n = st.number_input("n:", min_value=2, value=10)
        else:
            raw_meth = st.radio("Format:", ["Values", "Sums"], key="vraw")
            if raw_meth == "Values":
                raw = st.text_input("Data:", key="vtxt")
                if raw: 
                    data = [float(i) for i in raw.split(",")]
                    n = len(data); sx = sum(data); sx2 = sum([i**2 for i in data])
                    s_sq = (sx2 - (sx**2 / n)) / (n - 1)
            else:
                sx, sx2, n = st.number_input("Σx:"), st.number_input("Σx²:"), st.number_input("n:", min_value=2, value=10)
                s_sq = (sx2 - (sx**2/n))/(n-1)

        cl = st.slider("Confidence Level:", 0.80, 0.99, 0.95)
        if st.button("Calculate CI"):
            alpha = 1-cl; c1, c2 = stats.chi2.ppf(alpha/2, n-1), stats.chi2.ppf(1-alpha/2, n-1)
            lower, upper = ((n-1)*s_sq)/c2, ((n-1)*s_sq)/c1
            st.success(f"Variance CI: ({lower:.4f} , {upper:.4f})")
            col_plot, col_vals = st.columns([4, 1])
            with col_plot:
                plot_statistics("chi2", s_sq, c2, alpha, "Right-Tailed", df=n-1, mode="CI", ci_lower=lower, ci_upper=upper, ci_crit_low=c1)
            with col_vals:
                st.metric("Lower Bound", f"{lower:.4f}")
                st.metric("Upper Bound", f"{upper:.4f}")
                st.metric("χ² Lower (c1)", f"{c1:.4f}")
                st.metric("χ² Upper (c2)", f"{c2:.4f}")
                st.divider()
                st.markdown("**خطوات الحل:**")
                st.markdown(f"**1.** α = 1 - CL")
                st.markdown(f"α = 1 - {cl} = **{alpha:.4f}**")
                st.markdown(f"**2.** القيم الحرجة")
                st.markdown(f"c1 = χ²(α/2, n-1)")
                st.markdown(f"= χ²({alpha/2:.4f}, {int(n-1)}) = **{c1:.4f}**")
                st.markdown(f"c2 = χ²(1-α/2, n-1)")
                st.markdown(f"= χ²({1-alpha/2:.4f}, {int(n-1)}) = **{c2:.4f}**")
                st.markdown(f"**3.** القانون:")
                st.markdown(f"( (n-1)s² / c2 , (n-1)s² / c1 )")
                st.markdown(f"**4.** التعويض:")
                st.markdown(f"({int(n-1)} × {s_sq:.4f} / {c2:.4f})")
                st.markdown(f"= **{lower:.4f}**")
                st.markdown(f"({int(n-1)} × {s_sq:.4f} / {c1:.4f})")
                st.markdown(f"= **{upper:.4f}**")
            calculated = True

    elif param == "Proportion":
        p_hat_method = st.radio("How to provide p̂?", ["Directly", "From x and n"])
        if p_hat_method == "Directly":
            p_hat = st.number_input("p̂:", 0.0, 1.0, step=0.01)
            n = st.number_input("n:", min_value=1, value=100)
        else:
            x_count = st.number_input("x:", min_value=0)
            n = st.number_input("n:", min_value=1, value=100)
            p_hat = x_count / n if n > 0 else 0
        
        cl = st.slider("Confidence Level:", 0.80, 0.99, 0.95)
        if st.button("Calculate CI"):
            alpha = 1-cl; crit = stats.norm.ppf(1-alpha/2)
            margin = crit * np.sqrt((p_hat*(1-p_hat))/n)
            lower, upper = p_hat - margin, p_hat + margin
            st.success(f"Proportion CI: ({lower:.4f} , {upper:.4f})")
            col_plot, col_vals = st.columns([4, 1])
            with col_plot:
                plot_statistics("z", p_hat, crit, alpha, "Two-Tailed", mode="CI", ci_lower=lower, ci_upper=upper)
            with col_vals:
                st.metric("Lower Bound", f"{lower:.4f}")
                st.metric("Upper Bound", f"{upper:.4f}")
                st.metric("Critical Value", f"± {crit:.4f}")
                st.divider()
                st.markdown("**خطوات الحل:**")
                st.markdown(f"**1.** α = 1 - CL")
                st.markdown(f"α = 1 - {cl} = **{alpha:.4f}**")
                st.markdown(f"**2.** القيمة الحرجة")
                st.markdown(f"z(α/2) = **±{crit:.4f}**")
                st.markdown(f"**3.** هامش الخطأ E")
                st.markdown(f"E = z × √(p̂(1-p̂)/n)")
                st.markdown(f"E = {crit:.4f} × √({p_hat:.4f}×{1-p_hat:.4f}/{int(n)})")
                st.markdown(f"E = **{margin:.4f}**")
                st.markdown(f"**4.** الفترة = p̂ ± E")
                st.markdown(f"= {p_hat:.4f} ± {margin:.4f}")
                st.markdown(f"= (**{lower:.4f}** , **{upper:.4f}**)")
            calculated = True

    if calculated:
        st.divider()
        if st.checkbox("⚡ Link to Hypothesis Test"):
            h1 = st.radio("H1:", ["Two-Tailed (≠)", "Right-Tailed (>)", "Left-Tailed (<)"], horizontal=True)
            alpha_test = st.number_input("Significance α:", 0.01, 0.10, 0.05, key="link_alpha")
            
            if param == "Mean":
                mu0 = st.number_input("Null Mean μ0:")
                if st.button("Run Linked Test"):
                    dist = "z" if (sigma_known=="Yes" or n>=30) else "t"; df = n-1 if dist=="t" else None
                    z_t = (x_bar - mu0)/(std_dev/np.sqrt(n))
                    if "Two" in h1: 
                        p = 2*(1-stats.norm.cdf(abs(z_t))) if dist=="z" else 2*(1-stats.t.cdf(abs(z_t), df))
                        crit = stats.norm.ppf(1-alpha_test/2) if dist=="z" else stats.t.ppf(1-alpha_test/2, df)
                    elif ">" in h1: 
                        p = 1-stats.norm.cdf(z_t) if dist=="z" else 1-stats.t.cdf(z_t, df)
                        crit = stats.norm.ppf(1-alpha_test) if dist=="z" else stats.t.ppf(1-alpha_test, df)
                    else: 
                        p = stats.norm.cdf(z_t) if dist=="z" else stats.t.cdf(z_t, df)
                        crit = stats.norm.ppf(alpha_test) if dist=="z" else stats.t.ppf(alpha_test, df)
                    display_decision(p, alpha_test, z_t, crit, h1)
                    plot_statistics(dist, z_t, crit, alpha_test, h1, df, mode="Testing")

            elif param == "Variance":
                v0 = st.number_input("Null Variance σ0²:")
                if st.button("Run Linked Test"):
                    chi = ((n-1)*s_sq)/v0
                    crit = stats.chi2.ppf(1-alpha_test if ">" in h1 else alpha_test, n-1)
                    p = 1-stats.chi2.cdf(chi, n-1) if ">" in h1 else stats.chi2.cdf(chi, n-1)
                    display_decision(p, alpha_test, chi, crit, h1)
                    plot_statistics("chi2", chi, crit, alpha_test, h1, n-1, mode="Testing")

            elif param == "Proportion":
                p0_t = st.number_input("Null Proportion p0:")
                if st.button("Run Linked Test"):
                    zs = (p_hat - p0_t)/np.sqrt((p0_t*(1-p0_t))/n)
                    if "Two" in h1: p = 2*(1-stats.norm.cdf(abs(zs))); crit = stats.norm.ppf(1-alpha_test/2)
                    elif ">" in h1: p = 1-stats.norm.cdf(zs); crit = stats.norm.ppf(1-alpha_test)
                    else: p = stats.norm.cdf(zs); crit = stats.norm.ppf(alpha_test)
                    display_decision(p, alpha_test, zs, crit, h1)
                    plot_statistics("z", zs, crit, alpha_test, h1, mode="Testing")

# ---------------------------------------------------------
# 2. HYPOTHESIS TESTING SECTION
# ---------------------------------------------------------
else:
    st.header("🔬 Hypothesis Testing Mode")
    test_param = st.selectbox("Select Parameter to Test:", ["Mean", "Variance", "Proportion", "Two Means"])
    h1_type = st.radio("Alternative Hypothesis (H1):", ["Two-Tailed (≠)", "Right-Tailed (>)", "Left-Tailed (<)"])
    alpha = st.number_input("Significance Level (α):", 0.01, 0.10, 0.05)

    if test_param == "Mean":
        mu0 = st.number_input("Null Mean (μ0):")
        sigma_known = st.radio("Is Sigma (σ) Known?", ["Yes", "No"], key="ht_sigma")
        input_way = st.radio("Input Method:", ["Summary Statistics", "Raw Data Input"], key="ht_in_way")
        if input_way == "Summary Statistics":
            x_bar, val, n = st.number_input("x̄:"), st.number_input("s or σ:"), st.number_input("n:", value=30)
            std_dev = np.sqrt(val) if st.checkbox("Is Squared?") else val
        else:
            raw_m = st.radio("Format:", ["Values", "Sums"], key="ht_raw_m")
            if raw_m == "Values":
                txt = st.text_input("Data (comma separated):")
                if txt: 
                    data = [float(i) for i in txt.split(",")]
                    n = len(data); sx = sum(data); sx2 = sum([i**2 for i in data])
                    x_bar = sx/n; s_sq = (sx2 - (sx**2/n))/(n-1); std_dev = np.sqrt(s_sq)
                else: st.stop()
            else:
                sx, sx2, n = st.number_input("Σx:"), st.number_input("Σx²:"), st.number_input("n:", value=10)
                x_bar = sx/n; s_sq = (sx2 - (sx**2/n))/(n-1); std_dev = np.sqrt(s_sq)

        if st.button("Run Mean Test"):
            dist = "z" if (sigma_known=="Yes" or n>=30) else "t"; z_t = (x_bar - mu0)/(std_dev/np.sqrt(n)); df = n-1 if dist=="t" else None
            if "Two" in h1_type: 
                p = 2*(1-stats.norm.cdf(abs(z_t))) if dist=="z" else 2*(1-stats.t.cdf(abs(z_t), df))
                crit = stats.norm.ppf(1-alpha/2) if dist=="z" else stats.t.ppf(1-alpha/2, df)
            elif ">" in h1_type: 
                p = 1-stats.norm.cdf(z_t) if dist=="z" else 1-stats.t.cdf(z_t, df)
                crit = stats.norm.ppf(1-alpha) if dist=="z" else stats.t.ppf(1-alpha, df)
            else: 
                p = stats.norm.cdf(z_t) if dist=="z" else stats.t.cdf(z_t, df)
                crit = stats.norm.ppf(alpha) if dist=="z" else stats.t.ppf(alpha, df)
            display_decision(p, alpha, z_t, crit, h1_type)
            plot_statistics(dist, z_t, crit, alpha, h1_type, df, mode="Testing")

    elif test_param == "Variance":
        sigma0_sq = st.number_input("Hypothesized σ₀²:")
        v_in = st.radio("Input Method:", ["Summary", "Raw Data Input"], key="ht_v_in")
        if v_in == "Summary":
            v_val, n = st.number_input("s:"), st.number_input("n:", value=10)
            s_sq = v_val if st.checkbox("Is Squared?") else v_val**2
        else:
            raw_m = st.radio("Format:", ["Values", "Sums"], key="ht_v_raw")
            if raw_m == "Values":
                txt = st.text_input("Data:"); 
                if txt: 
                    data = [float(i) for i in txt.split(",")]
                    n = len(data); sx = sum(data); sx2 = sum([i**2 for i in data])
                    s_sq = (sx2 - (sx**2/n))/(n-1)
                else: st.stop()
            else:
                sx, sx2, n = st.number_input("Σx:"), st.number_input("Σx²:"), st.number_input("n:", value=10)
                s_sq = (sx2 - (sx**2/n))/(n-1)

        if st.button("Run Variance Test"):
            chi = ((n-1)*s_sq)/sigma0_sq
            p = 1-stats.chi2.cdf(chi, n-1) if ">" in h1_type else stats.chi2.cdf(chi, n-1)
            crit = stats.chi2.ppf(1-alpha if ">" in h1_type else alpha, n-1)
            display_decision(p, alpha, chi, crit, h1_type)
            plot_statistics("chi2", chi, crit, alpha, h1_type, n-1, mode="Testing")

    elif test_param == "Proportion":
        p_hat_meth_ht = st.radio("Provide p̂:", ["Directly", "From x and n"], key="ht_prop_meth")
        if p_hat_meth_ht == "Directly":
            p_hat = st.number_input("p̂:", 0.0, 1.0, step=0.01)
            n = st.number_input("n:", min_value=1, value=100)
        else:
            x_count = st.number_input("x:", min_value=0)
            n = st.number_input("n:", min_value=1, value=100)
            p_hat = x_count / n if n > 0 else 0
            
        p0 = st.number_input("p0 (Null):", 0.0, 1.0)
        
        if st.button("Run Proportion Test"):
            zs = (p_hat - p0)/np.sqrt((p0*(1-p0))/n)
            if "Two" in h1_type: p = 2*(1-stats.norm.cdf(abs(zs))); crit = stats.norm.ppf(1-alpha/2)
            elif ">" in h1_type: p = 1-stats.norm.cdf(zs); crit = stats.norm.ppf(1-alpha)
            else: p = stats.norm.cdf(zs); crit = stats.norm.ppf(alpha)
            display_decision(p, alpha, zs, crit, h1_type)
            plot_statistics("z", zs, crit, alpha, h1_type, mode="Testing")


    elif test_param == "Two Means":
        st.markdown("#### Testing the Difference Between Two Means")
        sigma_known_2 = st.radio("Are σ1 and σ2 Known?", ["Yes (Z-test)", "No (T-test)"], key="tm_sigma")

        st.markdown("**Sample 1:**")
        col1, col2, col3 = st.columns(3)
        with col1: x_bar1 = st.number_input("x̄₁:", key="tm_xbar1")
        with col2: n1 = st.number_input("n₁:", min_value=2, value=10, key="tm_n1")
        with col3:
            val1 = st.number_input("σ₁ or s₁:", value=1.0, key="tm_s1")
            is_sq1 = st.checkbox("Squared?", key="tm_sq1")
            s1 = np.sqrt(val1) if is_sq1 else val1

        st.markdown("**Sample 2:**")
        col1, col2, col3 = st.columns(3)
        with col1: x_bar2 = st.number_input("x̄₂:", key="tm_xbar2")
        with col2: n2 = st.number_input("n₂:", min_value=2, value=10, key="tm_n2")
        with col3:
            val2 = st.number_input("σ₂ or s₂:", value=1.0, key="tm_s2")
            is_sq2 = st.checkbox("Squared?", key="tm_sq2")
            s2 = np.sqrt(val2) if is_sq2 else val2

        if st.button("Run Two Means Test"):
            if sigma_known_2 == "Yes (Z-test)":
                # Z-test: σ1 σ2 known
                dist_tm = "z"
                df_tm = None
                test_val = (x_bar1 - x_bar2) / np.sqrt((s1**2/n1) + (s2**2/n2))
                method_label = "Z-test (σ known)"
                sp = None
                ratio = None
            else:
                # T-test: σ unknown
                ratio = s1 / s2
                if 0.5 <= ratio <= 2:
                    # Equal variances → pooled
                    sp_sq = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
                    sp = np.sqrt(sp_sq)
                    df_tm = int(n1 + n2 - 2)
                    test_val = (x_bar1 - x_bar2) / (sp * np.sqrt(1/n1 + 1/n2))
                    method_label = f"T-test (σ1=σ2, pooled, df={df_tm})"
                else:
                    # Unequal variances
                    sp = None
                    sp_sq = None
                    df_tm = int(min(n1-1, n2-1))
                    test_val = (x_bar1 - x_bar2) / np.sqrt((s1**2/n1) + (s2**2/n2))
                    method_label = f"T-test (σ1≠σ2, df={df_tm})"
                dist_tm = "t"

            # Critical value & p-value
            if dist_tm == "z":
                if "Two" in h1_type:
                    crit_tm = stats.norm.ppf(1-alpha/2)
                    p_tm = 2*(1-stats.norm.cdf(abs(test_val)))
                elif ">" in h1_type:
                    crit_tm = stats.norm.ppf(1-alpha)
                    p_tm = 1-stats.norm.cdf(test_val)
                else:
                    crit_tm = stats.norm.ppf(alpha)
                    p_tm = stats.norm.cdf(test_val)
            else:
                if "Two" in h1_type:
                    crit_tm = stats.t.ppf(1-alpha/2, df_tm)
                    p_tm = 2*(1-stats.t.cdf(abs(test_val), df_tm))
                elif ">" in h1_type:
                    crit_tm = stats.t.ppf(1-alpha, df_tm)
                    p_tm = 1-stats.t.cdf(test_val, df_tm)
                else:
                    crit_tm = stats.t.ppf(alpha, df_tm)
                    p_tm = stats.t.cdf(test_val, df_tm)

            st.info(f"Method used: **{method_label}**")
            display_decision(p_tm, alpha, test_val, crit_tm, h1_type)

            col_plot, col_vals = st.columns([4, 1])
            with col_plot:
                plot_statistics(dist_tm, test_val, crit_tm, alpha, h1_type, df_tm, mode="Testing")
            with col_vals:
                st.metric("Test Statistic", f"{test_val:.4f}")
                st.metric("Critical Value", f"±{abs(crit_tm):.4f}" if "Two" in h1_type else f"{crit_tm:.4f}")
                st.metric("P-Value", f"{p_tm:.4f}")
                st.divider()
                st.markdown("**خطوات الحل:**")
                st.markdown(f"**H₀:** μ₁ = μ₂")
                if "Two" in h1_type:   st.markdown(f"**H₁:** μ₁ ≠ μ₂")
                elif ">" in h1_type:   st.markdown(f"**H₁:** μ₁ > μ₂")
                else:                  st.markdown(f"**H₁:** μ₁ < μ₂")
                st.markdown(f"**α =** {alpha}")

                if sigma_known_2 == "Yes (Z-test)":
                    st.markdown("**القانون:**")
                    st.markdown("Z = (x̄₁-x̄₂) / √(σ₁²/n₁ + σ₂²/n₂)")
                    st.markdown(f"= ({x_bar1}-{x_bar2}) / √({s1**2:.4f}/{int(n1)} + {s2**2:.4f}/{int(n2)})")
                    st.markdown(f"= **{test_val:.4f}**")
                    if "Two" in h1_type:
                        st.markdown(f"z(α/2) = z({alpha/2:.4f}) = **±{crit_tm:.4f}**")
                    elif ">" in h1_type:
                        st.markdown(f"z(α) = z({alpha}) = **{crit_tm:.4f}**")
                    else:
                        st.markdown(f"z(α) = z({alpha}) = **{crit_tm:.4f}**")
                else:
                    st.markdown(f"s₁/s₂ = {s1:.4f}/{s2:.4f} = **{ratio:.4f}**")
                    if 0.5 <= ratio <= 2:
                        st.markdown("0.5 ≤ ratio ≤ 2 → σ₁ = σ₂ (Pooled)")
                        st.markdown("**sp² = [(n₁-1)s₁²+(n₂-1)s₂²] / (n₁+n₂-2)**")
                        sp_sq_val = ((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2)
                        st.markdown(f"= [{int(n1-1)}×{s1**2:.4f}+{int(n2-1)}×{s2**2:.4f}] / {int(n1+n2-2)}")
                        st.markdown(f"sp² = **{sp_sq_val:.4f}** → sp = **{np.sqrt(sp_sq_val):.4f}**")
                        st.markdown("**T = (x̄₁-x̄₂) / (sp×√(1/n₁+1/n₂))**")
                        st.markdown(f"= ({x_bar1}-{x_bar2}) / ({np.sqrt(sp_sq_val):.4f}×√(1/{int(n1)}+1/{int(n2)}))")
                        st.markdown(f"= **{test_val:.4f}**")
                        st.markdown(f"df = n₁+n₂-2 = **{df_tm}**")
                    else:
                        st.markdown("ratio < 0.5 or > 2 → σ₁ ≠ σ₂")
                        st.markdown("**T = (x̄₁-x̄₂) / √(s₁²/n₁ + s₂²/n₂)**")
                        st.markdown(f"= ({x_bar1}-{x_bar2}) / √({s1**2:.4f}/{int(n1)} + {s2**2:.4f}/{int(n2)})")
                        st.markdown(f"= **{test_val:.4f}**")
                        st.markdown(f"df = min(n₁-1,n₂-1) = min({int(n1-1)},{int(n2-1)}) = **{df_tm}**")
                    if "Two" in h1_type:
                        st.markdown(f"t(α/2,{df_tm}) = **±{crit_tm:.4f}**")
                    elif ">" in h1_type:
                        st.markdown(f"t(α,{df_tm}) = **{crit_tm:.4f}**")
                    else:
                        st.markdown(f"t(α,{df_tm}) = **{crit_tm:.4f}**")
