import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Métodos Numéricos", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;700;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Exo 2', sans-serif;
            font-size: 1.15rem;
            color: #0d2d6e !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #0d2d6e !important;
            font-weight: 900 !important;
        }

        h1 { font-size: 2.2rem !important; letter-spacing: 2px; }
        h2 { font-size: 1.8rem !important; }
        h3 { font-size: 1.4rem !important; }

        p, span, div, li, a {
            color: #0d2d6e !important;
        }

        /* Todos los textos de markdown */
        .stMarkdown, .stMarkdown p, .stMarkdown span,
        .stMarkdown div, [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span {
            color: #0d2d6e !important;
            font-weight: 600;
        }

        /* Labels de inputs */
        label, .stNumberInput label, .stTextInput label,
        .stSelectbox label, .stTextArea label {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            color: #0d2d6e !important;
        }

        /* Caption y texto pequeño */
        .stCaption, small {
            color: #1565C0 !important;
        }

        /* Expander */
        .streamlit-expanderHeader, [data-testid="stExpander"] summary {
            color: #0d2d6e !important;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
        }

        /* BOTONES GRANDES */
        div.stButton > button {
            width: 100%;
            padding: 1.2rem 2rem;
            font-size: 1.4rem !important;
            font-weight: 900 !important;
            font-family: 'Exo 2', sans-serif !important;
            letter-spacing: 2px;
            border-radius: 14px;
            border: none;
            background: linear-gradient(135deg, #1565C0, #0d47a1);
            color: white !important;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 24px rgba(21,101,192,0.4);
            margin-top: 0.8rem;
            text-transform: uppercase;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #1976D2, #1565C0);
            box-shadow: 0 8px 32px rgba(21,101,192,0.6);
            transform: translateY(-3px);
        }
        div.stButton > button p {
            color: white !important;
        }

        .stDataFrame { border-radius: 10px; font-size: 1.05rem; }
    </style>
""", unsafe_allow_html=True)

st.title("MÉTODOS NUMÉRICOS NANCY ANAKALY DELGADO GARCIA")

# ═══════════════════════════════════════════════════════
# BOTÓN IR A CONFIGURACIÓN
# ═══════════════════════════════════════════════════════
with st.expander("⚙️ ABRIR CONFIGURACIÓN", expanded=False):
    precision = st.number_input(
        "Decimales de precisión:",
        min_value=1, max_value=15, value=6, step=1
    )
    st.caption("Sintaxis Python/SymPy:")
    st.caption("`sin(x)`, `cos(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, `x**2`")

# También en sidebar por si acaso
with st.sidebar:
    st.header("Configuración")
    precision = st.number_input(
        "Decimales de precisión:",
        min_value=1, max_value=15, value=6, step=1,
        key="precision_sidebar"
    )
    st.markdown("---")
    st.caption("Sintaxis Python/SymPy:")
    st.caption("`sin(x)`, `cos(x)`, `exp(x)`, `log(x)`, `sqrt(x)`, `x**2`")

# ═══════════════════════════════════════════════════════
# SELECCIÓN DE MÉTODO CON BOTONES
# ═══════════════════════════════════════════════════════
st.markdown("### Selecciona el método:")

col1, col2, col3 = st.columns(3)

if "metodo" not in st.session_state:
    st.session_state.metodo = "Euler Mejorado"

with col1:
    if st.button("EULER MEJORADO", key="btn_euler"):
        st.session_state.metodo = "Euler Mejorado"
with col2:
    if st.button("RUNGE-KUTTA 4", key="btn_rk4"):
        st.session_state.metodo = "Runge-Kutta 4"
with col3:
    if st.button("NEWTON-RAPHSON", key="btn_newton"):
        st.session_state.metodo = "Newton-Raphson"

metodo = st.session_state.metodo
st.markdown(f"**Método activo: `{metodo}`**")
st.markdown("---")

# ═══════════════════════════════════════════════════════
# EULER MEJORADO
# ═══════════════════════════════════════════════════════
if metodo == "Euler Mejorado":

    st.header("Euler Mejorado (Heun)")
    st.markdown(r"""
    $$k_1 = f(x_n,\ y_n) \qquad k_2 = f(x_n + h,\ y_n + h \cdot k_1) \qquad y_{n+1} = y_n + \frac{h}{2}(k_1 + k_2)$$
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>f(x, y):</span>", unsafe_allow_html=True)
        funcion_input = st.text_input("", "x + y", key="euler_fxy", label_visibility="collapsed")
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>x₀ (valor inicial de x):</span>", unsafe_allow_html=True)
        x0_orig = st.number_input("", value=0.0, format="%.6f", key="euler_x0", label_visibility="collapsed")
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>y₀ (valor inicial de y):</span>", unsafe_allow_html=True)
        y0_orig = st.number_input("", value=1.0, format="%.6f", key="euler_y0", label_visibility="collapsed")
    with col2:
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>x final:</span>", unsafe_allow_html=True)
        xf = st.number_input("", value=1.0, format="%.6f", key="euler_xf", label_visibility="collapsed")
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>h (tamaño de paso):</span>", unsafe_allow_html=True)
        h  = st.number_input("", value=0.1, min_value=1e-10, format="%.6f", key="euler_h", label_visibility="collapsed")

    try:
        x_sym, y_sym = sp.symbols('x y')
        f_sym = sp.sympify(funcion_input)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**f(x, y):**")
            st.latex(sp.latex(f_sym))
        with col2:
            st.markdown("**∂f/∂y  (derivada automática):**")
            st.latex(sp.latex(sp.diff(f_sym, y_sym)))
    except Exception:
        pass

    if st.button("CALCULAR EULER MEJORADO", key="calc_euler"):
        try:
            if xf <= x0_orig:
                st.error("❌ x final debe ser mayor que x₀")
            else:
                x_sym, y_sym = sp.symbols('x y')
                f_sym = sp.sympify(funcion_input)
                f = sp.lambdify((x_sym, y_sym), f_sym, "numpy")

                n = int(round((xf - x0_orig) / h))
                x0, y0 = x0_orig, y0_orig

                x_vals  = [round(x0, int(precision))]
                y_vals  = [round(y0, int(precision))]
                k1_vals = []
                k2_vals = []

                for i in range(n):
                    k1    = float(f(x0, y0))
                    k2    = float(f(x0 + h, y0 + h * k1))
                    y_new = y0 + (h / 2) * (k1 + k2)

                    k1_vals.append(round(k1, int(precision)))
                    k2_vals.append(round(k2, int(precision)))

                    x0 = x0 + h
                    y0 = y_new

                    x_vals.append(round(x0, int(precision)))
                    y_vals.append(round(y0, int(precision)))

                st.success(f"✅ {n} pasos  |  x: {x0_orig} → {round(x0, int(precision))}")

                tabla = pd.DataFrame({
                    "i":                      range(len(x_vals)),
                    "xᵢ":                     x_vals,
                    "yᵢ":                     y_vals,
                    "k1 (pendiente inicial)": ["-"] + k1_vals,
                    "k2 (pendiente final)":   ["-"] + k2_vals,
                })
                st.subheader("Tabla de valores")
                st.dataframe(tabla, use_container_width=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines+markers',
                    name='Euler Mejorado',
                    line=dict(color='#00d4ff', width=2),
                    marker=dict(size=7, color='#ff6b6b')
                ))
                fig.update_layout(
                    title="Solución numérica — Euler Mejorado",
                    xaxis_title="x", yaxis_title="y(x)",
                    template="plotly_dark", hovermode="x unified"
                )
                st.subheader("Gráfica 2D")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════
# RUNGE-KUTTA 4
# ═══════════════════════════════════════════════════════
if metodo == "Runge-Kutta 4":

    st.header("Runge-Kutta de Orden 4 (RK4)")
    st.markdown(r"""
    $$k_1 = f(x_n,\ y_n) \quad k_2 = f\!\left(x_n+\tfrac{h}{2},\ y_n+\tfrac{h}{2}k_1\right) \quad k_3 = f\!\left(x_n+\tfrac{h}{2},\ y_n+\tfrac{h}{2}k_2\right) \quad k_4 = f(x_n+h,\ y_n+hk_3)$$
    $$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>f(x, y):</span>", unsafe_allow_html=True)
        funcion_input = st.text_input("", "x + y", key="rk4_fxy", label_visibility="collapsed")
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>x₀ (valor inicial de x):</span>", unsafe_allow_html=True)
        x0_orig = st.number_input("", value=0.0, format="%.6f", key="rk4_x0", label_visibility="collapsed")
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>y₀ (valor inicial de y):</span>", unsafe_allow_html=True)
        y0_orig = st.number_input("", value=1.0, format="%.6f", key="rk4_y0", label_visibility="collapsed")
    with col2:
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>x final:</span>", unsafe_allow_html=True)
        xf = st.number_input("", value=1.0, format="%.6f", key="rk4_xf", label_visibility="collapsed")
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>h (tamaño de paso):</span>", unsafe_allow_html=True)
        h  = st.number_input("", value=0.1, min_value=1e-10, format="%.6f", key="rk4_h", label_visibility="collapsed")

    try:
        x_sym, y_sym = sp.symbols('x y')
        f_sym = sp.sympify(funcion_input)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**f(x, y):**")
            st.latex(sp.latex(f_sym))
        with col2:
            st.markdown("**∂f/∂y  (derivada automática):**")
            st.latex(sp.latex(sp.diff(f_sym, y_sym)))
    except Exception:
        pass

    if st.button("CALCULAR RUNGE-KUTTA 4", key="calc_rk4"):
        try:
            if xf <= x0_orig:
                st.error("❌ x final debe ser mayor que x₀")
            else:
                x_sym, y_sym = sp.symbols('x y')
                f_sym = sp.sympify(funcion_input)
                f = sp.lambdify((x_sym, y_sym), f_sym, "numpy")

                n = int(round((xf - x0_orig) / h))
                x0, y0 = x0_orig, y0_orig

                x_vals  = [round(x0, int(precision))]
                y_vals  = [round(y0, int(precision))]
                k1_vals = []
                k2_vals = []
                k3_vals = []
                k4_vals = []

                for i in range(n):
                    k1    = float(f(x0,       y0))
                    k2    = float(f(x0 + h/2, y0 + h*k1/2))
                    k3    = float(f(x0 + h/2, y0 + h*k2/2))
                    k4    = float(f(x0 + h,   y0 + h*k3))
                    y_new = y0 + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

                    k1_vals.append(round(k1, int(precision)))
                    k2_vals.append(round(k2, int(precision)))
                    k3_vals.append(round(k3, int(precision)))
                    k4_vals.append(round(k4, int(precision)))

                    x0 = x0 + h
                    y0 = y_new

                    x_vals.append(round(x0, int(precision)))
                    y_vals.append(round(y0, int(precision)))

                st.success(f"✅ {n} pasos  |  x: {x0_orig} → {round(x0, int(precision))}")

                tabla = pd.DataFrame({
                    "i":   range(len(x_vals)),
                    "xᵢ":  x_vals,
                    "yᵢ":  y_vals,
                    "k1":  ["-"] + k1_vals,
                    "k2":  ["-"] + k2_vals,
                    "k3":  ["-"] + k3_vals,
                    "k4":  ["-"] + k4_vals,
                })
                st.subheader("Tabla de valores")
                st.dataframe(tabla, use_container_width=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines+markers',
                    name='Runge-Kutta 4',
                    line=dict(color='#a855f7', width=2),
                    marker=dict(size=7, color='#fbbf24')
                ))
                fig.update_layout(
                    title="Solución numérica — Runge-Kutta 4",
                    xaxis_title="x", yaxis_title="y(x)",
                    template="plotly_dark", hovermode="x unified"
                )
                st.subheader("Gráfica 2D")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════
# NEWTON-RAPHSON
# ═══════════════════════════════════════════════════════
if metodo == "Newton-Raphson":

    st.header("Newton-Raphson")
    st.markdown(r"""
    $$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>f(x):</span>", unsafe_allow_html=True)
        funcion_input = st.text_input("", "x**3 - x - 2", key="nr_fx", label_visibility="collapsed")
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>x₀ (valor inicial):</span>", unsafe_allow_html=True)
        x0_orig = st.number_input("", value=1.5, format="%.6f", key="nr_x0", label_visibility="collapsed")
    with col2:
        st.markdown("<span style='color:#1565C0;font-weight:bold;font-size:1.1rem'>x final (tolerancia):</span>", unsafe_allow_html=True)
        xf_tol = st.number_input(
            "", value=1e-6, format="%.2e", min_value=1e-15, max_value=1.0,
            key="nr_tol", label_visibility="collapsed"
        )

    try:
        x_sym  = sp.symbols('x')
        f_sym  = sp.sympify(funcion_input)
        df_sym = sp.diff(f_sym, x_sym)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**f(x):**")
            st.latex(sp.latex(f_sym))
        with col2:
            st.markdown("**f\'(x)  (derivada automática):**")
            st.latex(sp.latex(df_sym))
    except Exception:
        pass

    if st.button("CALCULAR NEWTON-RAPHSON", key="calc_newton"):
        try:
            x_sym  = sp.symbols('x')
            f_sym  = sp.sympify(funcion_input)
            df_sym = sp.diff(f_sym, x_sym)

            f  = sp.lambdify(x_sym, f_sym,  "numpy")
            df = sp.lambdify(x_sym, df_sym, "numpy")

            x0         = x0_orig
            pasos      = []
            x_vals     = []
            fx_vals    = []
            dfx_vals   = []
            error_vals = []
            convergio  = False

            for i in range(1000):   # límite interno de seguridad
                fx  = float(f(x0))
                dfx = float(df(x0))

                if abs(dfx) < 1e-15:
                    st.warning("⚠️ Derivada cercana a cero — el método no puede continuar.")
                    break

                x1  = x0 - fx / dfx
                err = abs(x1 - x0)

                pasos.append(i + 1)
                x_vals.append(round(x0, int(precision)))
                fx_vals.append(round(fx, int(precision)))
                dfx_vals.append(round(dfx, int(precision)))
                error_vals.append(f"{err:.{int(precision)}e}")

                x0 = x1

                if err < xf_tol:
                    convergio = True
                    break

            if convergio:
                st.success(f"✅ Convergió en {len(pasos)} pasos  |  Raíz ≈ {round(x0, int(precision))}")
            else:
                st.warning(f"⚠️ No convergió. Última aproximación: {round(x0, int(precision))}")

            tabla = pd.DataFrame({
                "Paso":       pasos,
                "xₙ":         x_vals,
                "f(xₙ)":      fx_vals,
                "f\'(xₙ)":    dfx_vals,
                "|xₙ₊₁-xₙ|": error_vals,
            })
            st.subheader("Tabla de pasos")
            st.dataframe(tabla, use_container_width=True)

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=pasos, y=x_vals,
                mode='lines+markers',
                name='xₙ',
                line=dict(color='#00d4ff', width=2),
                marker=dict(size=8, color='#ff6b6b')
            ))
            fig_conv.add_hline(
                y=round(x0, int(precision)),
                line_dash="dot", line_color="#ffd700",
                annotation_text=f"Raíz ≈ {round(x0, int(precision))}",
                annotation_position="bottom right"
            )
            fig_conv.update_layout(
                title="Convergencia de xₙ",
                xaxis_title="Paso", yaxis_title="xₙ",
                template="plotly_dark", hovermode="x unified"
            )

            x_plot = np.linspace(x0_orig - 3, x0_orig + 3, 500)
            try:
                y_plot = np.clip(np.array([float(f(xi)) for xi in x_plot]), -1e6, 1e6)
                fig_fx = go.Figure()
                fig_fx.add_trace(go.Scatter(
                    x=x_plot, y=y_plot,
                    mode='lines', name='f(x)',
                    line=dict(color='#ffd700', width=2)
                ))
                fig_fx.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
                fig_fx.add_trace(go.Scatter(
                    x=[round(x0, int(precision))], y=[0],
                    mode='markers',
                    name=f'Raíz ≈ {round(x0, int(precision))}',
                    marker=dict(size=14, color='#ff6b6b', symbol='x')
                ))
                fig_fx.update_layout(
                    title="f(x) y raíz encontrada",
                    xaxis_title="x", yaxis_title="f(x)",
                    template="plotly_dark"
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Convergencia")
                    st.plotly_chart(fig_conv, use_container_width=True)
                with col2:
                    st.subheader("f(x) y raíz")
                    st.plotly_chart(fig_fx, use_container_width=True)
            except Exception:
                st.subheader("Convergencia")
                st.plotly_chart(fig_conv, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
