import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Métodos Numéricos NANCY ANAKALY DELGADO GARCIA", layout="wide")

st.markdown("""
    <style>
        .main { background-color: #0f1117; }
        h1, h2, h3 { color: #00d4ff; }
        .stDataFrame { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("Métodos Numéricos NANCY ANAKALY DELGADO GARCIA")

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.header("Configuración")
    metodo = st.selectbox(
        "Método numérico:",
        ["Euler Mejorado", "Runge-Kutta 4", "Newton-Raphson"]
    )
    precision = st.number_input(
        "Decimales de precisión:",
        min_value=1, max_value=15, value=6, step=1
    )
    st.markdown("---")
    st.caption("Usa sintaxis Python/SymPy para las funciones.")
    st.caption("Ejemplos: `sin(x)*y`, `x**2 + y`, `exp(x) - y`")

# ═══════════════════════════════════════════════════════
# EULER MEJORADO
# ═══════════════════════════════════════════════════════
if metodo == "Euler Mejorado":

    st.subheader("Euler Mejorado (Heun)")
    st.markdown(r"""
    **Fórmulas:**
    $$k_1 = f(x_n,\ y_n)$$
    $$k_2 = f(x_n + h,\ y_n + h \cdot k_1)$$
    $$y_{n+1} = y_n + \frac{h}{2}(k_1 + k_2)$$
    """)

    col1, col2 = st.columns(2)
    with col1:
        funcion_input = st.text_input("Ingresa f(x, y):", "x + y")
        x0_orig = st.number_input("Valor inicial x₀:", value=0.0, format="%.6f")
        y0_orig = st.number_input("Valor inicial y₀:", value=1.0, format="%.6f")
    with col2:
        h = st.number_input("Tamaño de paso h:", value=0.1, min_value=1e-10, format="%.6f")
        n = st.number_input("Número de pasos n:", value=10, step=1, min_value=1)

    # Derivadas parciales automáticas (se muestran al escribir la función)
    try:
        x_sym, y_sym = sp.symbols('x y')
        f_sym = sp.sympify(funcion_input)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**f(x, y):**")
            st.latex(sp.latex(f_sym))
        with col2:
            st.markdown("**∂f/∂y (para verificar linealidad):**")
            st.latex(sp.latex(sp.diff(f_sym, y_sym)))
    except Exception:
        pass

    if st.button("▶ Calcular", key="euler"):
        try:
            x_sym, y_sym = sp.symbols('x y')
            f_sym = sp.sympify(funcion_input)
            f = sp.lambdify((x_sym, y_sym), f_sym, "numpy")

            x0, y0 = x0_orig, y0_orig

            # Inicializar listas con condición inicial
            x_vals  = [round(x0, int(precision))]
            y_vals  = [round(y0, int(precision))]
            k1_vals = []
            k2_vals = []

            # ── Loop Euler Mejorado ──────────────────────
            for i in range(int(n)):
                k1 = float(f(x0, y0))
                k2 = float(f(x0 + h, y0 + h * k1))
                y_new = y0 + (h / 2) * (k1 + k2)

                k1_vals.append(round(k1, int(precision)))
                k2_vals.append(round(k2, int(precision)))

                x0 = x0 + h
                y0 = y_new

                x_vals.append(round(x0, int(precision)))
                y_vals.append(round(y0, int(precision)))

            # ── Tabla ────────────────────────────────────
            tabla = pd.DataFrame({
                "i":                      range(len(x_vals)),
                "xᵢ":                     x_vals,
                "yᵢ":                     y_vals,
                "k1 (pendiente inicial)": ["-"] + k1_vals,
                "k2 (pendiente final)":   ["-"] + k2_vals,
            })

            st.subheader("Tabla de valores")
            st.dataframe(tabla, use_container_width=True)

            # ── Gráfica ───────────────────────────────────
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines+markers',
                name='Euler Mejorado',
                line=dict(color='#00d4ff', width=2),
                marker=dict(size=6, color='#ff6b6b')
            ))
            fig.update_layout(
                title="Euler Mejorado",
                xaxis_title="x",
                yaxis_title="y(x)",
                template="plotly_dark",
                hovermode="x unified"
            )
            st.subheader("Gráfica 2D")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error al calcular: {e}")


# ═══════════════════════════════════════════════════════
# RUNGE-KUTTA 4
# ═══════════════════════════════════════════════════════
if metodo == "Runge-Kutta 4":

    st.subheader("Runge-Kutta de Orden 4 (RK4)")
    st.markdown(r"""
    **Fórmulas:**
    $$k_1 = f(x_n,\ y_n)$$
    $$k_2 = f\!\left(x_n+\tfrac{h}{2},\ y_n+\tfrac{h}{2}k_1\right)$$
    $$k_3 = f\!\left(x_n+\tfrac{h}{2},\ y_n+\tfrac{h}{2}k_2\right)$$
    $$k_4 = f(x_n+h,\ y_n+h\,k_3)$$
    $$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
    """)

    col1, col2 = st.columns(2)
    with col1:
        funcion_input = st.text_input("Ingresa f(x, y):", "x + y")
        x0_orig = st.number_input("Valor inicial x₀:", value=0.0, format="%.6f")
        y0_orig = st.number_input("Valor inicial y₀:", value=1.0, format="%.6f")
    with col2:
        h = st.number_input("Tamaño de paso h:", value=0.1, min_value=1e-10, format="%.6f")
        n = st.number_input("Número de pasos n:", value=10, step=1, min_value=1)

    # Mostrar f(x,y) automáticamente
    try:
        x_sym, y_sym = sp.symbols('x y')
        f_sym = sp.sympify(funcion_input)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**f(x, y):**")
            st.latex(sp.latex(f_sym))
        with col2:
            st.markdown("**∂f/∂y:**")
            st.latex(sp.latex(sp.diff(f_sym, y_sym)))
    except Exception:
        pass

    if st.button("▶ Calcular", key="rk4"):
        try:
            x_sym, y_sym = sp.symbols('x y')
            f_sym = sp.sympify(funcion_input)
            f = sp.lambdify((x_sym, y_sym), f_sym, "numpy")

            x0, y0 = x0_orig, y0_orig

            # Inicializar listas con condición inicial
            x_vals  = [round(x0, int(precision))]
            y_vals  = [round(y0, int(precision))]
            k1_vals = []
            k2_vals = []
            k3_vals = []
            k4_vals = []

            # ── Loop RK4 ─────────────────────────────────
            for i in range(int(n)):
                k1 = float(f(x0,           y0))
                k2 = float(f(x0 + h/2,     y0 + h*k1/2))
                k3 = float(f(x0 + h/2,     y0 + h*k2/2))
                k4 = float(f(x0 + h,       y0 + h*k3))
                y_new = y0 + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

                k1_vals.append(round(k1, int(precision)))
                k2_vals.append(round(k2, int(precision)))
                k3_vals.append(round(k3, int(precision)))
                k4_vals.append(round(k4, int(precision)))

                x0 = x0 + h
                y0 = y_new

                x_vals.append(round(x0, int(precision)))
                y_vals.append(round(y0, int(precision)))

            # ── Tabla ─────────────────────────────────────
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

            # ── Gráfica ───────────────────────────────────
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines+markers',
                name='Runge-Kutta 4',
                line=dict(color='#a855f7', width=2),
                marker=dict(size=6, color='#fbbf24')
            ))
            fig.update_layout(
                title="Solución numérica — Runge-Kutta 4",
                xaxis_title="x",
                yaxis_title="y(x)",
                template="plotly_dark",
                hovermode="x unified"
            )
            st.subheader("Gráfica 2D")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error al calcular: {e}")


# ═══════════════════════════════════════════════════════
# NEWTON-RAPHSON
# ═══════════════════════════════════════════════════════
if metodo == "Newton-Raphson":

    st.subheader("Newton-Raphson")
    st.markdown(r"""
    **Fórmula de iteración:**
    $$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$
    """)

    col1, col2 = st.columns(2)
    with col1:
        funcion_input = st.text_input("Ingresa f(x):", "x**3 - x - 2")
        x0_orig = st.number_input("Valor inicial x₀:", value=1.5, format="%.6f")
    with col2:
        n = st.number_input("Número de pasos:", value=10, step=1, min_value=1)

    # Mostrar f(x) y f'(x) automáticamente
    try:
        x_sym = sp.symbols('x')
        f_sym = sp.sympify(funcion_input)
        df_sym = sp.diff(f_sym, x_sym)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**f(x):**")
            st.latex(sp.latex(f_sym))
        with col2:
            st.markdown("**f\'(x) — derivada automática:**")
            st.latex(sp.latex(df_sym))
    except Exception:
        pass

    if st.button("▶ Calcular", key="newton"):
        try:
            x_sym = sp.symbols('x')
            f_sym  = sp.sympify(funcion_input)
            df_sym = sp.diff(f_sym, x_sym)

            f  = sp.lambdify(x_sym, f_sym,  "numpy")
            df = sp.lambdify(x_sym, df_sym, "numpy")

            x0 = x0_orig
            iteraciones = []
            x_vals      = []
            fx_vals     = []
            dfx_vals    = []
            error_vals  = []

            # ── Loop Newton-Raphson ──────────────────────
            for i in range(int(n)):
                fx  = float(f(x0))
                dfx = float(df(x0))

                if abs(dfx) < 1e-15:
                    st.warning("⚠️ Derivada cercana a cero — el método no puede continuar.")
                    break

                x1  = x0 - fx / dfx
                err = abs(x1 - x0)

                iteraciones.append(i + 1)
                x_vals.append(round(x0, int(precision)))
                fx_vals.append(round(fx, int(precision)))
                dfx_vals.append(round(dfx, int(precision)))
                error_vals.append(f"{err:.{int(precision)}e}")

                x0 = x1

            st.success(f"Raíz aproximada: {round(x0, int(precision))} (después de {len(iteraciones)} pasos)")

            # ── Tabla ─────────────────────────────────────
            tabla = pd.DataFrame({
                "n":        iteraciones,
                "xₙ":       x_vals,
                "f(xₙ)":    fx_vals,
                "f\'(xₙ)":  dfx_vals,
                "|xₙ₊₁-xₙ|": error_vals,
            })

            st.subheader("Tabla de iteraciones")
            st.dataframe(tabla, use_container_width=True)

            # ── Gráfica 1: Convergencia de xₙ ────────────
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=iteraciones, y=x_vals,
                mode='lines+markers',
                name='xₙ',
                line=dict(color='#00d4ff', width=2),
                marker=dict(size=7, color='#ff6b6b')
            ))
            fig_conv.add_hline(
                y=round(x0, int(precision)),
                line_dash="dot", line_color="#ffd700",
                annotation_text=f"Raíz ≈ {round(x0, int(precision))}",
                annotation_position="bottom right"
            )
            fig_conv.update_layout(
                title="Convergencia de xₙ por iteración",
                xaxis_title="Iteración n",
                yaxis_title="xₙ",
                template="plotly_dark",
                hovermode="x unified"
            )

            # ── Gráfica 2: f(x) con la raíz marcada ──────
            x_plot = np.linspace(x0_orig - 3, x0_orig + 3, 500)
            try:
                y_plot = np.array([float(f(xi)) for xi in x_plot])
                # Recortar valores muy grandes para que la gráfica sea legible
                y_plot = np.clip(y_plot, -1e6, 1e6)

                fig_fx = go.Figure()
                fig_fx.add_trace(go.Scatter(
                    x=x_plot, y=y_plot,
                    mode='lines', name='f(x)',
                    line=dict(color='#ffd700', width=2)
                ))
                fig_fx.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
                fig_fx.add_trace(go.Scatter(
                    x=[round(x0, int(precision))], y=[0],
                    mode='markers', name=f'Raíz ≈ {round(x0, int(precision))}',
                    marker=dict(size=14, color='#ff6b6b', symbol='x')
                ))
                fig_fx.update_layout(
                    title="Gráfica de f(x) y raíz encontrada",
                    xaxis_title="x",
                    yaxis_title="f(x)",
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
            st.error(f"Error al calcular: {e}")



