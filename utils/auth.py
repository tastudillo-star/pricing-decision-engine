import streamlit as st

DEFAULT_APP_TITLE = "Pricing Decision Engine"
DEFAULT_LOGO_URL = "https://chiper.cl/wp-content/uploads/2023/06/cropped-favicon-192x192.png"


class Auth:
    # Allowlist CENTRAL dentro de la clase (categoría -> lista correos)
    ALLOWED_EMAILS = {
        "admin": ["tast@chiper.cl"],
        # "pricing_ops": ["ops1@empresa.cl", "ops2@empresa.cl"],
    }

    def __init__(
        self,
        app_title: str = DEFAULT_APP_TITLE,
        logo_url: str = DEFAULT_LOGO_URL,
        *,
        login_label: str = "Log in",
        logout_label: str = "Log out",
    ):
        self.app_title = app_title
        self.logo_url = logo_url
        self.login_label = login_label
        self.logout_label = logout_label

    def _sidebar_css(self) -> None:
        st.markdown(
            """
            <style>
              section[data-testid="stSidebar"] .stButton > button{
                padding: 0.35rem 0.60rem !important;
                min-height: 34px !important;
                font-size: 0.85rem !important;
                border-radius: 10px !important;
                margin: 0 !important;
              }

              .pde-chip{
                display:inline-flex; align-items:flex-start; gap:8px;
                padding:6px 10px;
                border:1px solid rgba(255,255,255,0.12);
                border-radius:999px;
                background:rgba(255,255,255,0.04);
                font-size:12px; line-height:1;
                margin-top:2px;
              }
              .pde-dot{
                width:8px; height:8px; border-radius:999px;
                background:#22c55e; display:inline-block; margin-top:2px;
              }
              .pde-dot.off{ background:rgba(255,255,255,0.30); }

              .pde-chip-text{
                display:flex; flex-direction:column;
                line-height:1.1;
              }
              .pde-name{ font-size:12px; }
              .pde-email{
                font-size:10px;
                color:#000 !important;
                margin-top:2px;
              }

              .pde-title{ font-weight:600; font-size:18px; line-height:1.15; margin:0; }

              .pde-blocked {
                border-radius: 14px;
                padding: 14px 16px;
                border: 1px solid rgba(245, 158, 11, 0.45);
                background: rgba(245, 158, 11, 0.14);
                margin: 10px 0 14px 0;
              }
              .pde-blocked.red {
                border: 1px solid rgba(239, 68, 68, 0.55);
                background: rgba(239, 68, 68, 0.14);
              }
              .pde-blocked-title {
                font-weight: 700;
                font-size: 15px;
                margin: 0 0 6px 0;
              }
              .pde-blocked-body {
                font-size: 13px;
                margin: 0;
                opacity: 0.95;
              }
              .pde-blocked-hint {
                font-size: 12px;
                margin: 8px 0 0 0;
                opacity: 0.85;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def sidebar(self) -> None:
        st.session_state.setdefault("_auth_action", None)

        with st.sidebar:
            self._sidebar_css()

            c1, c2 = st.columns([0.15, 0.85], vertical_alignment="center")
            with c1:
                try:
                    st.image(self.logo_url, width=30)
                except Exception:
                    pass
            with c2:
                st.markdown(f"<div class='pde-title'>{self.app_title}</div>", unsafe_allow_html=True)

            if not st.user.is_logged_in:
                st.markdown(
                    "<span class='pde-chip'><span class='pde-dot off'></span>Sesión: no iniciada</span>",
                    unsafe_allow_html=True,
                )
                if st.button(self.login_label, type="primary", use_container_width=True):
                    st.session_state["_auth_action"] = "login"
            else:
                name = (st.user.name or "").strip() or "Usuario"
                email = (getattr(st.user, "email", "") or "").strip()
                email_html = f"<div class='pde-email'>{email}</div>" if email else ""

                st.markdown(
                    f"""
                    <span class='pde-chip'>
                      <span class='pde-dot'></span>
                      <span class='pde-chip-text'>
                        <div class='pde-name'>Sesión: {name}</div>
                        {email_html}
                      </span>
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(self.logout_label, use_container_width=True):
                    st.session_state["_auth_action"] = "logout"

    def _handle_auth_actions(self) -> None:
        action = st.session_state.get("_auth_action")
        if not action:
            return

        st.session_state["_auth_action"] = None

        if action == "logout":
            st.logout()

        if action == "login":
            st.login()
            st.stop()

    def _require_category(self, category: str, *, strict: bool) -> None:
        email = (getattr(st.user, "email", "") or "").strip().lower()
        allowed = [e.strip().lower() for e in (self.ALLOWED_EMAILS.get(category) or [])]
        allowed_set = set(allowed)

        # Si la categoría no existe o está vacía:
        if not allowed_set:
            if strict:
                # estricto: bloquea
                st.markdown(
                    f"""
                    <div class="pde-blocked red">
                      <div class="pde-blocked-title">Acceso restringido</div>
                      <p class="pde-blocked-body">No hay allowlist configurada para esta página.</p>
                      <p class="pde-blocked-hint">Categoría: <b>{category}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.stop()
            # no estricto: deja pasar (fail-open)
            return

        # Si no hay email o no pertenece:
        if (not email) or (email not in allowed_set):
            st.markdown(
                f"""
                <div class="pde-blocked red">
                  <div class="pde-blocked-title">Acceso restringido</div>
                  <p class="pde-blocked-body">Tu cuenta no está autorizada para ver esta página.</p>
                  <p class="pde-blocked-hint">Tu correo es: <b>{email or "—"}</b> — Categoría autorizada: <b>{category}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.stop()

    def require_page(self, category: str | None = None, *, strict: bool = False) -> None:
        # 1) Render sidebar
        self.sidebar()

        # 2) Ejecutar acciones auth (antes de bloquear)
        self._handle_auth_actions()

        # 3) Gate login
        if not st.user.is_logged_in:
            st.markdown(
                """
                <div class="pde-blocked">
                  <div class="pde-blocked-title">Página bloqueada</div>
                  <p class="pde-blocked-body">Debes iniciar sesión para ver el contenido.</p>
                  <p class="pde-blocked-hint">Usa el botón <b>Log in</b> en la barra lateral. Y vuelve a
                  navegar a la página.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.stop()

        # 4) Gate autorización
        if category:
            self._require_category(category, strict=strict)
        else:
            # Estricto exige categoría
            if strict:
                st.markdown(
                    """
                    <div class="pde-blocked red">
                      <div class="pde-blocked-title">Acceso restringido</div>
                      <p class="pde-blocked-body">Esta página requiere categoría de acceso (modo estricto).</p>
                      <p class="pde-blocked-hint">Llama a <b>require_page("categoria", strict=True)</b>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.stop()

    @property
    def user(self):
        return st.user
